//! Delta writers handle row-level changes by combining data file and delete file writers.
//! The delta writer has three sub-writers:
//! - A data file writer for new and updated rows.
//! - A position delete file writer for deletions of existing rows (that have been written within this writer)
//! - An equality delete file writer for deletions of rows based on equality conditions (for rows that may exist in other data files).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::BooleanBuilder;
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_ord::partition::partition;
use arrow_row::{OwnedRow, RowConverter, Rows, SortField};
use arrow_select::filter::filter_record_batch;
use itertools::Itertools;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::record_batch_projector::RecordBatchProjector;
use crate::arrow::schema_to_arrow_schema;
use crate::spec::DataFile;
use crate::writer::base_writer::position_delete_writer::PositionDeleteWriterConfig;
use crate::writer::{CurrentFileStatus, IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// A builder for `DeltaWriter`.
#[derive(Clone, Debug)]
pub struct DeltaWriterBuilder<DWB, PDWB, EDWB> {
    data_writer_builder: DWB,
    pos_delete_writer_builder: PDWB,
    eq_delete_writer_builder: EDWB,
    unique_cols: Vec<i32>,
}

impl<DWB, PDWB, EDWB> DeltaWriterBuilder<DWB, PDWB, EDWB> {
    /// Creates a new `DeltaWriterBuilder`.
    pub fn new(
        data_writer_builder: DWB,
        pos_delete_writer_builder: PDWB,
        eq_delete_writer_builder: EDWB,
        unique_cols: Vec<i32>,
    ) -> Self {
        Self {
            data_writer_builder,
            pos_delete_writer_builder,
            eq_delete_writer_builder,
            unique_cols,
        }
    }
}

#[async_trait::async_trait]
impl<DWB, PDWB, EDWB> IcebergWriterBuilder for DeltaWriterBuilder<DWB, PDWB, EDWB>
where
    DWB: IcebergWriterBuilder,
    PDWB: IcebergWriterBuilder,
    EDWB: IcebergWriterBuilder,
    DWB::R: CurrentFileStatus,
{
    type R = DeltaWriter<DWB::R, PDWB::R, EDWB::R>;
    async fn build(self) -> Result<Self::R> {
        let data_writer = self.data_writer_builder.build().await?;
        let pos_delete_writer = self.pos_delete_writer_builder.build().await?;
        let eq_delete_writer = self.eq_delete_writer_builder.build().await?;
        DeltaWriter::try_new(
            data_writer,
            pos_delete_writer,
            eq_delete_writer,
            self.unique_cols,
        )
    }
}

/// Position information of a row in a data file.
pub struct Position {
    row_index: i64,
    file_path: String,
}

/// A writer that handles row-level changes by combining data file and delete file writers.
pub struct DeltaWriter<DW, PDW, EDW> {
    /// The data file writer for new and updated rows.
    pub data_writer: DW,
    /// The position delete file writer for deletions of existing rows (that have been written within
    /// this writer).
    pub pos_delete_writer: PDW,
    /// The equality delete file writer for deletions of rows based on equality conditions (for rows
    /// that may exist in other data files).
    pub eq_delete_writer: EDW,
    /// The list of unique columns used for equality deletes.
    pub unique_cols: Vec<i32>,
    /// A map of rows (projected to unique columns) to their corresponding position information.
    pub seen_rows: HashMap<OwnedRow, Position>,
    /// A projector to project the record batch to the unique columns.
    pub(crate) projector: RecordBatchProjector,
    /// A converter to convert the projected columns to rows for easy comparison.
    pub(crate) row_convertor: RowConverter,
}

impl<DW, PDW, EDW> DeltaWriter<DW, PDW, EDW>
where
    DW: IcebergWriter + CurrentFileStatus,
    PDW: IcebergWriter,
    EDW: IcebergWriter,
{
    fn try_new(
        data_writer: DW,
        pos_delete_writer: PDW,
        eq_delete_writer: EDW,
        unique_cols: Vec<i32>,
    ) -> Result<Self> {
        let arrow_schema = Arc::new(schema_to_arrow_schema(&data_writer.current_schema())?);
        let projector = RecordBatchProjector::new(
            arrow_schema,
            &unique_cols,
            |field| {
                if field.data_type().is_nested() {
                    return Ok(None);
                }
                field
                    .metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .map(|id_str| {
                        id_str.parse::<i64>().map_err(|e| {
                            Error::new(
                                ErrorKind::Unexpected,
                                format!("Failed to parse field ID {}: {}", id_str, e),
                            )
                        })
                    })
                    .transpose()
            },
            |_| false,
        )?;

        let row_convertor = RowConverter::new(
            projector
                .projected_schema_ref()
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;

        Ok(Self {
            data_writer,
            pos_delete_writer,
            eq_delete_writer,
            unique_cols,
            seen_rows: HashMap::new(),
            projector,
            row_convertor,
        })
    }

    async fn insert(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = self.extract_unique_column_rows(&batch)?;
        let file_path = self.data_writer.current_file_path();
        let start_row_index = self.data_writer.current_row_num();

        self.data_writer.write(batch.clone()).await?;

        for (i, row) in rows.iter().enumerate() {
            self.seen_rows.insert(row.owned(), Position {
                row_index: start_row_index as i64 + i as i64,
                file_path: file_path.clone(),
            });
        }

        Ok(())
    }

    async fn delete(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = self.extract_unique_column_rows(&batch)?;
        let mut file_array = vec![];
        let mut row_index_array = vec![];
        // Build a boolean array to track which rows need equality deletes.
        // True = row not seen before, needs equality delete
        // False = row was seen, already handled via position delete
        let mut needs_equality_delete = BooleanBuilder::new();

        for row in rows.iter() {
            if let Some(pos) = self.seen_rows.remove(&row.owned()) {
                // Row was previously inserted, use position delete
                row_index_array.push(pos.row_index);
                file_array.push(pos.file_path.clone());
                needs_equality_delete.append_value(false);
            } else {
                // Row not seen before, use equality delete
                needs_equality_delete.append_value(true);
            }
        }

        // Write position deletes for rows that were previously inserted
        let file_array: ArrayRef = Arc::new(StringArray::from(file_array));
        let row_index_array: ArrayRef = Arc::new(arrow_array::Int64Array::from(row_index_array));

        let position_batch =
            RecordBatch::try_new(PositionDeleteWriterConfig::arrow_schema(), vec![
                file_array,
                row_index_array,
            ])?;

        if position_batch.num_rows() > 0 {
            self.pos_delete_writer
                .write(position_batch)
                .await
                .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")))?;
        }

        // Write equality deletes for rows that were not previously inserted
        let eq_batch = filter_record_batch(&batch, &needs_equality_delete.finish())
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")))?;

        if eq_batch.num_rows() > 0 {
            self.eq_delete_writer
                .write(eq_batch)
                .await
                .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")))?;
        }

        Ok(())
    }

    fn extract_unique_column_rows(&mut self, batch: &RecordBatch) -> Result<Rows> {
        self.row_convertor
            .convert_columns(&self.projector.project_column(batch.columns())?)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")))
    }
}

#[async_trait::async_trait]
impl<DW, PDW, EDW> IcebergWriter for DeltaWriter<DW, PDW, EDW>
where
    DW: IcebergWriter + CurrentFileStatus,
    PDW: IcebergWriter,
    EDW: IcebergWriter,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        // Treat the last row as an op indicator +1 for insert, -1 for delete
        let ops = batch
            .column(batch.num_columns() - 1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(Error::new(
                ErrorKind::Unexpected,
                "Failed to downcast ops column",
            ))?;

        let partition =
            partition(&[batch.column(batch.num_columns() - 1).clone()]).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to partition batch: {e}"),
                )
            })?;

        for range in partition.ranges() {
            let batch = batch
                .project(&(0..batch.num_columns() - 1).collect_vec())
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!("Failed to project batch columns: {e}"),
                    )
                })?
                .slice(range.start, range.end - range.start);
            match ops.value(range.start) {
                1 => self.insert(batch).await?,
                -1 => self.delete(batch).await?,
                op => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("Ops column must be 1 (insert) or -1 (delete), not {op}"),
                    ));
                }
            }
        }

        Ok(())
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        let data_files = self.data_writer.close().await?;
        let pos_delete_files = self.pos_delete_writer.close().await?;
        let eq_delete_files = self.eq_delete_writer.close().await?;

        Ok(data_files
            .into_iter()
            .chain(pos_delete_files)
            .chain(eq_delete_files)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    mod delta_writer_tests {
        use std::collections::HashMap;
        use std::sync::Arc;

        use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;
        use tempfile::TempDir;

        use crate::Result;
        use crate::arrow::arrow_schema_to_schema;
        use crate::io::FileIOBuilder;
        use crate::spec::{
            DataFileFormat, NestedField, PrimitiveType, Schema as IcebergSchema, Type,
        };
        use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
        use crate::writer::base_writer::equality_delete_writer::{
            EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
        };
        use crate::writer::base_writer::position_delete_writer::{
            PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
        };
        use crate::writer::combined_writer::delta_writer::DeltaWriter;
        use crate::writer::file_writer::ParquetWriterBuilder;
        use crate::writer::file_writer::location_generator::{
            DefaultFileNameGenerator, DefaultLocationGenerator,
        };
        use crate::writer::{IcebergWriter, IcebergWriterBuilder};

        fn create_iceberg_schema() -> Arc<IcebergSchema> {
            Arc::new(
                IcebergSchema::builder()
                    .with_schema_id(0)
                    .with_fields(vec![
                        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                            .into(),
                    ])
                    .build()
                    .unwrap(),
            )
        }

        fn create_test_batch_with_ops(
            ids: Vec<i32>,
            names: Vec<Option<&str>>,
            ops: Vec<i32>,
        ) -> RecordBatch {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "1".to_string(),
                )])),
                Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "2".to_string(),
                )])),
                Field::new("op", DataType::Int32, false),
            ]));

            let id_array: ArrayRef = Arc::new(Int32Array::from(ids));
            let name_array: ArrayRef = Arc::new(StringArray::from(names));
            let op_array: ArrayRef = Arc::new(Int32Array::from(ops));

            RecordBatch::try_new(schema, vec![id_array, name_array, op_array]).unwrap()
        }

        #[tokio::test]
        async fn test_delta_writer_insert_only() -> Result<()> {
            let temp_dir = TempDir::new().unwrap();
            let file_io = FileIOBuilder::new_fs_io().build().unwrap();
            let schema = create_iceberg_schema();

            // Create data writer
            let data_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/data",
                temp_dir.path().to_str().unwrap()
            ));
            let data_file_name_gen =
                DefaultFileNameGenerator::new("data".to_string(), None, DataFileFormat::Parquet);
            let data_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                schema.clone(),
                None,
                file_io.clone(),
                data_location_gen,
                data_file_name_gen,
            );
            let data_writer = DataFileWriterBuilder::new(data_parquet_writer, None, 0);

            // Create position delete writer
            let pos_delete_schema = Arc::new(arrow_schema_to_schema(
                &PositionDeleteWriterConfig::arrow_schema(),
            )?);
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/pos_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let pos_delete_file_name_gen = DefaultFileNameGenerator::new(
                "pos_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let pos_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                pos_delete_schema,
                None,
                file_io.clone(),
                pos_delete_location_gen,
                pos_delete_file_name_gen,
            );
            let pos_delete_writer = PositionDeleteFileWriterBuilder::new(
                pos_delete_parquet_writer,
                PositionDeleteWriterConfig::new(None, 0, None),
            );

            // Create equality delete writer
            let eq_delete_config =
                EqualityDeleteWriterConfig::new(vec![1], schema.clone(), None, 0)?;
            let eq_delete_schema = Arc::new(arrow_schema_to_schema(
                eq_delete_config.projected_arrow_schema_ref(),
            )?);
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/eq_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let eq_delete_file_name_gen = DefaultFileNameGenerator::new(
                "eq_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let eq_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                eq_delete_schema,
                None,
                file_io.clone(),
                eq_delete_location_gen,
                eq_delete_file_name_gen,
            );
            let eq_delete_writer =
                EqualityDeleteFileWriterBuilder::new(eq_delete_parquet_writer, eq_delete_config);

            // Create delta writer
            let data_writer_instance = data_writer.build().await?;
            let pos_delete_writer_instance = pos_delete_writer.build().await?;
            let eq_delete_writer_instance = eq_delete_writer.build().await?;
            let mut delta_writer = DeltaWriter::try_new(
                data_writer_instance,
                pos_delete_writer_instance,
                eq_delete_writer_instance,
                vec![1], // unique on id column
            )?;

            // Write batch with only inserts
            let batch = create_test_batch_with_ops(
                vec![1, 2, 3],
                vec![Some("Alice"), Some("Bob"), Some("Charlie")],
                vec![1, 1, 1], // all inserts
            );

            delta_writer.write(batch).await?;
            let data_files = delta_writer.close().await?;

            // Should have 1 data file, 0 delete files
            assert_eq!(data_files.len(), 1);
            assert_eq!(data_files[0].content, crate::spec::DataContentType::Data);
            assert_eq!(data_files[0].record_count, 3);

            // Read back and verify
            let input_file = file_io.new_input(data_files[0].file_path.clone())?;
            let content = input_file.read().await?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(content)?.build()?;
            let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].num_rows(), 3);

            Ok(())
        }

        #[tokio::test]
        async fn test_delta_writer_insert_then_position_delete() -> Result<()> {
            let temp_dir = TempDir::new().unwrap();
            let file_io = FileIOBuilder::new_fs_io().build().unwrap();
            let schema = create_iceberg_schema();

            // Create writers (same setup as above)
            let data_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/data",
                temp_dir.path().to_str().unwrap()
            ));
            let data_file_name_gen =
                DefaultFileNameGenerator::new("data".to_string(), None, DataFileFormat::Parquet);
            let data_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                schema.clone(),
                None,
                file_io.clone(),
                data_location_gen,
                data_file_name_gen,
            );
            let data_writer = DataFileWriterBuilder::new(data_parquet_writer, None, 0);

            let pos_delete_schema = Arc::new(arrow_schema_to_schema(
                &PositionDeleteWriterConfig::arrow_schema(),
            )?);
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/pos_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let pos_delete_file_name_gen = DefaultFileNameGenerator::new(
                "pos_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let pos_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                pos_delete_schema,
                None,
                file_io.clone(),
                pos_delete_location_gen,
                pos_delete_file_name_gen,
            );
            let pos_delete_writer = PositionDeleteFileWriterBuilder::new(
                pos_delete_parquet_writer,
                PositionDeleteWriterConfig::new(None, 0, None),
            );

            let eq_delete_config =
                EqualityDeleteWriterConfig::new(vec![1], schema.clone(), None, 0)?;
            let eq_delete_schema = Arc::new(arrow_schema_to_schema(
                eq_delete_config.projected_arrow_schema_ref(),
            )?);
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/eq_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let eq_delete_file_name_gen = DefaultFileNameGenerator::new(
                "eq_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let eq_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                eq_delete_schema,
                None,
                file_io.clone(),
                eq_delete_location_gen,
                eq_delete_file_name_gen,
            );
            let eq_delete_writer =
                EqualityDeleteFileWriterBuilder::new(eq_delete_parquet_writer, eq_delete_config);

            let data_writer_instance = data_writer.build().await?;
            let pos_delete_writer_instance = pos_delete_writer.build().await?;
            let eq_delete_writer_instance = eq_delete_writer.build().await?;
            let mut delta_writer = DeltaWriter::try_new(
                data_writer_instance,
                pos_delete_writer_instance,
                eq_delete_writer_instance,
                vec![1],
            )?;

            // First, insert some rows
            let insert_batch = create_test_batch_with_ops(
                vec![1, 2, 3],
                vec![Some("Alice"), Some("Bob"), Some("Charlie")],
                vec![1, 1, 1],
            );
            delta_writer.write(insert_batch).await?;

            // Now delete rows that were just inserted (should create position deletes)
            let delete_batch =
                create_test_batch_with_ops(vec![1, 2], vec![Some("Alice"), Some("Bob")], vec![
                    -1, -1,
                ]);
            delta_writer.write(delete_batch).await?;

            let data_files = delta_writer.close().await?;

            // Should have 1 data file + 1 position delete file
            assert_eq!(data_files.len(), 2);

            let data_file = data_files
                .iter()
                .find(|f| f.content == crate::spec::DataContentType::Data)
                .unwrap();
            let pos_delete_file = data_files
                .iter()
                .find(|f| f.content == crate::spec::DataContentType::PositionDeletes)
                .unwrap();

            assert_eq!(data_file.record_count, 3);
            assert_eq!(pos_delete_file.record_count, 2);

            // Verify position delete file content
            let input_file = file_io.new_input(pos_delete_file.file_path.clone())?;
            let content = input_file.read().await?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(content)?.build()?;
            let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
            assert_eq!(batches[0].num_rows(), 2);

            Ok(())
        }

        #[tokio::test]
        async fn test_delta_writer_equality_delete() -> Result<()> {
            let temp_dir = TempDir::new().unwrap();
            let file_io = FileIOBuilder::new_fs_io().build().unwrap();
            let schema = create_iceberg_schema();

            // Create writers
            let data_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/data",
                temp_dir.path().to_str().unwrap()
            ));
            let data_file_name_gen =
                DefaultFileNameGenerator::new("data".to_string(), None, DataFileFormat::Parquet);
            let data_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                schema.clone(),
                None,
                file_io.clone(),
                data_location_gen,
                data_file_name_gen,
            );
            let data_writer = DataFileWriterBuilder::new(data_parquet_writer, None, 0);

            let pos_delete_schema = Arc::new(arrow_schema_to_schema(
                &PositionDeleteWriterConfig::arrow_schema(),
            )?);
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/pos_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let pos_delete_file_name_gen = DefaultFileNameGenerator::new(
                "pos_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let pos_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                pos_delete_schema,
                None,
                file_io.clone(),
                pos_delete_location_gen,
                pos_delete_file_name_gen,
            );
            let pos_delete_writer = PositionDeleteFileWriterBuilder::new(
                pos_delete_parquet_writer,
                PositionDeleteWriterConfig::new(None, 0, None),
            );

            let eq_delete_config =
                EqualityDeleteWriterConfig::new(vec![1], schema.clone(), None, 0)?;
            let eq_delete_schema = Arc::new(arrow_schema_to_schema(
                eq_delete_config.projected_arrow_schema_ref(),
            )?);
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/eq_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let eq_delete_file_name_gen = DefaultFileNameGenerator::new(
                "eq_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let eq_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                eq_delete_schema,
                None,
                file_io.clone(),
                eq_delete_location_gen,
                eq_delete_file_name_gen,
            );
            let eq_delete_writer =
                EqualityDeleteFileWriterBuilder::new(eq_delete_parquet_writer, eq_delete_config);

            let data_writer_instance = data_writer.build().await?;
            let pos_delete_writer_instance = pos_delete_writer.build().await?;
            let eq_delete_writer_instance = eq_delete_writer.build().await?;
            let mut delta_writer = DeltaWriter::try_new(
                data_writer_instance,
                pos_delete_writer_instance,
                eq_delete_writer_instance,
                vec![1],
            )?;

            // Delete rows that were never inserted (should create equality deletes)
            let delete_batch =
                create_test_batch_with_ops(vec![99, 100], vec![Some("X"), Some("Y")], vec![-1, -1]);
            delta_writer.write(delete_batch).await?;

            let data_files = delta_writer.close().await?;

            // Should have only 1 equality delete file
            assert_eq!(data_files.len(), 1);
            assert_eq!(
                data_files[0].content,
                crate::spec::DataContentType::EqualityDeletes
            );
            assert_eq!(data_files[0].record_count, 2);

            Ok(())
        }

        #[tokio::test]
        async fn test_delta_writer_invalid_op() -> Result<()> {
            let temp_dir = TempDir::new().unwrap();
            let file_io = FileIOBuilder::new_fs_io().build().unwrap();
            let schema = create_iceberg_schema();

            // Create writers
            let data_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/data",
                temp_dir.path().to_str().unwrap()
            ));
            let data_file_name_gen =
                DefaultFileNameGenerator::new("data".to_string(), None, DataFileFormat::Parquet);
            let data_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                schema.clone(),
                None,
                file_io.clone(),
                data_location_gen,
                data_file_name_gen,
            );
            let data_writer = DataFileWriterBuilder::new(data_parquet_writer, None, 0);

            let pos_delete_schema = Arc::new(arrow_schema_to_schema(
                &PositionDeleteWriterConfig::arrow_schema(),
            )?);
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/pos_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let pos_delete_file_name_gen = DefaultFileNameGenerator::new(
                "pos_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let pos_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                pos_delete_schema,
                None,
                file_io.clone(),
                pos_delete_location_gen,
                pos_delete_file_name_gen,
            );
            let pos_delete_writer = PositionDeleteFileWriterBuilder::new(
                pos_delete_parquet_writer,
                PositionDeleteWriterConfig::new(None, 0, None),
            );

            let eq_delete_config =
                EqualityDeleteWriterConfig::new(vec![1], schema.clone(), None, 0)?;
            let eq_delete_schema = Arc::new(arrow_schema_to_schema(
                eq_delete_config.projected_arrow_schema_ref(),
            )?);
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(format!(
                "{}/eq_delete",
                temp_dir.path().to_str().unwrap()
            ));
            let eq_delete_file_name_gen = DefaultFileNameGenerator::new(
                "eq_delete".to_string(),
                None,
                DataFileFormat::Parquet,
            );
            let eq_delete_parquet_writer = ParquetWriterBuilder::new(
                WriterProperties::builder().build(),
                eq_delete_schema,
                None,
                file_io.clone(),
                eq_delete_location_gen,
                eq_delete_file_name_gen,
            );
            let eq_delete_writer =
                EqualityDeleteFileWriterBuilder::new(eq_delete_parquet_writer, eq_delete_config);

            let data_writer_instance = data_writer.build().await?;
            let pos_delete_writer_instance = pos_delete_writer.build().await?;
            let eq_delete_writer_instance = eq_delete_writer.build().await?;
            let mut delta_writer = DeltaWriter::try_new(
                data_writer_instance,
                pos_delete_writer_instance,
                eq_delete_writer_instance,
                vec![1],
            )?;

            // Invalid operation code
            let batch = create_test_batch_with_ops(vec![1], vec![Some("Alice")], vec![99]);

            let result = delta_writer.write(batch).await;
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Ops column must be 1 (insert) or -1 (delete)")
            );

            Ok(())
        }
    }
}
