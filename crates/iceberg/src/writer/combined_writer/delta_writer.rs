//! Delta writers handle row-level changes by combining data file and delete file writers.
//! The delta writer has three sub-writers:
//! - A data file writer for new and updated rows.
//! - A position delete file writer for deletions of existing rows (that have been written within this writer)
//! - An equality delete file writer for deletions of rows based on equality conditions (for rows that may exist in other data files).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::BooleanBuilder;
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray, make_array};
use arrow_buffer::NullBuffer;
use arrow_ord::partition::partition;
use arrow_row::{OwnedRow, RowConverter, Rows, SortField};
use arrow_schema::{DataType, Field, FieldRef, Fields};
use arrow_select::filter::filter_record_batch;
use itertools::Itertools;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::schema_to_arrow_schema;
use crate::spec::DataFile;
use crate::writer::base_writer::position_delete_writer::PositionDeleteWriterConfig;
use crate::writer::{CurrentFileStatus, IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// A projector that projects an Arrow RecordBatch to a subset of its columns based on field IDs.
#[derive(Debug)]
pub(crate) struct BatchProjector {
    // Arrow arrays can be nested, so we need a Vec<Vec<usize>> to represent the indices of the columns to project.
    field_indices: Vec<Vec<usize>>,
    projected_schema: arrow_schema::SchemaRef,
}

//The batchprojector is extremely inspired by rinsingwaves impl. thanks to them!
impl BatchProjector {
    pub fn new<F1, F2>(
        original_schema: &arrow_schema::Schema,
        field_ids: &[i32],
        field_id_fetch_fn: F1,
        nested_field_fetch: F2,
    ) -> Result<Self>
    where
        F1: Fn(&Field) -> Result<Option<i32>>,
        F2: Fn(&Field) -> bool,
    {
        let mut field_indices = Vec::with_capacity(field_ids.len());
        let mut projected_fields = Vec::with_capacity(field_ids.len());

        for &field_id in field_ids {
            if let Some((field, indices)) = Self::fetch_field_index(
                original_schema.fields(),
                field_id,
                &field_id_fetch_fn,
                &nested_field_fetch,
            )? {
                field_indices.push(indices);
                projected_fields.push(field);
            } else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Field ID {} not found in schema {:?}",
                        field_id, original_schema
                    ),
                ));
            }
        }

        if field_indices.is_empty() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "No valid fields found for the provided field IDs",
            ));
        }

        let projected_schema = arrow_schema::Schema::new(projected_fields);
        Ok(Self {
            field_indices,
            projected_schema: Arc::new(projected_schema),
        })
    }

    fn projected_schema_ref(&self) -> arrow_schema::SchemaRef {
        self.projected_schema.clone()
    }

    fn fetch_field_index<F1, F2>(
        fields: &Fields,
        target_field_id: i32,
        field_id_fetch_fn: &F1,
        nested_field_fetch: &F2,
    ) -> Result<Option<(FieldRef, Vec<usize>)>>
    where
        F1: Fn(&Field) -> Result<Option<i32>>,
        F2: Fn(&Field) -> bool,
    {
        for (pos, field) in fields.iter().enumerate() {
            let id = field_id_fetch_fn(field)?;
            if let Some(field_id) = id {
                if field_id == target_field_id {
                    return Ok(Some((field.clone(), vec![pos])));
                }
            }
            if let DataType::Struct(inner_struct) = field.data_type() {
                if nested_field_fetch(field) {
                    if let Some((field, mut sub_indices)) = Self::fetch_field_index(
                        &inner_struct,
                        target_field_id,
                        field_id_fetch_fn,
                        nested_field_fetch,
                    )? {
                        sub_indices.insert(0, pos);
                        return Ok(Some((field, sub_indices)));
                    }
                }
            }
        }
        Ok(None)
    }

    fn project_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let columns = self.project_columns(batch.columns())?;
        RecordBatch::try_new(self.projected_schema.clone(), columns)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")))
    }

    pub fn project_columns(&self, batch: &[ArrayRef]) -> Result<Vec<ArrayRef>> {
        self.field_indices
            .iter()
            .map(|indices| Self::get_col_by_id(batch, indices))
            .collect()
    }

    fn get_col_by_id(batch: &[ArrayRef], field_index: &[usize]) -> Result<ArrayRef> {
        if field_index.is_empty() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Field index cannot be empty",
            ));
        }

        let mut iter = field_index.iter();
        let first_index = *iter.next().unwrap();
        let mut array = batch[first_index].clone();
        let mut null_buffer = array.logical_nulls();

        for &i in iter {
            let struct_array = array
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Expected struct array when traversing nested fields",
                    )
                })?;

            array = struct_array.column(i).clone();
            null_buffer = NullBuffer::union(null_buffer.as_ref(), array.logical_nulls().as_ref());
        }

        Ok(make_array(
            array.to_data().into_builder().nulls(null_buffer).build()?,
        ))
    }
}

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
    pub(crate) projector: BatchProjector,
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
        let projector = BatchProjector::new(
            &schema_to_arrow_schema(&data_writer.current_schema())?,
            &unique_cols,
            |field| {
                if field.data_type().is_nested() {
                    return Ok(None);
                }
                field
                    .metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .map(|id_str| {
                        id_str.parse::<i32>().map_err(|e| {
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

        // Write first to ensure the data is persisted before updating our tracking state.
        // This prevents inconsistent state if the write fails.
        self.data_writer.write(batch.clone()).await?;

        // Only record positions after successful write
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
            .convert_columns(&self.projector.project_columns(batch.columns())?)
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
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    fn test_field_id_fetch(field: &Field) -> Result<Option<i32>> {
        // Mock field ID extraction - use the field name as ID for testing
        match field.name().as_str() {
            "id" => Ok(Some(1)),
            "name" => Ok(Some(2)),
            "address" => Ok(Some(3)),
            "street" => Ok(Some(4)),
            "city" => Ok(Some(5)),
            "age" => Ok(Some(6)),
            _ => Ok(None),
        }
    }

    fn test_nested_field_fetch(field: &Field) -> bool {
        // Allow traversing into struct fields
        matches!(field.data_type(), DataType::Struct(_))
    }

    fn create_test_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new(
                "address",
                DataType::Struct(
                    vec![
                        Field::new("street", DataType::Utf8, true),
                        Field::new("city", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ),
            Field::new("age", DataType::Int32, true),
        ])
    }

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(create_test_schema());

        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name_array = Arc::new(StringArray::from(vec![Some("John"), Some("Jane"), None]));

        let street_array = Arc::new(StringArray::from(vec![
            Some("123 Main St"),
            None,
            Some("789 Oak Ave"),
        ]));
        let city_array = Arc::new(StringArray::from(vec![Some("NYC"), Some("LA"), None]));

        let address_array = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("street", DataType::Utf8, true)),
                street_array as ArrayRef,
            ),
            (
                Arc::new(Field::new("city", DataType::Utf8, true)),
                city_array as ArrayRef,
            ),
        ]));

        let age_array = Arc::new(Int32Array::from(vec![Some(25), Some(30), None]));

        RecordBatch::try_new(schema, vec![
            id_array as ArrayRef,
            name_array as ArrayRef,
            address_array as ArrayRef,
            age_array as ArrayRef,
        ])
        .unwrap()
    }

    #[test]
    fn test_projector_simple_fields() {
        let schema = create_test_schema();
        let batch = create_test_batch();

        // Project id and name fields
        let projector = BatchProjector::new(
            &schema,
            &[1, 2], // id, name
            test_field_id_fetch,
            test_nested_field_fetch,
        )
        .unwrap();

        let projected = projector.project_columns(batch.columns()).unwrap();

        assert_eq!(projected.len(), 2);

        // Check id column
        let id_array = projected[0].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(id_array.values(), &[1, 2, 3]);

        // Check name column
        let name_array = projected[1].as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_array.value(0), "John");
        assert_eq!(name_array.value(1), "Jane");
        assert!(name_array.is_null(2));
    }

    #[test]
    fn test_projector_nested_fields() {
        let schema = create_test_schema();
        let batch = create_test_batch();

        // Project nested street field
        let projector = BatchProjector::new(
            &schema,
            &[4], // street
            test_field_id_fetch,
            test_nested_field_fetch,
        )
        .unwrap();

        let projected = projector.project_columns(batch.columns()).unwrap();

        assert_eq!(projected.len(), 1);

        let street_array = projected[0].as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(street_array.value(0), "123 Main St");
        assert!(street_array.is_null(1));
        assert_eq!(street_array.value(2), "789 Oak Ave");
    }

    #[test]
    fn test_projector_mixed_fields() {
        let schema = create_test_schema();
        let batch = create_test_batch();

        // Project id, street, and age
        let projector = BatchProjector::new(
            &schema,
            &[1, 4, 6], // id, street, age
            test_field_id_fetch,
            test_nested_field_fetch,
        )
        .unwrap();

        let projected = projector.project_columns(batch.columns()).unwrap();

        assert_eq!(projected.len(), 3);

        // Check projected schema
        assert_eq!(projector.projected_schema.fields().len(), 3);
        assert_eq!(projector.projected_schema.field(0).name(), "id");
        assert_eq!(projector.projected_schema.field(1).name(), "street");
        assert_eq!(projector.projected_schema.field(2).name(), "age");
    }

    #[test]
    fn test_projector_field_not_found() {
        let schema = create_test_schema();

        let result = BatchProjector::new(
            &schema,
            &[999], // non-existent field ID
            test_field_id_fetch,
            test_nested_field_fetch,
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Field ID 999 not found")
        );
    }

    #[test]
    fn test_projector_empty_field_ids() {
        let schema = create_test_schema();

        let result = BatchProjector::new(
            &schema,
            &[], // empty field IDs
            test_field_id_fetch,
            test_nested_field_fetch,
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No valid fields found")
        );
    }

    #[test]
    fn test_get_col_by_id_empty_index() {
        let batch = create_test_batch();
        let result = BatchProjector::get_col_by_id(batch.columns(), &[]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Field index cannot be empty")
        );
    }

    #[test]
    fn test_projector_null_propagation() {
        // Create a batch where the struct itself has nulls
        let schema = Arc::new(create_test_schema());

        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name_array = Arc::new(StringArray::from(vec![
            Some("John"),
            Some("Jane"),
            Some("Bob"),
        ]));

        let street_array = Arc::new(StringArray::from(vec![
            Some("123 Main St"),
            Some("456 Elm St"),
            Some("789 Oak Ave"),
        ]));
        let city_array = Arc::new(StringArray::from(vec![
            Some("NYC"),
            Some("LA"),
            Some("Chicago"),
        ]));

        // Create address array with one null struct
        let address_fields = vec![
            (
                Arc::new(Field::new("street", DataType::Utf8, true)),
                street_array as ArrayRef,
            ),
            (
                Arc::new(Field::new("city", DataType::Utf8, true)),
                city_array as ArrayRef,
            ),
        ];

        // Make the second address struct null
        let null_buffer = NullBuffer::from(vec![true, false, true]);
        let address_data = StructArray::from(address_fields).into_data();
        let address_array = Arc::new(StructArray::from(
            address_data
                .into_builder()
                .nulls(Some(null_buffer))
                .build()
                .unwrap(),
        ));

        let age_array = Arc::new(Int32Array::from(vec![Some(25), Some(30), Some(35)]));

        let batch = RecordBatch::try_new(schema, vec![
            id_array as ArrayRef,
            name_array as ArrayRef,
            address_array as ArrayRef,
            age_array as ArrayRef,
        ])
        .unwrap();

        let projector = BatchProjector::new(
            &batch.schema(),
            &[4], // street
            test_field_id_fetch,
            test_nested_field_fetch,
        )
        .unwrap();

        let projected = projector.project_columns(batch.columns()).unwrap();
        let street_array = projected[0].as_any().downcast_ref::<StringArray>().unwrap();

        // The street should be null when the parent address struct is null
        assert!(!street_array.is_null(0)); // address not null, street has value
        assert!(street_array.is_null(1)); // address is null, so street should be null
        assert!(!street_array.is_null(2)); // address not null, street has value
    }

    #[test]
    fn test_project_batch_method() {
        let schema = create_test_schema();
        let batch = create_test_batch();

        let projector = BatchProjector::new(
            &schema,
            &[1, 2], // id, name
            test_field_id_fetch,
            test_nested_field_fetch,
        )
        .unwrap();

        let projected_batch = projector.project_batch(&batch).unwrap();

        assert_eq!(projected_batch.num_columns(), 2);
        assert_eq!(projected_batch.num_rows(), 3);
        assert_eq!(projected_batch.schema().field(0).name(), "id");
        assert_eq!(projected_batch.schema().field(1).name(), "name");
    }

    // Tests for DeltaWriter
    mod delta_writer_tests {
        use super::*;
        use std::collections::HashMap;
        use arrow_array::{Int32Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;
        use tempfile::TempDir;

        use crate::arrow::arrow_schema_to_schema;
        use crate::io::FileIOBuilder;
        use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema as IcebergSchema, Type};
        use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
        use crate::writer::base_writer::equality_delete_writer::{
            EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
        };
        use crate::writer::base_writer::position_delete_writer::PositionDeleteFileWriterBuilder;
        use crate::writer::file_writer::location_generator::{
            DefaultFileNameGenerator, DefaultLocationGenerator,
        };
        use crate::writer::file_writer::ParquetWriterBuilder;
        use crate::writer::IcebergWriterBuilder;

        fn create_iceberg_schema() -> Arc<IcebergSchema> {
            Arc::new(
                IcebergSchema::builder()
                    .with_schema_id(0)
                    .with_fields(vec![
                        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                            .into(),
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
            let data_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/data", temp_dir.path().to_str().unwrap()),
            );
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
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/pos_delete", temp_dir.path().to_str().unwrap()),
            );
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
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/eq_delete", temp_dir.path().to_str().unwrap()),
            );
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
            assert_eq!(
                data_files[0].content,
                crate::spec::DataContentType::Data
            );
            assert_eq!(data_files[0].record_count, 3);

            // Read back and verify
            let input_file = file_io.new_input(data_files[0].file_path.clone())?;
            let content = input_file.read().await?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(content)?
                .build()?;
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
            let data_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/data", temp_dir.path().to_str().unwrap()),
            );
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
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/pos_delete", temp_dir.path().to_str().unwrap()),
            );
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
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/eq_delete", temp_dir.path().to_str().unwrap()),
            );
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
            let delete_batch = create_test_batch_with_ops(
                vec![1, 2],
                vec![Some("Alice"), Some("Bob")],
                vec![-1, -1],
            );
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
            let reader = ParquetRecordBatchReaderBuilder::try_new(content)?
                .build()?;
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
            let data_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/data", temp_dir.path().to_str().unwrap()),
            );
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
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/pos_delete", temp_dir.path().to_str().unwrap()),
            );
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
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/eq_delete", temp_dir.path().to_str().unwrap()),
            );
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
            let data_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/data", temp_dir.path().to_str().unwrap()),
            );
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
            let pos_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/pos_delete", temp_dir.path().to_str().unwrap()),
            );
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
            let eq_delete_location_gen = DefaultLocationGenerator::with_data_location(
                format!("{}/eq_delete", temp_dir.path().to_str().unwrap()),
            );
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
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Ops column must be 1 (insert) or -1 (delete)"));

            Ok(())
        }
    }
}
