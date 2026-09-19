import pyarrow.parquet as pq

# 1. Open the Parquet file pointer (does not load data into RAM)
parquet_file = pq.ParquetFile('training_2025-01-01_2025-02-01.parquet')

# 2. Get metadata dimensions
print(f"Total Rows: {parquet_file.metadata.num_rows}")
print(f"Total Columns: {parquet_file.metadata.num_columns}")

# 3. Print the schema (Column names + Data Types)
print("\n--- Schema ---")
print(parquet_file.schema)

# 4. Preview just the first 5 rows safely
print("\n--- First 5 Rows Preview ---")
first_batch = next(parquet_file.iter_batches(batch_size=5))
preview_df = first_batch.to_pandas()
print(preview_df)
