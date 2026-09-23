-- Bronze | model_allocations — reference dimension (static full snapshot, batch read)
CREATE OR REFRESH MATERIALIZED VIEW model_allocations
  COMMENT 'Target asset-class weights per model portfolio'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS SELECT
  *,
  _metadata.file_path AS _source_file,
  current_timestamp() AS _ingested_at
FROM read_files(
  '${landing_path}/reference/model_allocations.parquet',
  format => 'parquet'
);
