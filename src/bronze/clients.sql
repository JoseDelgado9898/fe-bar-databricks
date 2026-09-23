-- Bronze | clients — reference dimension (static full snapshot, batch read)
CREATE OR REFRESH MATERIALIZED VIEW clients
  COMMENT 'Client / household reference data'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS SELECT
  *,
  _metadata.file_path AS _source_file,
  current_timestamp() AS _ingested_at
FROM read_files(
  '${landing_path}/reference/clients.parquet',
  format => 'parquet'
);
