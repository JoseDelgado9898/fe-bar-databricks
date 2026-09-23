-- Bronze | accounts — reference dimension (static full snapshot, batch read)
CREATE OR REFRESH MATERIALIZED VIEW accounts
  COMMENT 'Account reference data (type, model portfolio, custodian)'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS SELECT
  *,
  _metadata.file_path AS _source_file,
  current_timestamp() AS _ingested_at
FROM read_files(
  '${landing_path}/reference/accounts.parquet',
  format => 'parquet'
);
