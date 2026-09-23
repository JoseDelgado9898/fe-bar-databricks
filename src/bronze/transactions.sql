-- Bronze | transactions — daily trade / cash activity
-- Incremental streaming ingest (Auto Loader): one file lands per business day.
CREATE OR REFRESH STREAMING TABLE transactions
  COMMENT 'Raw daily transactions (buys, sells, contributions, withdrawals, dividends)'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS SELECT
  *,
  _metadata.file_path              AS _source_file,
  _metadata.file_modification_time AS _source_file_modified,
  current_timestamp()              AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/transactions',
  format => 'parquet'
);
