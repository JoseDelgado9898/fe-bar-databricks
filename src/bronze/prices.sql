-- Bronze | prices — daily end-of-day closing prices per security
-- Incremental streaming ingest (Auto Loader): one file lands per business day.
CREATE OR REFRESH STREAMING TABLE prices
  COMMENT 'Raw daily EOD prices landed from the market-data feed'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS SELECT
  *,
  _metadata.file_path              AS _source_file,
  _metadata.file_modification_time AS _source_file_modified,
  current_timestamp()              AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/prices',
  format => 'parquet'
);
