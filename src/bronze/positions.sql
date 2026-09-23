-- Bronze | positions — daily custodian position snapshots
-- Incremental streaming ingest (Auto Loader): one file lands per business day.
CREATE OR REFRESH STREAMING TABLE positions
  COMMENT 'Raw daily position snapshots landed from custodian feeds'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS SELECT
  *,
  _metadata.file_path              AS _source_file,
  _metadata.file_modification_time AS _source_file_modified,
  current_timestamp()              AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/positions',
  format => 'parquet'
);
