# wealth_advisor_worklist

A Databricks Asset Bundle (DABs) for a wealth-advisor client worklist: a medallion
pipeline, an ML model, a Genie space, and a Dash app — all deployed as code.

## Resources

| Resource | File | Description |
|----------|------|-------------|
| Bronze pipeline | `resources/bronze_ingest.pipeline.yml` | Ingests the raw landing zone; transforms in `src/` (bronze → silver → gold). |
| Training job | `resources/client_needs_attention.job.yml` | Serverless job running `ml/client_needs_attention.py` (logistic classifier → MLflow → UC model). |
| Genie space | `resources/client_portfolio_health_and_alerts.genie_space.yml` | NL Q&A over the gold tables. |
| App | `resources/fe_bar_portfolio_manager.app.yml` | Dash portfolio-manager app (`app/`) reading Lakebase synced tables. |

## Layout

- `src/` — pipeline transforms (SQL). **Note:** the bronze pipeline globs `../src/**`, so only pipeline source belongs here.
- `ml/` — ML notebook(s) run by jobs (kept out of `src/` on purpose).
- `app/` — Dash app source (`app.py`, `app.yaml`, `requirements.txt`).
- `data-gen/` — local synthetic-data generator; excluded from bundle deploys and git.

## Usage

```bash
databricks bundle validate -t dev --profile aws_workspace
databricks bundle deploy   -t dev --profile aws_workspace
databricks bundle run <resource_key> -t dev --profile aws_workspace
```

Key variables (see `databricks.yml`): `catalog`, `bronze_schema`, `landing_path`.
