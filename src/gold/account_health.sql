-- Gold | account_health — per-account portfolio metrics and signal inputs.
-- One row per account: total value, cash weight, worst asset-class drift vs the
-- model target, taxable unrealized losses, and a concentrated-position-drop flag.
CREATE OR REFRESH MATERIALIZED VIEW `fe-bar-josed`.gold.account_health
  COMMENT 'Per-account value, asset-class weights, model drift, taxable losses, concentration'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS
WITH acct AS (
  SELECT
    account_id, client_id, advisor_id, account_type, model_portfolio,
    sum(market_value) AS account_value,
    sum(CASE WHEN asset_class = 'Equity'       THEN market_value ELSE 0 END) AS equity_val,
    sum(CASE WHEN asset_class = 'Fixed Income' THEN market_value ELSE 0 END) AS fixed_income_val,
    sum(CASE WHEN asset_class = 'Alternatives' THEN market_value ELSE 0 END) AS alternatives_val,
    sum(CASE WHEN asset_class = 'Cash'         THEN market_value ELSE 0 END) AS cash_val,
    sum(CASE WHEN account_type IN ('Taxable', 'Joint Taxable') AND unrealized_gain < 0
             THEN unrealized_gain ELSE 0 END) AS taxable_unrealized_loss
  FROM `fe-bar-josed`.silver.holdings
  GROUP BY account_id, client_id, advisor_id, account_type, model_portfolio
),
model_wide AS (
  SELECT model_portfolio,
    max(CASE WHEN asset_class = 'Equity'       THEN target_weight END) AS equity_t,
    max(CASE WHEN asset_class = 'Fixed Income' THEN target_weight END) AS fixed_income_t,
    max(CASE WHEN asset_class = 'Alternatives' THEN target_weight END) AS alternatives_t,
    max(CASE WHEN asset_class = 'Cash'         THEN target_weight END) AS cash_t
  FROM `fe-bar-josed`.bronze.model_allocations
  GROUP BY model_portfolio
),
concentration AS (
  SELECT h.account_id,
         max(CASE WHEN h.market_value / ac.account_value > 0.20 AND h.return_5d < -0.12
                  THEN 1 ELSE 0 END) = 1 AS has_concentrated_drop
  FROM `fe-bar-josed`.silver.holdings h
  JOIN acct ac ON h.account_id = ac.account_id
  GROUP BY h.account_id
)
SELECT
  a.account_id, a.client_id, a.advisor_id, a.account_type, a.model_portfolio,
  round(a.account_value, 2)                 AS account_value,
  round(a.cash_val / a.account_value, 4)    AS cash_weight,
  round(a.taxable_unrealized_loss, 2)       AS taxable_unrealized_loss,
  round(greatest(
    abs(a.equity_val       / a.account_value - coalesce(m.equity_t, 0)),
    abs(a.fixed_income_val / a.account_value - coalesce(m.fixed_income_t, 0)),
    abs(a.alternatives_val / a.account_value - coalesce(m.alternatives_t, 0)),
    abs(a.cash_val         / a.account_value - coalesce(m.cash_t, 0))
  ), 4)                                     AS max_drift,
  coalesce(c.has_concentrated_drop, false)  AS has_concentrated_drop
FROM acct a
LEFT JOIN model_wide m    ON a.model_portfolio = m.model_portfolio
LEFT JOIN concentration c ON a.account_id       = c.account_id;
