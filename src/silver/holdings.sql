-- Silver | holdings — latest enriched holdings, one row per account & security.
-- Cleans bronze (drops ingest metadata, avg_cost, acquired_date), joins the
-- reference dimensions, and adds unrealized gain, a long-term flag, and the
-- security's 5-trading-day return.
CREATE OR REFRESH MATERIALIZED VIEW `fe-bar-josed`.silver.holdings
  COMMENT 'Latest enriched holdings (account x security) with gain/loss and 5-day price move'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS
WITH latest_positions AS (
  SELECT *
  FROM `fe-bar-josed`.bronze.positions
  WHERE as_of_date = (SELECT max(as_of_date) FROM `fe-bar-josed`.bronze.positions)
),
price_ranked AS (
  SELECT symbol, close_price,
         row_number() OVER (PARTITION BY symbol ORDER BY as_of_date DESC) AS rn
  FROM `fe-bar-josed`.bronze.prices
),
price_move AS (
  SELECT l.symbol,
         round(l.close_price / p.close_price - 1, 4) AS return_5d
  FROM price_ranked l
  JOIN price_ranked p ON l.symbol = p.symbol AND p.rn = 6
  WHERE l.rn = 1
)
SELECT
  p.as_of_date,
  a.client_id,
  cl.advisor_id,
  p.account_id,
  a.account_type,
  a.model_portfolio,
  p.symbol,
  s.name                                          AS security_name,
  s.asset_class,
  s.sector,
  p.quantity,
  p.close_price,
  p.market_value,
  p.cost_basis_total,
  round(p.market_value - p.cost_basis_total, 2)   AS unrealized_gain,
  datediff(p.as_of_date, p.acquired_date) >= 365  AS is_long_term,
  coalesce(pm.return_5d, 0)                        AS return_5d
FROM latest_positions p
JOIN `fe-bar-josed`.bronze.accounts   a  ON p.account_id = a.account_id
JOIN `fe-bar-josed`.bronze.clients    cl ON a.client_id  = cl.client_id
JOIN `fe-bar-josed`.bronze.securities s  ON p.symbol     = s.symbol
LEFT JOIN price_move pm ON p.symbol = pm.symbol;
