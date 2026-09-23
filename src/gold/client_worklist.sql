-- Gold | client_worklist — one row per client: the advisor's ranked call list.
-- Rolls account_health up to the client, joins client + advisor detail, and
-- derives six signal flags, a weighted priority score, and a plain-English reason.
CREATE OR REFRESH MATERIALIZED VIEW `fe-bar-josed`.gold.client_worklist
  COMMENT 'Per-client worklist: signal flags, priority score, and the reason to call'
  TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')
AS
WITH client_roll AS (
  SELECT
    client_id,
    advisor_id,
    round(sum(account_value), 2)                                       AS portfolio_value,
    count(*)                                                           AS num_accounts,
    max(max_drift)                                                     AS max_drift,
    round(sum(cash_weight * account_value) / sum(account_value), 4)    AS cash_weight,
    round(sum(taxable_unrealized_loss), 2)                             AS taxable_unrealized_loss,
    max(CASE WHEN has_concentrated_drop THEN 1 ELSE 0 END) = 1         AS has_concentrated_drop,
    max(CASE WHEN account_type = 'Traditional IRA' THEN 1 ELSE 0 END) = 1 AS has_traditional_ira
  FROM `fe-bar-josed`.gold.account_health
  GROUP BY client_id, advisor_id
),
flags AS (
  SELECT
    r.client_id, r.advisor_id, r.portfolio_value, r.num_accounts,
    r.max_drift, r.cash_weight, r.taxable_unrealized_loss,
    cl.first_name, cl.last_name, cl.email, cl.state, cl.age,
    cl.risk_profile, cl.last_review_date,
    r.has_concentrated_drop                                    AS flag_concentrated_drop,
    (r.max_drift > 0.10)                                       AS flag_drift,
    (r.taxable_unrealized_loss < -2000)                        AS flag_tax_loss_harvest,
    (cl.age >= 73 AND r.has_traditional_ira)                   AS flag_rmd,
    (r.cash_weight > 0.12)                                     AS flag_cash_drag,
    (cl.last_review_date < add_months(current_date(), -12))    AS flag_review_overdue
  FROM client_roll r
  JOIN `fe-bar-josed`.bronze.clients cl ON r.client_id = cl.client_id
)
SELECT
  f.client_id, f.first_name, f.last_name, f.email, f.state, f.age, f.risk_profile,
  adv.advisor_name, adv.advisor_email,
  f.portfolio_value, f.num_accounts, f.max_drift, f.cash_weight,
  f.taxable_unrealized_loss, f.last_review_date,
  f.flag_concentrated_drop, f.flag_drift, f.flag_tax_loss_harvest,
  f.flag_rmd, f.flag_cash_drag, f.flag_review_overdue,
  (  cast(f.flag_concentrated_drop AS INT) * 40
   + cast(f.flag_drift             AS INT) * 25
   + cast(f.flag_tax_loss_harvest  AS INT) * 20
   + cast(f.flag_rmd               AS INT) * 15
   + cast(f.flag_cash_drag         AS INT) * 10
   + cast(f.flag_review_overdue    AS INT) * 5 )                        AS priority_score,
  concat_ws('; ',
    CASE WHEN f.flag_concentrated_drop THEN 'Concentrated position fell >12% in 5 days' END,
    CASE WHEN f.flag_drift             THEN concat('Allocation ', cast(round(f.max_drift * 100, 0) AS INT), '% off model target') END,
    CASE WHEN f.flag_tax_loss_harvest  THEN concat('Tax-loss harvesting: $', cast(round(-f.taxable_unrealized_loss, 0) AS INT), ' of taxable losses') END,
    CASE WHEN f.flag_rmd               THEN 'RMD due (age 73+ with Traditional IRA)' END,
    CASE WHEN f.flag_cash_drag         THEN concat('Cash drag: ', cast(round(f.cash_weight * 100, 0) AS INT), '% held in cash') END,
    CASE WHEN f.flag_review_overdue    THEN 'Annual review overdue' END
  )                                                                     AS reason,
  (  cast(f.flag_concentrated_drop AS INT) + cast(f.flag_drift AS INT)
   + cast(f.flag_tax_loss_harvest AS INT)  + cast(f.flag_rmd AS INT)
   + cast(f.flag_cash_drag AS INT)         + cast(f.flag_review_overdue AS INT) ) > 0 AS needs_attention
FROM flags f
JOIN `fe-bar-josed`.bronze.advisors adv ON f.advisor_id = adv.advisor_id;
