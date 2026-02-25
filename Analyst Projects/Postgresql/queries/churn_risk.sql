WITH transaction_gaps AS (
  SELECT
    cc_num,
    first_name || ' ' || last_name          AS customer,
    trans_datetime,
    LEAD(trans_datetime) OVER (
      PARTITION BY cc_num
      ORDER BY trans_datetime
    )                                        AS next_transaction
  FROM transactions
  WHERE is_fraud = 0
)
SELECT
  customer,
  cc_num,
  trans_datetime                             AS last_activity,
  next_transaction,
  EXTRACT(DAY FROM (next_transaction - trans_datetime)) AS gap_days
FROM transaction_gaps
WHERE next_transaction IS NOT NULL
    -- normally a 30-90 day gap
  AND (next_transaction - trans_datetime) > INTERVAL '7 days'
ORDER BY gap_days DESC;