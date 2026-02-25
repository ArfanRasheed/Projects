WITH running AS (
  SELECT
    cc_num,
    first_name || ' ' || last_name       AS customer,
    trans_datetime,
    category,
    amount,
    ROUND(SUM(amount) OVER (
      PARTITION BY cc_num
      ORDER BY trans_datetime
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ), 2)                                AS running_total_spend
  FROM transactions
  WHERE is_fraud = 0
)
SELECT DISTINCT ON (cc_num)
  cc_num,
  customer,
  running_total_spend                    AS lifetime_total_spend
FROM running
ORDER BY cc_num, trans_datetime DESC;