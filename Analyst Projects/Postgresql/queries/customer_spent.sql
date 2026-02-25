WITH customer_totals AS (
	SELECT
		cc_num,
		first_name || ' ' || last_name AS customer_name,
		COUNT(*) AS total_transactions,
		ROUND(SUM(amount), 2) AS lifetime_spend
	FROM transactions
	WHERE is_fraud = 0
	GROUP BY cc_num, first_name, last_name
)
SELECT
	customer_name,
	lifetime_spend,
	total_transactions,
	CASE
		WHEN lifetime_spend >= 50000 THEN 'Platinum'
		WHEN lifetime_spend >= 25000 THEN 'Gold'
		WHEN lifetime_spend >= 10000 THEN 'Silver'
		ELSE 'Standard'
	END AS customer_tier,
	RANK() OVER (ORDER BY lifetime_spend DESC) AS spend_rank
FROM customer_totals
ORDER BY lifetime_spend DESC
LIMIT 50;

