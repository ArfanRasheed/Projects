WITH monthly_spend AS (
	SELECT
		DATE_TRUNC('month', trans_datetime) AS month,
		ROUND(SUM(amount), 2) AS total_spend
	FROM transactions
	WHERE is_fraud = 0
	GROUP BY 1
)
SELECT
	month,
	total_spend,
	LAG(total_spend) OVER (ORDER BY month) AS prior_month_spend,
	ROUND(
		(total_spend - LAG(total_spend) OVER (ORDER BY month))
		/ NULLIF(LAG(total_spend) OVER (ORDER BY month), 0) * 100
	, 2) AS mom_growth_pct
FROM monthly_spend
ORDER BY month;

