WITH daily_spend AS (
	SELECT
		DATE_TRUNC('day', trans_datetime) AS day,
		ROUND(SUM(amount), 2) AS daily_total
	FROM transactions
	WHERE is_fraud = 0
	GROUP BY 1
)
SELECT
	day,
	daily_total,
	ROUND(AVG(daily_total) OVER (
		ORDER BY day
		ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
	), 2) AS rolling_30d_avg
FROM daily_spend
ORDER BY day;

