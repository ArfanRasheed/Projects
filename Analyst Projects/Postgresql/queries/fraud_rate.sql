SELECT
	category,
	COUNT(*)                                            AS total_transactions,
	SUM(is_fraud)                                      AS fraud_count,
	ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)         AS fraud_rate_pct,
	ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_dollar_exposure
FROM transactions
GROUP BY category
HAVING COUNT(*) > 100
ORDER BY fraud_rate_pct DESC;

