SELECT
	COALESCE(state, 'ALL STATES')    AS state,
	COALESCE(category, 'ALL CATEGORIES') AS category,
	COUNT(*)                         AS total_transactions,
	ROUND(SUM(amount), 2)           AS total_spend,
	ROUND(AVG(amount), 2)           AS avg_spend
FROM transactions
WHERE is_fraud = 0
GROUP BY ROLLUP(state, category)
ORDER BY
	CASE WHEN state IS NULL THEN 2 ELSE 0 END,
	CASE WHEN category IS NULL THEN 2 ELSE 0 END,
	total_spend DESC;

