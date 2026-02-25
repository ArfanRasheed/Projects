WITH merchant_ranks AS (
	SELECT
		category,
		merchant,
		ROUND(SUM(amount), 2) AS total_revenue,
		COUNT(*) AS transaction_count,
		ROW_NUMBER() OVER (
			PARTITION BY category
			ORDER BY SUM(amount) DESC
		) AS rank_in_category
	FROM transactions
	WHERE is_fraud = 0
	GROUP BY category, merchant
)
SELECT
	category,
	merchant,
	total_revenue,
	transaction_count
FROM merchant_ranks
WHERE rank_in_category = 1
ORDER BY total_revenue DESC;

