SELECT
    category,
    COUNT(*) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_spend,
    ROUND(AVG(amount), 2) AS avg_transaction,
    ROUND(MAX(amount), 2) AS max_transaction
FROM transactions
GROUP BY category
ORDER BY total_spend DESC;
