SELECT
		t.trans_num,
		t.first_name || ' ' || t.last_name AS customer,
		t.category,
		t.amount                               AS transaction_amount,
		ROUND(avg_spend.avg_amt, 2)            AS customer_avg_spend,
		ROUND(t.amount / avg_spend.avg_amt, 1) AS spend_multiplier
FROM transactions t
JOIN (
		SELECT cc_num, AVG(amount) AS avg_amt
		FROM transactions
		WHERE is_fraud = 0
		GROUP BY cc_num
) avg_spend ON t.cc_num = avg_spend.cc_num
WHERE t.amount > (avg_spend.avg_amt * 3)
	AND t.is_fraud = 0
ORDER BY spend_multiplier DESC
LIMIT 100;

