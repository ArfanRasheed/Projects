CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    trans_datetime TIMESTAMP,
    cc_num BIGINT,
    merchant VARCHAR(255),
    category VARCHAR(100),
    amount DECIMAL(10, 2),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    gender CHAR(1),
    street VARCHAR(255),
    city VARCHAR(100),
    state CHAR(2),
    zip VARCHAR(10),
    lat DECIMAL(9, 6),
    long DECIMAL(9, 6),
    city_pop INTEGER,
    job VARCHAR(255),
    dob DATE,
    trans_num VARCHAR(50),
    unix_time BIGINT,
    merch_lat DECIMAL(9, 6),
    merch_long DECIMAL(9, 6),
    is_fraud SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_cc_num ON transactions(cc_num);
CREATE INDEX IF NOT EXISTS idx_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_datetime ON transactions(trans_datetime);
CREATE INDEX IF NOT EXISTS idx_is_fraud ON transactions(is_fraud);

ALTER TABLE transactions ADD COLUMN row_index INTEGER;