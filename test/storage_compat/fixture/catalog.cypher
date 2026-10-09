CREATE NODE TABLE Acct(id INT64 PRIMARY KEY, email STRING, bal DECIMAL(18, 4), note STRING);
UNWIND range(1, 50) AS i CREATE (:Acct {id: i, email: 'u' + CAST(i AS STRING) + '@x', bal: CAST(i AS DECIMAL(18, 4)) / 3, note: CASE WHEN i % 10 = 0 THEN repeat('long', 4000) ELSE 's' END});
COMMENT ON TABLE Acct IS 'accounts';
CREATE SEQUENCE ticket START 100 INCREMENT 7;
RETURN nextval('ticket');
RETURN nextval('ticket');
CREATE SEQUENCE unused;
CREATE MACRO double_it(x) AS x * 2;
