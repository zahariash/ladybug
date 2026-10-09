CREATE NODE TABLE Orders(id INT64 PRIMARY KEY, region STRING, amount INT64) PARTITION BY HASH (region) PARTITIONS 3;
CREATE NODE TABLE Tagged(id INT64 PRIMARY KEY, kind STRING) PARTITION BY LIST (kind);
UNWIND range(0, 999) AS i CREATE (:Orders {id: i, region: 'r' + CAST(i % 5 AS STRING), amount: i * 3});
UNWIND range(0, 999) AS i CREATE (:Tagged {id: i, kind: 'k' + CAST(i % 4 AS STRING)});
