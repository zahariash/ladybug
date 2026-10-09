CREATE NODE TABLE Ranked(id INT64 PRIMARY KEY, rank INT64);
ALTER TABLE Ranked SET SORTED BY (id ASC) CSR;
CREATE REL TABLE Follows(FROM Ranked TO Ranked, w INT64);
ALTER TABLE Follows SET SORTED BY (FROM ASC, TO ASC) CSR;
UNWIND range(0, 999) AS i CREATE (:Ranked {id: i, rank: i % 17});
UNWIND range(0, 998) AS i MATCH (a:Ranked {id: i}) WITH a, i MATCH (b:Ranked {id: i + 1}) CREATE (a)-[:Follows {w: i}]->(b);
