MATCH (p:Person {id: 4242}) SET p.name = 'upgraded', p.score = 1.5;
MATCH (p:Person {id: 4243}) DETACH DELETE p;
CREATE (:Person {id: 300000, name: 'new', age: 5});
MATCH (a:Person {id: 300000}), (b:Person {id: 1}) CREATE (a)-[:Knows {weight: 2.0}]->(b);
