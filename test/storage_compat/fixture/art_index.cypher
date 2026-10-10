CALL enable_default_hash_index=false;
CREATE NODE TABLE Tag(name STRING PRIMARY KEY, n INT64);
CREATE ART INDEX tag_pk FOR (t:Tag) ON (t.name);
UNWIND range(0, 2999) AS i CREATE (:Tag {name: 'tag' + CAST(i AS STRING), n: i});
CREATE (:Tag {name: 'ó', n: -1});
CALL enable_default_hash_index=true;
