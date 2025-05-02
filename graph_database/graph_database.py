import logging
from typing import List, Dict, Any

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphDatabaseManager:
    def __init__(
            self,
            uri: str = "bolt://localhost:7687",
            user: str = "neo4j",
            password: str = "your_password"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._initialize_constraints()

    def _initialize_constraints(self):
        with self.driver.session() as session:
            # Create constraints for unique IDs
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:NewsChunk) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")

    def add_chunk(self, chunk_id: str, content: str, metadata: Dict[str, Any]):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:NewsChunk {id: $chunk_id})
                SET c.content = $content,
                    c.published_date = $published_date,
                    c.source = $source,
                    c.embedding_id = $embedding_id
                """,
                chunk_id=chunk_id,
                content=content,
                published_date=metadata.get("published_date"),
                source=metadata.get("source"),
                embedding_id=metadata.get("id")
            )

    def add_semantic_edge(self, chunk1_id: str, chunk2_id: str, similarity: float):
        if similarity < 0.7:  # Configurable threshold
            return

        with self.driver.session() as session:
            session.run(
                """
                MATCH (c1:NewsChunk {id: $chunk1_id})
                MATCH (c2:NewsChunk {id: $chunk2_id})
                MERGE (c1)-[r:SIMILAR_TO]-(c2)
                SET r.score = $similarity,
                    r.created_at = datetime()
                """,
                chunk1_id=chunk1_id,
                chunk2_id=chunk2_id,
                similarity=similarity
            )

    def add_temporal_edge(self, chunk1_id: str, chunk2_id: str, time_diff_hours: float):
        if time_diff_hours > 72:  # Only connect if within 3 days
            return

        with self.driver.session() as session:
            session.run(
                """
                MATCH (c1:NewsChunk {id: $chunk1_id})
                MATCH (c2:NewsChunk {id: $chunk2_id})
                MERGE (c1)-[r:TEMPORAL_PROXIMITY]-(c2)
                SET r.hours_diff = $time_diff,
                    r.created_at = datetime()
                """,
                chunk1_id=chunk1_id,
                chunk2_id=chunk2_id,
                time_diff=time_diff_hours
            )

    def add_entity_edges(self, chunk_id: str, entities: List[str]):
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    """
                    MATCH (c:NewsChunk {id: $chunk_id})
                    MERGE (e:Entity {name: $entity})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.created_at = datetime()
                    """,
                    chunk_id=chunk_id,
                    entity=entity
                )

    def get_expanded_context(
            self,
            seed_chunks: List[str],
            max_hops: int = 2,
            max_results: int = 10,
            min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (seed:NewsChunk)
                WHERE seed.id IN $seed_ids
                CALL apoc.path.expandConfig(seed, {
                    relationshipFilter: "SIMILAR_TO|TEMPORAL_PROXIMITY|MENTIONS",
                    minLevel: 1,
                    maxLevel: $max_hops
                })
                YIELD path
                WITH DISTINCT last(nodes(path)) as node, 
                     reduce(score = 0, r IN relationships(path) | 
                         score + CASE type(r)
                             WHEN 'SIMILAR_TO' THEN r.score
                             WHEN 'TEMPORAL_PROXIMITY' THEN 1 - (r.hours_diff / 72)
                             ELSE 0.5
                         END
                     ) / length(path) as relevance_score
                WHERE relevance_score >= $min_similarity
                RETURN node.id as chunk_id,
                       node.content as content,
                       node.embedding_id as embedding_id,
                       node.published_date as published_date,
                       node.source as source,
                       relevance_score
                ORDER BY relevance_score DESC
                LIMIT $max_results
                """,
                seed_ids=seed_chunks,
                max_hops=max_hops,
                min_similarity=min_similarity,
                max_results=max_results
            )

            return [dict(record) for record in result]

    def close(self):
        self.driver.close()
