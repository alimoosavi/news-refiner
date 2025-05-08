import logging
import time
from datetime import timedelta, datetime
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)


class VectorDatabaseManager:
    # Add constant for collection name
    DEFAULT_COLLECTION = "news"

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6333,
            embedding_dim: int = 1536,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = self.DEFAULT_COLLECTION  # Always use default collection
        self.embedding_dim = embedding_dim

        # Initialize collection
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the vector database with a single collection"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                # Create collection with payload indexes
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )

                # Create payload indexes
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="keywords",
                    field_schema="keyword"
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="content",
                    field_schema="text"
                )

                logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {str(e)}")
            raise

    def hybrid_search(
            self,
            query_vector: List[float],
            keywords: List[str],
            limit: int = 5,
            score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search using both vector similarity and keyword matching"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="keywords",
                            match=models.MatchAny(any=keywords)
                        )
                    ]
                ),
                limit=limit,
                score_threshold=score_threshold
            )

            return [{
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]

        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []

    def _ensure_collection(self, collection_name: str) -> None:
        """
        Ensure that a collection exists, creating it if necessary
        
        Args:
            collection_name: Name of the collection to check/create
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                logger.info(f"Created new collection: {collection_name}")
            else:
                logger.info(f"Using existing collection: {collection_name}")

        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {str(e)}")
            raise

    def store(self, vector: List[float], payload: Dict[str, Any]) -> str:
        """Store a vector in the default collection"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                point_id = self.client.count(collection_name=self.collection_name).count

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload
                        )
                    ]
                )
                return str(point_id)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error storing vector after {max_retries} attempts: {str(e)}")
                    return None
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2

    def store_batch(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> List[str]:
        """
        Store a batch of vectors and their payloads

        Args:
            vectors: List of embedding vectors
            payloads: List of payloads corresponding to the vectors

        Returns:
            List[str]: List of stored vector IDs
        """
        try:
            start_id = self.client.count(collection_name=self.collection_name).count
            points = [
                models.PointStruct(
                    id=start_id + i,
                    vector=vector,
                    payload=payload
                )
                for i, (vector, payload) in enumerate(zip(vectors, payloads))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return [str(start_id + i) for i in range(len(vectors))]
        except Exception as e:
            logger.error(f"Error storing batch of vectors: {str(e)}")
            return []

    def search(
            self,
            query_vector: List[float],
            collection_name: Optional[str] = None,
            limit: int = 5,
            score_threshold: float = 0.7,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            collection = collection_name or self.collection_name
            query_filter = None
            
            if metadata_filters:
                conditions = []
                for key, value in metadata_filters.items():
                    if isinstance(value, dict):
                        for op, op_val in value.items():
                            if isinstance(op_val, str) and 'T' in op_val:  # Check if it's a datetime string
                                # Convert ISO datetime string to timestamp
                                timestamp = datetime.fromisoformat(op_val.replace('Z', '+00:00')).timestamp()
                                if op == "$gte":
                                    conditions.append(
                                        models.FieldCondition(
                                            key=key,
                                            match=models.Range(
                                                gte=timestamp
                                            )
                                        )
                                    )
                                elif op == "$lte" or op == "$lt":
                                    conditions.append(
                                        models.FieldCondition(
                                            key=key,
                                            match=models.Range(
                                                lte=timestamp
                                            )
                                        )
                                    )
                            else:
                                # Handle non-datetime values
                                if op == "$gte":
                                    conditions.append(
                                        models.FieldCondition(
                                            key=key,
                                            match=models.Range(
                                                gte=op_val
                                            )
                                        )
                                    )
                                elif op == "$lte" or op == "$lt":
                                    conditions.append(
                                        models.FieldCondition(
                                            key=key,
                                            match=models.Range(
                                                lte=op_val
                                            )
                                        )
                                    )
                    else:
                        # Direct equality match
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                if conditions:
                    query_filter = models.Filter(
                        must=conditions
                    )

            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
    
            return [{
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
    
        except Exception as e:
            logger.error(f"Error during search in collection {collection}: {str(e)}")
            return []

    def delete(self, idx: str) -> bool:
        """
        Delete a vector and its metadata

        Args:
            idx: ID of the vector to delete

        Returns:
            bool: True if deleted, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[int(idx)]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting vector: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database

        Returns:
            Dictionary with stats
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "total_vectors": collection_info.vectors_count,
                "collection_name": self.collection_name,
                "embedding_dim": self.embedding_dim,
                "host": self.client._client.host,
                "port": self.client._client.port
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}

    def check_connection(self) -> bool:
        """Check if the connection to Qdrant is healthy"""
        try:
            # Try to get collection info as a health check
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False

    def cleanup_old_vectors(self, older_than_days: int = 30) -> int:
        """Delete vectors older than specified days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()

            # Find points to delete
            points_to_delete = self.search(
                query_vector=None,
                metadata_filters={
                    "published_date": {
                        "$lt": cutoff_date
                    }
                },
                limit=1000  # Batch size
            )

            if not points_to_delete:
                return 0

            # Delete points
            deleted_count = 0
            for point in points_to_delete:
                if self.delete(point["id"]):
                    deleted_count += 1

            return deleted_count

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return 0

    def time_range_vector_search(
        self,
        query_vector: List[float],
        time_range: tuple[str, str],
        keywords: List[str],
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors within a specific time range

        Args:
            query_vector: The query embedding vector
            time_range: Tuple of (start_time, end_time) in ISO format
            keywords: List of keywords for filtering
            limit: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing search results with scores and payloads
        """
        try:
            start_time, end_time = time_range
            
            # Convert ISO format strings to timestamps
            start_timestamp = datetime.fromisoformat(start_time).timestamp()
            end_timestamp = datetime.fromisoformat(end_time).timestamp()
            
            # Create time range filter combined with keyword matching
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="published_date",
                        match=models.Range(
                            gte=start_timestamp,
                            lte=end_timestamp
                        )
                    )
                ],
                should=[
                    models.FieldCondition(
                        key="keywords",
                        match=models.MatchAny(any=keywords)
                    )
                ]
            )

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )

            return [{
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]

        except Exception as e:
            logger.error(f"Error during time range vector search: {str(e)}")
            return []
