from typing import List, Dict, Any, Optional
import logging
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
            logger.error(f"Error storing vector: {str(e)}")
            return None

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
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: The query embedding vector
            collection_name: Optional name of collection to search in
            limit: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing search results with scores and payloads
        """
        try:
            # Use specified collection or default
            collection = collection_name or self.collection_name
            
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
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
