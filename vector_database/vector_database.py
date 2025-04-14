from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """
    Manager for vector database operations using Qdrant
    Handles storing and managing vector embeddings and their associated metadata
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6333,
            collection_name: str = "news_vectors",
            embedding_dim: int = 1536,
    ):
        """
        Initialize the Qdrant vector database manager

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            embedding_dim: Dimension of embeddings to store
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize collection
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the vector database, creating collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            raise

    def store(self, vector: List[float], payload: Dict[str, Any]) -> str:
        """
        Store a vector and its associated payload in the database

        Args:
            vector: The embedding vector to store
            payload: Dictionary containing metadata and content

        Returns:
            str: ID of the stored vector or None if failed
        """
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

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: The query embedding vector
            k: Number of results to return

        Returns:
            List of dictionaries containing search results with scores and payloads
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k
            )
            
            return [{
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
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
