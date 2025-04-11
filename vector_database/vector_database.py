import os
import pickle
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np

from vector_database_manager import logger


class VectorDatabaseManager:
    """
    Manager for vector database operations using FAISS
    Handles storing, loading, and managing vector embeddings and their associated metadata
    """

    def __init__(
            self,
            index_path: str = "./vector_db/index.faiss",
            metadata_path: str = "./vector_db/metadata.pkl",
            embedding_dim: int = 1536,
            index_type: str = "flat",
            save_every: int = 100,
    ):
        """
        Initialize the vector database manager

        Args:
            index_path: Path to store the FAISS index
            metadata_path: Path to store metadata associated with vectors
            embedding_dim: Dimension of embeddings to store
            index_type: Type of FAISS index ('flat', 'ivf', etc.)
            save_every: Automatically save every N additions
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.save_every = save_every

        # Counter to track additions for periodic saving
        self.addition_counter = 0

        # Metadata storage
        self.metadata = []

        # Initialize the index
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the vector database, creating or loading an existing index"""
        # Create directory for files if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Check if index already exists
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a new FAISS index based on the specified type"""
        logger.info(f"Creating new FAISS index with dimension {self.embedding_dim}")

        if self.index_type == "flat":
            # Standard L2 distance index (exact but slower)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            nlist = 100  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
            self.index.train(np.random.rand(1000, self.embedding_dim).astype(np.float32))
        elif self.index_type == "hnsw":
            # HNSW index for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 is M parameter
        else:
            # Default to flat index if type is not recognized
            logger.warning(f"Unknown index type '{self.index_type}', defaulting to flat index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Initialize empty metadata list
        self.metadata = []

    def _load(self) -> None:
        """Load existing index and metadata from disk"""
        try:
            logger.info(f"Loading existing index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)

            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")

            # Validate consistency between index and metadata
            if hasattr(self.index, 'ntotal') and self.index.ntotal != len(self.metadata):
                logger.warning(
                    f"Inconsistency detected: index has {self.index.ntotal} vectors but metadata has {len(self.metadata)} entries"
                )
        except Exception as e:
            logger.error(f"Error loading index or metadata: {str(e)}")
            logger.info("Creating new index and metadata")
            self._create_new_index()

    def save(self) -> bool:
        """
        Save the index and metadata to disk

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Saving index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)

            logger.info(f"Saving metadata to {self.metadata_path}")
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            logger.info(f"Saved index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving index or metadata: {str(e)}")
            return False

    def store(self, vector: List[float], payload: Dict[str, Any]) -> int:
        """
        Store a vector and its associated payload in the database

        Args:
            vector: The embedding vector to store
            payload: Dictionary containing metadata and content to associate with the vector

        Returns:
            int: Index of the stored vector or -1 if failed
        """
        try:
            # Convert vector to numpy array
            vector_np = np.array([vector]).astype('float32')

            # Store the vector
            idx = self.index.ntotal  # Current index before adding
            self.index.add(vector_np)

            # Store the payload
            self.metadata.append(payload)

            # Increment addition counter
            self.addition_counter += 1

            # Check if we should save based on addition count
            if self.addition_counter >= self.save_every:
                self.save()
                self.addition_counter = 0

            return idx
        except Exception as e:
            logger.error(f"Error storing vector: {str(e)}")
            return -1

    def store_batch(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Store a batch of vectors and their payloads

        Args:
            vectors: List of embedding vectors
            payloads: List of payloads corresponding to the vectors

        Returns:
            Tuple[int, int]: (start_index, count) of stored vectors
        """
        if not vectors or len(vectors) != len(payloads):
            logger.error("Invalid batch: vectors and payloads must be non-empty and have the same length")
            return (-1, 0)

        try:
            # Convert vectors to numpy array
            vectors_np = np.array(vectors).astype('float32')

            # Get current index before adding
            start_idx = self.index.ntotal

            # Add vectors to index
            self.index.add(vectors_np)

            # Add payloads to metadata
            self.metadata.extend(payloads)

            # Increment addition counter and check if we should save
            self.addition_counter += len(vectors)
            if self.addition_counter >= self.save_every:
                self.save()
                self.addition_counter = 0

            return (start_idx, len(vectors))
        except Exception as e:
            logger.error(f"Error storing batch of vectors: {str(e)}")
            return (-1, 0)

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
            # Convert query to numpy array
            query_np = np.array([query_vector]).astype('float32')

            # Search in the index
            distances, indices = self.index.search(query_np, k)

            # Format results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]

                # Skip invalid indices
                if idx == -1 or idx >= len(self.metadata):
                    continue

                # Get metadata for this vector
                meta = self.metadata[idx]

                # Convert distance to similarity score (closer to 1 is better)
                # L2 distance: smaller is better, so we invert it
                score = float(1.0 / (1.0 + distances[0][i]))

                # Add to results
                results.append({
                    "id": idx,
                    "score": score,
                    "distance": float(distances[0][i]),  # Original distance
                    "payload": meta
                })

            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []

    def get_payload(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get the payload for a specific vector by index

        Args:
            idx: Index of the vector

        Returns:
            Payload dictionary or None if not found
        """
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None

    def delete(self, idx: int) -> bool:
        """
        Delete a vector and its metadata
        Note: FAISS doesn't support direct deletion in all index types,
        so this implementation marks metadata as deleted but keeps the vector

        Args:
            idx: Index of the vector to delete

        Returns:
            bool: True if marked as deleted, False otherwise
        """
        try:
            if 0 <= idx < len(self.metadata):
                # Mark as deleted in metadata
                self.metadata[idx]["deleted"] = True
                return True
            return False
        except Exception as e:
            logger.error(f"Error marking vector as deleted: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database

        Returns:
            Dictionary with stats
        """
        active_count = sum(1 for m in self.metadata if not m.get("deleted", False))

        return {
            "total_vectors": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            "active_vectors": active_count,
            "deleted_vectors": len(self.metadata) - active_count,
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "index_path": self.index_path,
            "metadata_path": self.metadata_path
        }
