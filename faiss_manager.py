import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any


class FAISSManager:
    def __init__(self, index_path: str = "news_index.faiss", payload_path: str = "news_payloads.json"):
        """
        Initialize FAISS manager with separate paths for index and payloads
        
        Args:
            index_path: Path to store/load FAISS index
            payload_path: Path to store/load payload metadata
        """
        self.index_path = index_path
        self.payload_path = payload_path
        self.index = None
        self.payloads = []
        self._dimension = 1536  # OpenAI text-embedding-ada-002 dimension

    def create_index(self, dimension: int = 1536) -> None:
        """Initialize a new FAISS index with L2 distance metric"""
        self._dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.payloads = []

    def add_to_index(self, embeddings: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        """
        Add batch of embeddings with associated payloads to the index

        Args:
            embeddings: numpy array of shape (n, dimension)
            payloads: list of dictionaries with metadata for each embedding
        """
        if self.index is None:
            self.create_index(embeddings.shape[1])

        # Validate input shapes
        if len(embeddings) != len(payloads):
            raise ValueError("Mismatch between embeddings and payloads count")

        if embeddings.shape[1] != self._dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self._dimension}, got {embeddings.shape[1]}")

        # Add to FAISS index and store payloads
        self.index.add(embeddings.astype(np.float32))
        self.payloads.extend(payloads)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the index

        Args:
            query_embedding: numpy array of shape (1, dimension)
            k: number of nearest neighbors to return

        Returns:
            List of payload dictionaries ordered by similarity
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Load or create an index first.")

        # Convert to float32 for FAISS compatibility
        query_embedding = query_embedding.astype(np.float32)

        # Search the index
        distances, indices = self.index.search(query_embedding, k)

        # Return payloads with scores
        return [{
            **self.payloads[i],
            "score": float(distances[0][j])
        } for j, i in enumerate(indices[0])]

    def save_index(self) -> None:
        """Persist index and payloads to disk using configured paths"""
        if self.index is None:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.payload_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save payloads as JSON
        with open(self.payload_path, 'w', encoding='utf-8') as f:
            json.dump(self.payloads, f, ensure_ascii=False, default=str)

    def load_index(self) -> None:
        """Load index and payloads from configured paths"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file {self.index_path} not found")

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        self._dimension = self.index.d

        # Load payloads if they exist
        if os.path.exists(self.payload_path):
            with open(self.payload_path, 'r', encoding='utf-8') as f:
                self.payloads = json.load(f)
        else:
            self.payloads = []

    def get_index_size(self) -> int:
        """Return number of vectors in the index"""
        return self.index.ntotal if self.index else 0

    def clear_index(self) -> None:
        """Reset the index and payloads"""
        self.index = None
        self.payloads = []
        self._dimension = 1536