import logging
import time
from typing import List, Dict, Any, Optional

# For embedding generation
from langchain_openai import OpenAIEmbeddings

# Import the Vector Database Manager
from vector_database.vector_database import VectorDatabaseManager

# Configure logging
logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever class for querying and retrieving the most similar chunks
    to a given query from the vector database
    """

    def __init__(
            self,
            vector_db_manager: VectorDatabaseManager,
            openai_api_key: Optional[str] = None,
            model_name: str = "text-embedding-ada-002",
            max_retries: int = 3,
            retry_delay: int = 2,
            default_top_k: int = 5,
            similarity_threshold: float = 0.6,
    ):
        """
        Initialize the Retriever

        Args:
            vector_db_manager: Vector database manager for retrieving embeddings
            openai_api_key: OpenAI API key for embeddings
            model_name: Name of the embedding model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            default_top_k: Default number of results to return
            similarity_threshold: Minimum similarity score threshold (0-1)
        """
        self.vector_db = vector_db_manager
        self.openai_api_key = openai_api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold

        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.openai_api_key
        )

    def get_embedding_with_retries(self, text: str) -> List[float]:
        """
        Get embedding with retry logic

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If all retries fail
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Failed to get embedding after {self.max_retries} retries: {str(e)}")
                    raise

                logger.warning(f"Embedding error (retry {retries}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar chunks to the query

        Args:
            query: The search query text
            top_k: Number of results to return (defaults to self.default_top_k)
            threshold: Minimum similarity score (defaults to self.similarity_threshold)
            metadata_filters: Optional filters to apply to metadata fields

        Returns:
            List of similar chunks with their metadata and similarity scores
        """
        try:
            # Set defaults if not provided
            if top_k is None:
                top_k = self.default_top_k
            if threshold is None:
                threshold = self.similarity_threshold

            # Get the query embedding
            query_embedding = self.get_embedding_with_retries(query)

            # Search the vector database
            raw_results = self.vector_db.search(query_embedding, k=top_k * 2)  # Get more results to filter

            # Filter results
            filtered_results = self._filter_results(raw_results, threshold, metadata_filters)

            # Limit to top_k after filtering
            limited_results = filtered_results[:top_k]

            # Format the results
            formatted_results = self._format_results(limited_results)

            logger.info(f"Retrieved {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving results for query '{query[:50]}...': {str(e)}")
            return []

    def retrieve_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by their vector IDs

        Args:
            ids: List of vector IDs to retrieve

        Returns:
            List of chunks with their metadata
        """
        results = []
        for idx in ids:
            payload = self.vector_db.get_payload(idx)
            if payload:
                results.append({
                    "id": idx,
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text" and k != "deleted"}
                })
        return results

    def _filter_results(
            self,
            results: List[Dict[str, Any]],
            threshold: float,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter results by threshold and metadata filters

        Args:
            results: Raw results from vector search
            threshold: Minimum similarity score
            metadata_filters: Optional filters to apply to metadata fields

        Returns:
            Filtered results
        """
        # First filter by threshold
        threshold_filtered = [r for r in results if r["score"] >= threshold]

        # If no metadata filters, return threshold filtered results
        if not metadata_filters:
            return threshold_filtered

        # Apply metadata filters
        filtered_results = []
        for result in threshold_filtered:
            payload = result["payload"]
            include = True

            # Check each filter condition
            for key, value in metadata_filters.items():
                # Skip text field which is handled separately
                if key == "text":
                    continue

                if key not in payload:
                    include = False
                    break

                # Handle different filter types
                if isinstance(value, list):
                    # List means "one of these values"
                    if payload[key] not in value:
                        include = False
                        break
                elif isinstance(value, dict):
                    # Dict with operators like {"$gt": 5}
                    for op, op_val in value.items():
                        if op == "$gt" and not payload[key] > op_val:
                            include = False
                            break
                        elif op == "$gte" and not payload[key] >= op_val:
                            include = False
                            break
                        elif op == "$lt" and not payload[key] < op_val:
                            include = False
                            break
                        elif op == "$lte" and not payload[key] <= op_val:
                            include = False
                            break
                        elif op == "$ne" and payload[key] == op_val:
                            include = False
                            break
                else:
                    # Direct equality comparison
                    if payload[key] != value:
                        include = False
                        break

            if include:
                filtered_results.append(result)

        return filtered_results

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format the raw results into a more usable structure

        Args:
            results: Raw results from vector search

        Returns:
            Formatted results
        """
        formatted = []
        for result in results:
            payload = result["payload"]

            # Extract text content
            text = payload.get("text", "")

            # Extract all other fields as metadata, excluding 'text' and 'deleted'
            metadata = {k: v for k, v in payload.items() if k != "text" and k != "deleted"}

            formatted.append({
                "id": result["id"],
                "score": result["score"],
                "text": text,
                "metadata": metadata
            })

        return formatted

    def retrieve_by_source(self, source: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by source

        Args:
            source: Source identifier to filter by
            top_k: Maximum number of results to return

        Returns:
            List of chunks from the specified source
        """
        return self.retrieve("", top_k=top_k, metadata_filters={"source": source})

    def retrieve_by_date_range(
            self,
            start_date: str,
            end_date: str,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by published date range

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            top_k: Maximum number of results to return

        Returns:
            List of chunks within the date range
        """
        return self.retrieve(
            "",
            top_k=top_k,
            metadata_filters={
                "published_date": {
                    "$gte": start_date,
                    "$lte": end_date + "T23:59:59"  # Include the entire end day
                }
            }
        )

    def hybrid_search(
            self,
            query: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic search with metadata filtering

        Args:
            query: Search query text
            metadata_filters: Filters to apply to metadata fields
            top_k: Number of results to return

        Returns:
            List of relevant chunks
        """
        # Retrieve based on vector similarity and metadata filters
        return self.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)