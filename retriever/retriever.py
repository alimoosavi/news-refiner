import logging
import time
from typing import List, Dict, Any, Optional

# For embedding generation
from langchain_openai import OpenAIEmbeddings

from vector_database.vector_database import VectorDatabaseManager
from processors.news_categorizer import NewsCategorizer
from config import config

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
            categorizer: NewsCategorizer = None,
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
            categorizer: NewsCategorizer instance for query categorization
            openai_api_key: OpenAI API key for embeddings
            model_name: Name of the embedding model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            default_top_k: Default number of results to return
            similarity_threshold: Minimum similarity score threshold (0-1)
        """
        self.vector_db = vector_db_manager
        self.categorizer = categorizer
        self.openai_api_key = openai_api_key or config.openai.api_key
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

    async def categorize_query(self, query: str) -> str:
        """
        Categorize a query using the NewsCategorizer

        Args:
            query: The query text to categorize

        Returns:
            Category name (e.g., "politics", "sports", etc.)
        """
        if not self.categorizer:
            logger.warning("No categorizer provided, using default collection")
            return "general"
            
        try:
            # Use the categorizer to determine the query's category
            category = await self.categorizer.categorize_text(query)
            logger.info(f"Query categorized as: {category}")
            return category
        except Exception as e:
            logger.error(f"Error categorizing query: {str(e)}")
            return "general"

    async def retrieve_by_topic(
            self,
            query: str,
            topic: Optional[str] = None,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar news articles to the query, filtered by topic

        Args:
            query: The search query text
            topic: Optional topic to filter by (if None, will be auto-detected)
            top_k: Number of results to return (defaults to self.default_top_k)
            threshold: Minimum similarity score (defaults to self.similarity_threshold)
            metadata_filters: Optional filters to apply to metadata fields

        Returns:
            List of similar news articles with their metadata and similarity scores
        """
        try:
            # Set defaults if not provided
            if top_k is None:
                top_k = self.default_top_k
            if threshold is None:
                threshold = self.similarity_threshold
                
            # Determine the topic/category if not provided
            if topic is None and self.categorizer:
                topic = await self.categorize_query(query)
            
            # Get the query embedding
            query_embedding = self.get_embedding_with_retries(query)
            
            # Determine which collection to search based on topic
            collection_name = topic.lower() if topic else None
            
            # Search the vector database in the specific collection
            raw_results = self.vector_db.search(
                query_vector=query_embedding,
                collection_name=collection_name,
                limit=top_k * 2,  # Get more results to filter
                score_threshold=threshold
            )
            
            # Format the results
            formatted_results = self._format_results(raw_results)
            
            # Apply additional metadata filters if provided
            if metadata_filters:
                formatted_results = self._apply_metadata_filters(formatted_results, metadata_filters)
            
            # Limit to top_k after filtering
            limited_results = formatted_results[:top_k]
            
            logger.info(f"Retrieved {len(limited_results)} results for query: {query[:50]}... in topic: {topic}")
            return limited_results

        except Exception as e:
            logger.error(f"Error retrieving results for query '{query[:50]}...' in topic {topic}: {str(e)}")
            return []

    def _apply_metadata_filters(
            self,
            results: List[Dict[str, Any]],
            metadata_filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to results

        Args:
            results: Formatted results to filter
            metadata_filters: Filters to apply to metadata fields

        Returns:
            Filtered results
        """
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include = True
            
            # Check each filter condition
            for key, value in metadata_filters.items():
                if key not in metadata:
                    include = False
                    break
                    
                # Handle different filter types
                if isinstance(value, list):
                    # List means "one of these values"
                    if metadata[key] not in value:
                        include = False
                        break
                elif isinstance(value, dict):
                    # Dict with operators like {"$gt": 5}
                    for op, op_val in value.items():
                        if op == "$gt" and not metadata[key] > op_val:
                            include = False
                            break
                        elif op == "$gte" and not metadata[key] >= op_val:
                            include = False
                            break
                        elif op == "$lt" and not metadata[key] < op_val:
                            include = False
                            break
                        elif op == "$lte" and not metadata[key] <= op_val:
                            include = False
                            break
                        elif op == "$ne" and metadata[key] == op_val:
                            include = False
                            break
                else:
                    # Direct equality comparison
                    if metadata[key] != value:
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
            # Extract payload and score
            payload = result.get("payload", {})
            score = result.get("score", 0.0)
            
            # Extract content from payload
            content = payload.get("content", "")
            
            # Extract all other fields as metadata
            metadata = {k: v for k, v in payload.items() if k != "content"}
            
            formatted.append({
                "id": result.get("id", ""),
                "score": score,
                "content": content,
                "metadata": metadata
            })
            
        return formatted

    async def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar news to the query across all topics

        Args:
            query: The search query text
            top_k: Number of results to return (defaults to self.default_top_k)
            threshold: Minimum similarity score (defaults to self.similarity_threshold)
            metadata_filters: Optional filters to apply to metadata fields

        Returns:
            List of similar news with their metadata and similarity scores
        """
        # Auto-detect the topic and search within that collection
        return await self.retrieve_by_topic(
            query=query,
            topic=None,  # Auto-detect
            top_k=top_k,
            threshold=threshold,
            metadata_filters=metadata_filters
        )

    async def retrieve_by_source(self, query: str, source: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve news by source and query

        Args:
            query: Search query text
            source: Source identifier to filter by
            top_k: Maximum number of results to return

        Returns:
            List of news from the specified source matching the query
        """
        return await self.retrieve(
            query=query, 
            top_k=top_k, 
            metadata_filters={"source": source}
        )

    async def retrieve_by_date_range(
            self,
            query: str,
            start_date: str,
            end_date: str,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve news by published date range and query

        Args:
            query: Search query text
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            top_k: Maximum number of results to return

        Returns:
            List of news within the date range matching the query
        """
        return await self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filters={
                "published_date": {
                    "$gte": start_date,
                    "$lte": end_date + "T23:59:59"  # Include the entire end day
                }
            }
        )


class Retriever:
    def __init__(
            self,
            vector_db_manager: VectorDatabaseManager,
            preprocessor: NewsPreprocessor,
            openai_api_key: Optional[str] = None,
            model_name: str = "text-embedding-ada-002",
            max_retries: int = 3,
            default_top_k: int = 5,
            similarity_threshold: float = 0.6,
    ):
        self.vector_db = vector_db_manager
        self.preprocessor = preprocessor
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api_key or config.openai.api_key
        )
        self.max_retries = max_retries
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold

    async def search(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using hybrid approach (vector similarity + keywords)
        """
        try:
            # Process query to get keywords
            chunks = await self.preprocessor.process_news(query)
            if not chunks or not chunks[0].keywords:
                return []
            
            # Get query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Perform hybrid search
            results = self.vector_db.hybrid_search(
                query_vector=query_embedding,
                keywords=chunks[0].keywords,
                limit=top_k or self.default_top_k,
                score_threshold=threshold or self.similarity_threshold
            )
            
            # Apply additional metadata filters if provided
            if metadata_filters:
                results = self._apply_metadata_filters(results, metadata_filters)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    async def search_by_date_range(
            self,
            query: str,
            start_date: str,
            end_date: str,
            top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search within a date range"""
        return await self.search(
            query=query,
            top_k=top_k,
            metadata_filters={
                "published_date": {
                    "$gte": start_date,
                    "$lte": end_date + "T23:59:59"
                }
            }
        )

    async def search_by_source(
            self,
            query: str,
            source: str,
            top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search from a specific source"""
        return await self.search(
            query=query,
            top_k=top_k,
            metadata_filters={"source": source}
        )