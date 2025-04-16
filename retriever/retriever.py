import logging
import time
from typing import List, Dict, Any, Optional

# For embedding generation
from langchain_openai import OpenAIEmbeddings

from vector_database.vector_database import VectorDatabaseManager
from processors.news_preprocessor import QueryPreprocessor
from config import config

# Configure logging
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
            self,
            vector_db_manager: VectorDatabaseManager,
            query_preprocessor: QueryPreprocessor,
            openai_api_key: Optional[str] = None,
            model_name: str = "text-embedding-ada-002",
            max_retries: int = 3,
            default_top_k: int = 5,
            similarity_threshold: float = 0.6,
    ):
        self.vector_db = vector_db_manager
        self.query_preprocessor = query_preprocessor
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
        """Search using hybrid approach (vector similarity + keywords)"""
        try:
            # Process query to get keywords
            query_chunk = await self.query_preprocessor.process_query(query)
            if not query_chunk or not query_chunk.keywords:
                logger.warning("No valid query chunk or keywords generated")
                return []

            # Get query embedding
            query_embedding = await self.embeddings.aembed_query(query_chunk.content)

            # Perform hybrid search
            results = self.vector_db.hybrid_search(
                query_vector=query_embedding,
                keywords=query_chunk.keywords,
                limit=top_k or self.default_top_k,
                score_threshold=threshold or self.similarity_threshold
            )

            # Apply additional metadata filters if provided
            if metadata_filters:
                results = self._apply_metadata_filters(results, metadata_filters)

            # Format results
            return self._format_results(results)

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

    def _apply_metadata_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to search results"""
        filtered_results = []

        for result in results:
            metadata = result.get("payload", {})
            include = True

            for key, value in filters.items():
                if key not in metadata:
                    include = False
                    break

                # Handle complex filters with operators
                if isinstance(value, dict):
                    for op, op_val in value.items():
                        if op == "$gte" and not metadata[key] >= op_val:
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
        """Format the raw results into a more usable structure"""
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
