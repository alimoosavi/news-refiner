import logging

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings

from config import config
from processors.news_preprocessor import QueryPreprocessor
from vector_database.vector_database import VectorDatabaseManager

logger = logging.getLogger(__name__)

from .reranker import Reranker


class Retriever:
    # Time windows in days for progressive search
    TIME_WINDOWS = [(0, 2), (2, 7), (7, 30)]
    MIN_RELEVANT_RESULTS = 3  # Minimum number of relevant results before stopping

    def __init__(
            self,
            vector_db_manager: VectorDatabaseManager,
            query_preprocessor: QueryPreprocessor,
            openai_api_key: Optional[str] = None,
            model_name: str = "text-embedding-ada-002",
            max_retries: int = 3,
            default_top_k: int = 5,
            similarity_threshold: float = 0.6
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
        self.reranker = Reranker()

    async def search(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            metadata_filters: Optional[Dict[str, Any]] = None,
            use_keywords: bool = True
    ) -> List[Dict[str, Any]]:
        """Search using semantic vector search followed by optional keyword filtering"""
        try:
            # Process query and generate embedding
            query_chunk = await self.query_preprocessor.process_query(query)
            if not query_chunk:
                logger.warning("No valid query chunk generated")
                return []
                
            query_embedding = await self.embeddings.aembed_query(query_chunk.content)
            
            # Perform pure semantic search
            results = self.vector_db.search(
                query_vector=query_embedding,
                limit=(top_k or self.default_top_k) * 2,  # Get more results for keyword filtering
                score_threshold=threshold or self.similarity_threshold,
                metadata_filters=metadata_filters
            )
            
            if not results:
                return []
            
            # Format results first
            formatted_results = self._format_results(results)
            
            # Apply keyword filtering if enabled and keywords exist
            if use_keywords and query_chunk.keywords:
                filtered_results = self._filter_by_keywords(formatted_results, query_chunk.keywords)
                # If keyword filtering returns too few results, fall back to semantic-only
                if len(filtered_results) < (top_k or self.default_top_k) // 2:
                    logger.info(f"Keyword filtering returned only {len(filtered_results)} results, using semantic-only")
                    final_results = formatted_results
                else:
                    final_results = filtered_results
            else:
                final_results = formatted_results
            
            # Rerank the final results
            if final_results:
                reranked_results = await self.reranker.rerank(
                    query=query,
                    results=final_results,
                    top_k=top_k or self.default_top_k
                )
                return reranked_results
            
            return []

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _filter_by_keywords(self, results: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """Filter semantic search results by keyword matching"""
        filtered_results = []
        
        for result in results:
            content = result.get("content", "").lower()
            metadata_keywords = result.get("metadata", {}).get("keywords", [])
            
            # Check if any query keyword appears in content or metadata keywords
            keyword_match = False
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check in content
                if keyword_lower in content:
                    keyword_match = True
                    break
                # Check in metadata keywords
                if any(keyword_lower in meta_kw.lower() for meta_kw in metadata_keywords):
                    keyword_match = True
                    break
            
            if keyword_match:
                filtered_results.append(result)
        
        logger.info(f"Keyword filtering: {len(results)} -> {len(filtered_results)} results")
        return filtered_results

    async def search_by_date_range(
            self,
            query: str,
            start_date: str,
            end_date: str,
            top_k: Optional[int] = None,
            use_keywords: bool = True
    ) -> List[Dict[str, Any]]:
        """Search within a date range using semantic search + optional keyword filtering"""
        return await self.search(
            query=query,
            top_k=top_k,
            use_keywords=use_keywords,
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
            top_k: Optional[int] = None,
            use_keywords: bool = True
    ) -> List[Dict[str, Any]]:
        """Search from a specific source using semantic search + optional keyword filtering"""
        return await self.search(
            query=query,
            top_k=top_k,
            use_keywords=use_keywords,
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
                "metadata": metadata,
                "relevance_explanation": ""  # Add empty relevance explanation field
            })

        return formatted
