import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from db.db_manager import DBManager, RawNews
from config import config
from processors.news_preprocessor import NewsPreprocessor
from vector_database.vector_database import VectorDatabaseManager
from graph_database.graph_database import GraphDatabaseManager
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class NewsProcessingPipeline:
    def __init__(
            self,
            db_manager: DBManager,
            vector_db_manager: VectorDatabaseManager,
            graph_db_manager: GraphDatabaseManager,
            preprocessor: NewsPreprocessor,
            batch_size: int = 25,
            similarity_threshold: float = 0.7,
            temporal_window_hours: int = 72
    ):
        self.db_manager = db_manager
        self.vector_db = vector_db_manager
        self.graph_db = graph_db_manager
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.temporal_window_hours = temporal_window_hours
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.openai.api_key
        )

    async def _process_news_item(self, item: RawNews) -> bool:
        """Process a single news item and build graph connections"""
        try:
            # Process chunks as before
            chunks = await self.preprocessor.process_news(item.content)
            if not chunks:
                return False

            meaningful_chunks = [chunk for chunk in chunks if chunk.is_meaningful]
            if not meaningful_chunks:
                return False

            # Store chunks and build graph
            chunk_embeddings = []
            chunk_ids = []
            
            for chunk in meaningful_chunks:
                # Generate embedding
                embedding = await self.embeddings.aembed_query(chunk.content)
                chunk_embeddings.append(embedding)
                
                # Store in vector database
                metadata = {
                    "news_id": item.id,
                    "source": item.source,
                    "published_date": item.published_date.isoformat(),
                    "content": chunk.content,
                    "keywords": chunk.keywords,
                    "website_link": chunk.website_link
                }
                
                chunk_id = self.vector_db.store(
                    vector=embedding,
                    payload=metadata
                )
                chunk_ids.append(chunk_id)
                
                # Add to graph database
                self.graph_db.add_chunk(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    metadata=metadata
                )
                
                # Add entity edges
                self.graph_db.add_entity_edges(chunk_id, chunk.keywords)

            # Build semantic similarity edges
            embeddings_array = np.array(chunk_embeddings)
            similarities = cosine_similarity(embeddings_array)
            
            for i in range(len(chunk_ids)):
                for j in range(i + 1, len(chunk_ids)):
                    similarity = similarities[i][j]
                    if similarity >= self.similarity_threshold:
                        self.graph_db.add_semantic_edge(
                            chunk_ids[i],
                            chunk_ids[j],
                            float(similarity)
                        )

            # Build temporal edges with recent chunks
            recent_chunks = self._get_recent_chunks(
                item.published_date,
                exclude_ids=chunk_ids
            )
            
            for recent in recent_chunks:
                time_diff = (item.published_date - recent["published_date"]).total_seconds() / 3600
                if time_diff <= self.temporal_window_hours:
                    for chunk_id in chunk_ids:
                        self.graph_db.add_temporal_edge(
                            chunk_id,
                            recent["id"],
                            time_diff
                        )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process news {item.id}: {str(e)}")
            return False

    def _get_recent_chunks(
            self,
            current_date: datetime,
            exclude_ids: List[str],
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recently processed chunks for temporal edge building"""
        start_date = current_date - timedelta(hours=self.temporal_window_hours)
        
        # Use vector database's search capabilities to find recent chunks
        results = self.vector_db.search(
            query_vector=None,  # No semantic search needed
            metadata_filters={
                "published_date": {
                    "$gte": start_date.isoformat(),
                    "$lt": current_date.isoformat()
                }
            },
            limit=limit
        )
        
        return [r for r in results if str(r["id"]) not in exclude_ids]

    async def run(self, limit: int = 100) -> Dict[str, Any]:
        """Run the pipeline"""
        start_time = datetime.now()
        total_stats = {
            "total_processed": 0,
            "total_chunks": 0,
            "meaningful_chunks": 0,
            "failed": 0,
            "duration_seconds": 0,
            "items_per_second": 0
        }

        try:
            # Get unprocessed news
            news_items = self.db_manager.get_unprocessed_news(limit=limit)
            if not news_items:
                logger.info("No unprocessed news found")
                return total_stats

            # Process each news item
            processed_ids = set()
            for item in news_items:
                try:
                    if await self._process_news_item(item):
                        processed_ids.add(item.id)
                        total_stats["total_processed"] += 1
                    else:
                        total_stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Failed to process news {item.id}: {str(e)}")
                    total_stats["failed"] += 1
    
            # Mark successfully processed items
            if processed_ids:
                self.db_manager.mark_news_as_processed(list(processed_ids))

            # Calculate timing stats
            duration = (datetime.now() - start_time).total_seconds()
            total_stats["duration_seconds"] = duration
            total_stats["items_per_second"] = (
                total_stats["total_processed"] / duration if duration > 0 else 0
            )

            return total_stats

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    async def _generate_embedding(self, content: str) -> List[float]:
        """Generate embedding for news content"""
        try:
            return await self.embeddings.aembed_query(content)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    async def _store_in_vector_db(self, item: RawNews, embedding: List[float], category: str) -> bool:
        """Store news item in vector database"""
        try:
            collection_name = f"news_{category.lower()}"
            metadata = {
                "news_id": item.id,
                "source": item.source,
                "category": category,
                "published_date": item.published_date.isoformat(),
                "processed_date": datetime.now().isoformat(),
                "content": item.content
            }
            
            point_id = self.vector_db.store(
                vector=embedding,
                payload=metadata,
                collection_name=collection_name
            )
            return point_id is not None
        except Exception as e:
            logger.error(f"Vector storage failed for news {item.id}: {str(e)}")
            return False

    async def process_batch(self, news_items: List[RawNews]) -> Dict[str, Any]:
        """Process a batch of news items"""
        stats = {
            "total_processed": 0,
            "categories": {},
            "failed": 0
        }
        
        try:
            # Step 1: Categorize all news items
            categories = await self.categorizer.categorize_batch(self.db_manager, len(news_items))
            if not categories:
                logger.warning("No categories returned from categorizer")
                return stats

            # Track successfully processed items
            processed_ids = set()
            
            # Step 2: Process each news item
            for item in news_items:
                if item.id not in categories:
                    logger.warning(f"No category assigned for news {item.id}")
                    stats["failed"] += 1
                    continue

                try:
                    category = categories[item.id]
                    
                    # Generate embedding
                    embedding = await self._generate_embedding(item.content)
                    
                    # Store in vector database
                    if await self._store_in_vector_db(item, embedding, category):
                        processed_ids.add(item.id)
                        stats["total_processed"] += 1
                        
                        # Update category stats
                        stats["categories"][category] = stats["categories"].get(category, 0) + 1
                    else:
                        stats["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process news {item.id}: {str(e)}")
                    stats["failed"] += 1
                    continue

            # Mark successfully processed items
            if processed_ids:
                self.db_manager.mark_news_as_processed(list(processed_ids))

            return stats

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise