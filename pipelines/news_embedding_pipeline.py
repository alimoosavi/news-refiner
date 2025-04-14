import logging
from typing import List, Dict, Any
from datetime import datetime

from db.db_manager import DBManager, RawNews
from config import config
from processors.news_categorizer import NewsCategorizer
from vector_database.vector_database import VectorDatabaseManager
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class NewsProcessingPipeline:
    def __init__(
            self,
            db_manager: DBManager,
            vector_db_manager: VectorDatabaseManager,
            categorizer: NewsCategorizer,
            batch_size: int = 25
    ):
        self.db_manager = db_manager
        self.vector_db = vector_db_manager
        self.categorizer = categorizer
        self.batch_size = batch_size
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.openai.api_key,
            max_retries=config.openai.max_retries,
            timeout=config.openai.timeout
        )

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

    async def run(self, limit: int = 100) -> Dict[str, Any]:
        """Run the pipeline"""
        start_time = datetime.now()
        total_stats = {
            "total_processed": 0,
            "categories": {},
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

            # Process in batches
            for i in range(0, len(news_items), self.batch_size):
                batch = news_items[i:i + self.batch_size]
                batch_stats = await self.process_batch(batch)
                
                # Update total stats
                total_stats["total_processed"] += batch_stats["total_processed"]
                total_stats["failed"] += batch_stats["failed"]
                for category, count in batch_stats["categories"].items():
                    if category not in total_stats["categories"]:
                        total_stats["categories"][category] = 0
                    total_stats["categories"][category] += count

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