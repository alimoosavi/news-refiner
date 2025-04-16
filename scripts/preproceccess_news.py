import asyncio
import logging
from datetime import datetime

from db.db_manager import DBManager
from processors.news_preprocessor import NewsPreprocessor
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def process_sample_news():
    try:
        # Initialize DB Manager
        db_manager = DBManager(
            user=config.database.user,
            password=config.database.passkey,
            host=config.database.hostname,
            port=config.database.port,
            database=config.database.name,
            logger=logger
        )

        # Initialize News Preprocessor
        preprocessor = NewsPreprocessor(logger)

        # Get 10 unprocessed news items
        news_items = db_manager.get_unprocessed_news(limit=10)
        logger.info(f"Retrieved {len(news_items)} unprocessed news items")

        processed_ids = []

        # Process each news item
        for item in news_items:
            try:
                logger.info(f"Processing news ID: {item.id} from source: {item.source}")
                
                # Process the news content
                chunks = await preprocessor.process_news(item.content)
                
                if chunks:
                    logger.info(f"Successfully processed news ID {item.id} into {len(chunks)} chunks")
                    
                    # Print detailed information for each chunk
                    for i, chunk in enumerate(chunks, 1):
                        logger.info(f"\nChunk #{i} for news ID {item.id}:")
                        logger.info(f"Content preview: {chunk.content[:200]}...")
                        logger.info(f"Keywords: {', '.join(chunk.keywords)}")
                        logger.info(f"Is Meaningful: {chunk.is_meaningful}")
                        logger.info(f"Website Link: {chunk.website_link or 'None'}")
                    
                    processed_ids.append(item.id)
                else:
                    logger.warning(f"No valid chunks produced for news ID {item.id}")
                
                logger.info("-" * 80)
                
            except Exception as e:
                logger.error(f"Error processing news ID {item.id}: {str(e)}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
    finally:
        if 'db_manager' in locals():
            db_manager.close_connections()


if __name__ == "__main__":
    asyncio.run(process_sample_news())