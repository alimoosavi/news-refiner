import asyncio
import logging
from datetime import datetime

from config import config
from db.db_manager import DBManager
from processors.news_categorizer import NewsCategorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("news_categorizer")


async def process_news_batch():
    """Process and categorize a batch of news articles"""
    try:
        # Initialize DB manager
        db_manager = DBManager(
            user=config.database.user,
            password=config.database.passkey,
            host=config.database.hostname,
            port=config.database.port,
            database=config.database.name,
            logger=logger
        )

        # Initialize categorizer
        categorizer = NewsCategorizer(logger)

        # Get start time
        start_time = datetime.now()
        logger.info("Starting news categorization process")

        # Process batch
        categories = await categorizer.categorize_batch(db_manager, batch_size=10)

        # Log results
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Categorized {len(categories)} articles in {processing_time:.2f} seconds")

        # Print category distribution
        category_counts = {}
        for category in categories.values():
            category_counts[category] = category_counts.get(category, 0) + 1

        logger.info("Category distribution:")
        for category, count in category_counts.items():
            logger.info(f"{category}: {count} articles")

    except Exception as e:
        logger.error(f"News categorization failed: {str(e)}")
        raise
    finally:
        if 'db_manager' in locals():
            db_manager.close_connections()


def main():
    """Main entry point"""
    try:
        asyncio.run(process_news_batch())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")


if __name__ == "__main__":
    main()
