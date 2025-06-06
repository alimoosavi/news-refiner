import asyncio
import logging
from datetime import timedelta

from cache.cache_manager import CacheManager
from celery_app import app
from collectors.concurrent_telegram_collector import ConcurrentTelegramChannelsCollector
from config import config
from db.db_manager import DBManager

# Remove duplicate app.conf.update configuration here
# The configuration should only be in celery_app.py


async def fetch_news(db_manager, cache_manager, logger):
    """Fetch news from Telegram channels, filter based on last observed date, and update the database."""

    logger.info("Starting to fetch news ...")

    collector = ConcurrentTelegramChannelsCollector(api_id=config.messenger.api_id,
                                                    api_hash=config.messenger.api_hash,
                                                    logger=logger)

    await collector.start()
    news_data = await collector.fetch_selected_channels()
    await collector.stop()

    # Process and filter news
    filtered_news = {}

    for channel, news_list in news_data.items():
        last_published_date = cache_manager.get(channel)

        fresh_news = [
            news for news in news_list
            if last_published_date is None or news['published_date'] > last_published_date
        ]

        if fresh_news:
            latest_date = max(news['published_date'] for news in fresh_news)

            cache_manager.set(channel, latest_date)

            db_manager.store_raw_news(fresh_news)
            filtered_news[channel] = fresh_news

    logger.info(f"Fetched {sum(len(msgs) for msgs in filtered_news.values())} new news_list.")


@app.task(bind=True, max_retries=3)
def collect_news(self):
    """Celery task for collecting news from Telegram channels"""
    logger = logging.getLogger("news_collector")

    try:
        # Initialize fresh connections for each task
        db_manager = DBManager(
            user=config.database.user,
            password=config.database.passkey,
            host=config.database.hostname,
            port=config.database.port,
            database=config.database.name,
            connector=config.database.connector,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow
        )
        cache_manager = CacheManager(file_path=config.cache.file_path)

        # Run async collection
        asyncio.run(fetch_news(db_manager, cache_manager, logger))

    except Exception as e:
        logger.error(f"Collection failed: {str(e)}")
        self.retry(
            exc=e,
            countdown=60,
            max_retries=config.collector.max_retries
        )


def start_worker():
    """Entry point for collectors worker process"""
    app.worker_main()

if __name__ == '__main__':
    start_worker()
