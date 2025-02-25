import asyncio
import logging
from collections import defaultdict

from config import config
from db_manager import DBManager
from news_crawler_client import NewsCrawlerClient
from schema import ShortNews
from utils import preprocess_persian_document, extract_link, extract_title

NEWS_SOURCES = ['IRNA', 'ISNA', 'FARS', 'JAHAN_FOURI']
LINKS_BATCH_SIZE = 40


async def process_news(db_manager: DBManager, news_crawler_client: NewsCrawlerClient, logger: logging.Logger):
    """Processes unprocessed news by extracting links, fetching new content, and storing results."""
    short_news_mapping = defaultdict(dict)
    news_links_mapping = defaultdict(dict)

    # unprocessed_news = db_manager.get_unprocessed_news()
    non_title_news = []
    for item in db_manager.get_raw_news_by_source_and_status(source='JAHAN_FOURI', has_processed=True):
        title = extract_title('JAHAN_FOURI', item.content)
        if title is None:
            non_title_news.append(item.content)
        print('title', title, '\n', item.content, '\n\n', '-' * 10, '\n\n')

    print('\n\n', 'non_title_news  ', non_title_news)
    # if not unprocessed_news:
    #     logger.info("No unprocessed news found.")
    #     return
    #
    # for item in unprocessed_news:
    #     links = extract_link(content=item.content,
    #                          source=item.source)
    #     if len(links) > 0:
    #         news_links_mapping[item.source] = {**news_links_mapping[item.source],
    #                                            item.id: links}
    #     else:
    #         short_news_mapping[item.source] = {**short_news_mapping[item.source], item.id: ShortNews(
    #             source=item.source,
    #             body=preprocess_persian_document(item.content),
    #             timestamp=item.published_date,
    #         )}
    #
    # # Handling short-news
    # short_news_flat = [
    #     short_news
    #     for short_news_dict in short_news_mapping.values()
    #     for short_news in short_news_dict.values()
    # ]
    #
    # raw_news_ids = [
    #     news_id
    #     for short_news_dict in short_news_mapping.values()
    #     for news_id in short_news_dict.keys()
    # ]
    #
    # db_manager.process_and_insert_news(news_list=short_news_flat, raw_news_ids=raw_news_ids)


async def main():
    """Main function to initialize components and start processing news."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("news_collector")

    # Initialize DBManager
    db_manager = DBManager(
        db_name=config.database.name,
        user=config.database.user,
        password=config.database.password,
        host=config.database.hostname,
        port=config.database.port,
    )

    # Initialize NewsCrawlerClient
    news_crawler_client = NewsCrawlerClient(base_url=config.news_crawler.base_url)

    # Process news asynchronously
    await process_news(db_manager=db_manager, news_crawler_client=news_crawler_client, logger=logger)

    # Close DB connection
    db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
