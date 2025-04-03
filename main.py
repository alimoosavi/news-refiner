import asyncio
import logging
from collections import defaultdict

from config import config
from db_manager import DBManager
from news_crawler_client import NewsCrawlerClient
from schema import ShortNews
from utils import preprocess_persian_document, extract_link, extract_title_and_body

NEWS_SOURCES = ['IRNA', 'ISNA', 'FARS', 'JAHAN_FOURI']
LINKS_BATCH_SIZE = 40


async def process_news(db_manager: DBManager,
                       news_crawler_client: NewsCrawlerClient,
                       logger: logging.Logger):
    """Processes unprocessed news by extracting links, fetching new content, and storing results."""
    # Todo: handle short news ( which did not detect any link for them)
    news_links_mapping = defaultdict(dict)
    for source in ['IRNA']:
        unprocessed_news = db_manager.get_raw_news_by_source_and_status(source=source,
                                                                        has_processed=False)
        # print(len(unprocessed_news))
        for item in unprocessed_news:
            links = extract_link(content=item.content, source=item.source)
            print(links)


    # for source in ['IRNA']:
    #     short_news = []
    #     for item in db_manager.get_raw_news_by_source_and_status(source=source, has_processed=True):
    #         entry = extract_title_and_body(source, item.content)
    #         if entry is None:
    #             continue
    #
    #         title, body = entry
    #         if title is not None and body is not None:
    #             processed_body = preprocess_persian_document(body)
    #             if len(processed_body) > 0:
    #                 short_news.append(ShortNews(source='JAHAN_FOURI',
    #                                             title=title,
    #                                             body=preprocess_persian_document(body),
    #                                             timestamp=item.published_date))
    #
    #     for item in short_news:
    #         print(item, '\n\n *** \n\n')


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
