# from datetime import timedelta
#
# import numpy as np
# from celery import Celery
# import logging
# from config import config
# from db_manager import DBManager, RawNews
# from faiss_manager import FAISSManager
# from openai_interface import OpenAIInterface
# from utils import count_tokens
#
# # Celery application setup
# app = Celery('news_processor', broker=config.celery.broker_url)
# app.conf.update(
#     task_serializer='json',
#     accept_content=['json'],
#     result_expires=3600,
#     worker_concurrency=config.celery.concurrency,
#     task_time_limit=config.celery.task_time_limit
# )
#
# # Component initialization with config
# faiss_mgr = FAISSManager(
#     index_path=config.faiss.index_path,
#     payload_path=config.faiss.payload_path
# )
# openai = OpenAIInterface(
#     api_key=config.openai.api_key,
#     model=config.openai.model,
#     timeout=config.openai.timeout,
#     max_retries=config.openai.max_retries
# )
# db = DBManager(
#     user=config.database.user,
#     password=config.database.password,
#     host=config.database.hostname,
#     port=config.database.port,
#     database=config.database.name,
#     connector=config.database.connector,
#     pool_size=config.database.pool_size,
#     max_overflow=config.database.max_overflow
# )
#
#
# @app.task(bind=True, max_retries=3)
# def process_news_batch(self, batch_size: int = None):
#     """Process a batch of unprocessed news articles with retry logic"""
#     try:
#         batch_size = batch_size or config.processing.batch_size
#         with db.get_session() as session:
#             # Fetch unprocessed news with pessimistic lock
#             unprocessed_news = session.query(RawNews) \
#                 .filter_by(has_processed=False) \
#                 .limit(batch_size) \
#                 .with_for_update() \
#                 .all()
#
#             if not unprocessed_news:
#                 return {"status": "no unprocessed news"}
#
#             processed_data = [
#                 {
#                     "raw_news_id": news.id,
#                     "source": news.source,
#                     "timestamp": news.timestamp.isoformat(),
#                     "text": news.content,
#                     "token_count": count_tokens(news.content)
#                 }
#
#                 for news in unprocessed_news
#             ]
#
#             # Batch process embeddings
#             embeddings = []
#             text_chunks = [item["text"] for item in processed_data]
#             for i in range(0, len(text_chunks), config.processing.embedding_batch_size):
#                 batch = text_chunks[i:i + config.processing.embedding_batch_size]
#                 embeddings.extend(openai.get_embeddings(batch))
#
#             # Prepare FAISS payloads
#             payloads = [{
#                 "id": f"{item['raw_news_id']}-{idx}",
#                 "metadata": item
#             } for idx, item in enumerate(processed_data)]
#
#             # Update vector store
#             faiss_mgr.add_to_index(
#                 embeddings=np.array(embeddings, dtype=np.float32),
#                 payloads=payloads
#             )
#
#             # Mark as processed
#             news_ids = [news.id for news in unprocessed_news]
#             db.mark_news_as_processed(news_ids)
#
#             # Periodic index persistence
#             if len(payloads) >= config.faiss.save_interval:
#                 faiss_mgr.save_index()
#
#             return {
#                 "processed_articles": len(unprocessed_news),
#                 "created_chunks": len(payloads),
#                 "total_tokens": sum(item["token_count"] for item in processed_data)
#             }
#
#     except Exception as e:
#         self.retry(
#             exc=e,
#             countdown=60,
#             max_retries=config.openai.max_retries
#         )
#
#
# @app.on_after_configure.connect
# def setup_periodic_tasks(sender, **kwargs):
#     """Configure scheduled processing using config values"""
#     sender.add_periodic_task(
#         timedelta(minutes=config.processing.interval_minutes),
#         process_news_batch.s(batch_size=config.processing.batch_size),
#         name='scheduled-news-processing'
#     )
#
#
# def start_worker():
#     """Entry point for worker process"""
#     app.worker_main()
#
#
# if __name__ == '__main__':
#     start_worker()
