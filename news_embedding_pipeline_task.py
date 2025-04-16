import logging
import os
import asyncio
from celery import Celery
from datetime import timedelta

# Set environment variable to use 'spawn' instead of 'fork'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from config import config
from db.db_manager import DBManager
from processors.news_preprocessor import NewsPreprocessor
from pipelines.news_embedding_pipeline import NewsProcessingPipeline
from vector_database.vector_database import VectorDatabaseManager

# Configure Celery
app = Celery('news_processor', broker=config.celery.broker_url)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_expires=3600,
    worker_concurrency=1,
    task_time_limit=config.celery.task_time_limit,
    # Use 'solo' pool to avoid forking issues on macOS
    worker_pool='solo',
    beat_schedule={
        'process-news-embeddings': {
            'task': 'news_embedding_pipeline_task.process_news_embeddings',
            'schedule': timedelta(minutes=1),  # Changed from 5 to 1
            'kwargs': {'limit': 5}
        }
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.task(bind=True, max_retries=3)
def process_news_embeddings(self, limit: int = 10):
    """Celery task for processing news embeddings"""
    try:
        db_manager = DBManager(
            user=config.database.user,
            password=config.database.passkey,
            host=config.database.hostname,
            port=config.database.port,
            database=config.database.name
        )

        vector_database_manager = VectorDatabaseManager(
            host=config.vector_db.host,
            port=config.vector_db.port,
            embedding_dim=config.openai.embedding_dim
        )

        preprocessor = NewsPreprocessor(logger)

        # Initialize pipeline
        pipeline = NewsProcessingPipeline(
            db_manager=db_manager,
            vector_db_manager=vector_database_manager,
            preprocessor=preprocessor,
            batch_size=config.processing.batch_size
        )

        # Run the pipeline
        stats = asyncio.run(pipeline.run(limit=limit))
        logger.info(f"Pipeline stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Pipeline task failed: {str(e)}")
        self.retry(exc=e, countdown=60)
    finally:
        if 'db_manager' in locals():
            db_manager.close_connections()

def start_worker():
    """Entry point for worker process"""
    app.worker_main()

if __name__ == "__main__":
    start_worker()
