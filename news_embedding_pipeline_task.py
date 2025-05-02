import asyncio
import logging
import os

from celery_app import app
from config import config
from db.db_manager import DBManager
from graph_database.graph_database import GraphDatabaseManager
from pipelines.news_embedding_pipeline import NewsProcessingPipeline
from processors.news_preprocessor import NewsPreprocessor
from vector_database.vector_database import VectorDatabaseManager

# Set environment variable to use 'spawn' instead of 'fork'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

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

        graph_database_manager = GraphDatabaseManager(
            uri=config.graph_db.uri,
            user=config.graph_db.user,
            password=config.graph_db.password
        )

        preprocessor = NewsPreprocessor(logger)

        # Initialize pipeline
        pipeline = NewsProcessingPipeline(
            db_manager=db_manager,
            vector_db_manager=vector_database_manager,
            graph_db_manager=graph_database_manager,
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
        if 'graph_database_manager' in locals():
            graph_database_manager.close()


def start_worker():
    """Entry point for worker process"""
    app.worker_main()


if __name__ == "__main__":
    start_worker()
