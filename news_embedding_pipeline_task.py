import logging

from config import config
from db.db_manager import DBManager
from pipelines.news_embedding_pipeline import NewsEmbeddingPipeline
from vector_database.vector_database import VectorDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Set up DB manager
    db_manager = DBManager(
        user=config.database.user,
        password=config.database.password,
        host=config.database.hostname,
        port=config.database.port,
        database=config.database.name,
        connector=config.database.connector,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow
    )

    vector_database_manager = VectorDatabaseManager(
        index_path="./faiss_data/news.index",
        metadata_path="./faiss_data/news_metadata.pkl"
    )

    # Initialize pipeline
    pipeline = NewsEmbeddingPipeline(
        db_manager=db_manager,
        openai_api_key=config.openai.api_key,
        chunk_size=1000,
        chunk_overlap=200,
        batch_size=25,
        vector_db_manager=vector_database_manager
    )

    # Run the pipeline (process 100 news items)
    stats = pipeline.run(limit=100)
    print(f"Pipeline stats: {stats}")


# Example usage
if __name__ == "__main__":
    main()
