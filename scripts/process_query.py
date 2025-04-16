import asyncio
import logging
from typing import List, Dict, Any

from processors.news_preprocessor import QueryPreprocessor
from vector_database.vector_database import VectorDatabaseManager
from retriever.retriever import Retriever
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def process_and_search_query(
        query: str,
        top_k: int = 5,
) -> List[Dict[str, Any]]:
    try:
        # Initialize components
        vector_db = VectorDatabaseManager(
            host=config.vector_db.host,
            port=config.vector_db.port
        )

        query_preprocessor = QueryPreprocessor(logger)

        retriever = Retriever(
            vector_db_manager=vector_db,
            query_preprocessor=query_preprocessor,
            openai_api_key=config.openai.api_key
        )

        results = await retriever.search(
            query=query,
            top_k=top_k
        )

        return results

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return []


async def main():
    # Example queries
    queries = [
        {
            "text": "ترامپ چه گفت",
            "source": "irna"
        }
    ]

    for query_info in queries:
        logger.info(f"\nProcessing query: {query_info['text']}")

        results = await process_and_search_query(
            query=query_info["text"],
            top_k=query_info.get("top_k", 5),
        )

        # Display results
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult #{i}:")
            logger.info(f"Score: {result['score']:.3f}")
            logger.info(f"Content preview: {result['content'][:200]}...")
            logger.info(f"Source: {result['metadata'].get('source', 'Unknown')}")
            logger.info(f"Published Date: {result['metadata'].get('published_date', 'Unknown')}")
            if result['metadata'].get('keywords'):
                logger.info(f"Keywords: {', '.join(result['metadata']['keywords'])}")
            logger.info("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
