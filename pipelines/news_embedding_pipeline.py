import logging
import os
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# Database components
from db.db_manager import DBManager, RawNews  # Update with your actual module name
from config import config

# Text processing and chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI embeddings
from langchain_openai import OpenAIEmbeddings

# Import the Vector Database Manager
from vector_database.vector_database import VectorDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsEmbeddingPipeline:
    """
    Pipeline to process news articles and generate embeddings
    """

    def __init__(
            self,
            db_manager: DBManager,
            vector_db_manager: VectorDatabaseManager,
            openai_api_key: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            batch_size: int = 25,
            max_retries: int = 3,
            retry_delay: int = 2,
    ):
        """
        Initialize the news embedding pipeline

        Args:
            db_manager: Database manager for accessing raw news
            vector_db_manager: Vector database manager for storing embeddings
            openai_api_key: OpenAI API key for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            batch_size: Number of articles to process in a batch
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.db_manager = db_manager
        self.vector_db = vector_db_manager

        # Set API key from args or environment
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY env var")

        # Text processing config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.openai_api_key
        )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking and embedding

        Args:
            text: Raw text content

        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Basic cleanup
        text = text.strip()

        return text

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata

        Args:
            text: Text to split
            metadata: Metadata to associate with each chunk

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        chunks = []

        # Add source info to text if small enough
        if len(text) < self.chunk_size * 2:
            # For shorter texts, just create a single chunk
            chunks.append({
                "text": text,
                "metadata": metadata
            })
        else:
            # For longer texts, use the splitter
            split_texts = self.text_splitter.split_text(text)

            # Create chunks with metadata
            for i, chunk_text in enumerate(split_texts):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["chunk_total"] = len(split_texts)

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

        return chunks

    def get_embedding_with_retries(self, text: str) -> List[float]:
        """
        Get embedding with retry logic

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If all retries fail
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Failed to get embedding after {self.max_retries} retries: {str(e)}")
                    raise

                logger.warning(f"Embedding error (retry {retries}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)

    def store(self, text: str, metadata: Dict[str, Any]) -> int:
        """
        Generate embedding for a text chunk and store it in the vector database

        Args:
            text: Text chunk to embed and store
            metadata: Metadata to associate with the chunk

        Returns:
            int: Index of the stored vector or -1 if failed
        """
        try:
            # Generate embedding
            embedding = self.get_embedding_with_retries(text)

            # Create payload with text and metadata
            payload = {
                "text": text,
                **metadata
            }

            # Store in vector database
            return self.vector_db.store(embedding, payload)
        except Exception as e:
            logger.error(f"Error storing text embedding: {str(e)}")
            return -1

    def process_news_batch(self, news_items: List[RawNews]) -> int:
        """
        Process a batch of news items

        Args:
            news_items: List of news items to process

        Returns:
            Number of chunks processed and stored
        """
        processed_chunks = 0
        news_ids_to_mark = []

        for item in news_items:
            try:
                # Extract data from news item
                news_id = item.id
                content = item.content
                source = item.source
                pub_date = item.published_date

                # Basic metadata
                metadata = {
                    "news_id": news_id,
                    "source": source,
                    "published_date": pub_date.isoformat(),
                    "processed_date": datetime.now().isoformat()
                }

                # Preprocess text
                preprocessed_text = self.preprocess_text(content)

                # Skip if content is too short after preprocessing
                if len(preprocessed_text) < 50:
                    logger.warning(f"News item {news_id} content too short after preprocessing, skipping")
                    news_ids_to_mark.append(news_id)
                    continue

                # Chunk text
                chunks = self.chunk_text(preprocessed_text, metadata)

                # Process each chunk
                chunk_vectors = []
                chunk_payloads = []

                for chunk in chunks:
                    chunk_text = chunk["text"]
                    chunk_metadata = chunk["metadata"]

                    # Store the chunk
                    vector_id = self.store(chunk_text, chunk_metadata)
                    if vector_id >= 0:
                        processed_chunks += 1

                # Mark for processing
                news_ids_to_mark.append(news_id)

            except Exception as e:
                logger.error(f"Error processing news item {item.id}: {str(e)}")

        # Mark news as processed in database
        if news_ids_to_mark:
            try:
                self.db_manager.mark_news_as_processed(news_ids_to_mark)
                logger.info(f"Marked {len(news_ids_to_mark)} news items as processed")
            except Exception as e:
                logger.error(f"Error marking news as processed: {str(e)}")

        # Save vector database state
        self.vector_db.save()

        return processed_chunks

    def run(self, limit: int = 100) -> Dict[str, Any]:
        """
        Run the pipeline on unprocessed news

        Args:
            limit: Maximum number of news items to process

        Returns:
            Statistics about the processing
        """
        start_time = time.time()
        total_processed = 0
        total_chunks = 0

        try:
            # Get unprocessed news
            logger.info(f"Fetching up to {limit} unprocessed news items")
            news_items = self.db_manager.get_unprocessed_news(limit=limit)
            logger.info(f"Found {len(news_items)} unprocessed news items")

            for i in range(0, len(news_items), self.batch_size):
                batch = news_items[i:i + self.batch_size]
                logger.info(
                    f"Processing batch {i // self.batch_size + 1}/{(len(news_items) + self.batch_size - 1) // self.batch_size}")

                # Process batch
                chunks_processed = self.process_news_batch(batch)
                total_processed += len(batch)
                total_chunks += chunks_processed

        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            raise

        # Get vector database stats
        vector_db_stats = self.vector_db.get_stats()

        # Calculate pipeline stats
        duration = time.time() - start_time
        stats = {
            "total_news_processed": total_processed,
            "total_chunks_created": total_chunks,
            "duration_seconds": duration,
            "items_per_second": total_processed / duration if duration > 0 else 0,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(time.time()).isoformat(),
            "vector_db_stats": vector_db_stats
        }

        logger.info(
            f"Pipeline completed. Processed {total_processed} news items into {total_chunks} chunks in {duration:.2f} seconds")

        return stats


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

    # Set up Vector Database Manager
    vector_db_manager = VectorDatabaseManager(
        index_path="./vector_db/news.index",
        metadata_path="./vector_db/news_metadata.pkl",
        embedding_dim=1536,  # OpenAI embedding dimension
        index_type="flat",
        save_every=100
    )

    # Initialize pipeline with the vector database manager
    pipeline = NewsEmbeddingPipeline(
        db_manager=db_manager,
        vector_db_manager=vector_db_manager,
        openai_api_key=config.openai.api_key,
        chunk_size=1000,
        chunk_overlap=200,
        batch_size=25
    )

    # Run the pipeline (process 100 news items)
    stats = pipeline.run(limit=100)
    print(f"Pipeline stats: {stats}")


# Example usage
if __name__ == "__main__":
    main()