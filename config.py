from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class MessengerConfig(BaseSettings):
    """Configuration settings for the Telegram bot."""

    model_config = SettingsConfigDict(env_prefix="MESSENGER_")

    api_key: str
    api_id: str
    api_hash: str
    app_title: str


class CollectorConfig(BaseSettings):
    """Configuration settings for the collector."""
    model_config = SettingsConfigDict(env_prefix="COLLECTOR_")

    max_retries: int = Field(3)


class CacheConfig(BaseSettings):
    """Configuration settings for the cache."""
    # file_path: str = Field(..., env="CACHE_FILE_PATH")

    model_config = SettingsConfigDict(env_prefix="CACHE_")
    file_path: str


class DatabaseConfig(BaseSettings):
    """Database configuration with environment variable prefix"""
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
        extra="allow"
    )

    hostname: str = Field(..., min_length=3)
    name: str = "news_db"
    user: str = "postgres"  # Changed default user
    passkey: str = Field(..., min_length=1)  # Required password for PostgreSQL
    port: int = Field(5432, ge=1024, le=65535)  # Changed default port
    connector: str = "psycopg2"  # Changed default connector
    pool_size: int = Field(10, ge=1, le=50)
    max_overflow: int = Field(20, ge=0, le=100)


class NewsCrawlerConfig(BaseSettings):
    """News crawler configuration"""
    model_config = SettingsConfigDict(env_prefix="NEWS_CRAWLER_")

    base_url: str = Field(..., min_length=3)


class OpenAIConfig(BaseSettings):
    """OpenAI configuration with secret handling"""
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        secrets_dir='/run/secrets'
    )

    api_key: str = Field(..., repr=False, min_length=20)
    model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    max_retries: int = Field(3, ge=1, le=10)
    timeout: int = Field(30, ge=5, le=120)


class CeleryConfig(BaseSettings):
    """Celery configuration with Redis defaults"""
    model_config = SettingsConfigDict(env_prefix="CELERY_")

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    concurrency: int = Field(4, ge=1, le=16)
    task_time_limit: int = Field(300, ge=60, le=3600)


class FAISSConfig(BaseSettings):
    """Vector store configuration"""
    model_config = SettingsConfigDict(env_prefix="FAISS_")

    index_path: str = Field("news_embeddings.faiss", min_length=3)  # Match .env
    payload_path: Optional[str] = None
    index_dim: int = 1536
    save_interval: int = Field(1000, ge=100, le=10000)


class ProcessingConfig(BaseSettings):
    """News processing pipeline configuration"""
    model_config = SettingsConfigDict(env_prefix="PROCESSING_")

    batch_size: int = Field(100, ge=1, le=1000)  # Match .env
    interval: int = Field(1, ge=1, le=60)  # Changed to seconds to match .env
    max_tokens: int = Field(512, ge=256, le=2048)
    overlap_percent: float = Field(0.1, ge=0.0, le=1.0)
    embedding_batch_size: int = Field(50, ge=1, le=100)


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", min_length=5)
    file: Optional[str] = Field("/var/log/news_processor.log", min_length=3)


class VectorDatabaseConfig(BaseSettings):
    """Vector database configuration"""
    model_config = SettingsConfigDict(env_prefix="VECTOR_DB_")

    host: str = Field("localhost", min_length=3)
    port: int = Field(6333, ge=1024, le=65535)
    grpc_port: int = Field(6334, ge=1024, le=65535)
    collection: str = Field("news_vectors", min_length=1)
    index_path: str = Field("./faiss_data/news.index", min_length=3)
    metadata_path: str = Field("./faiss_data/news_metadata.pkl", min_length=3)

# In the ConfigManager class, add:
class ConfigManager(BaseSettings):
    """Main configuration aggregator"""
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False
    )

    messenger: MessengerConfig = Field(default_factory=MessengerConfig)
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    crawler: NewsCrawlerConfig = Field(default_factory=NewsCrawlerConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    vector_db: VectorDatabaseConfig = Field(default_factory=VectorDatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls):
        """Load configuration with environment variables"""
        return cls()


# Singleton instance
config = ConfigManager.load()
