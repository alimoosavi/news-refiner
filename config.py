from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables first
load_dotenv('.env')

class DatabaseConfig(BaseSettings):
    """Database configuration with environment variable prefix"""
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
        extra='ignore'
    )

    hostname: str = Field(..., min_length=3)
    name: str = "news_db"
    user: str = "root"
    password: str = Field(..., min_length=8)
    port: int = Field(3306, ge=1024, le=65535)
    connector: str = "mysqlconnector"
    pool_size: int = 10
    max_overflow: int = 20

class NewsCrawlerConfig(BaseSettings):
    """News crawler configuration"""
    model_config = SettingsConfigDict(env_prefix="NEWS_CRAWLER_")

    base_url: str = Field(..., min_length=3)

class OpenAIConfig(BaseSettings):
    """OpenAI configuration with secret handling"""
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        secrets_dir='/run/secrets'  # For Docker secrets
    )

    api_key: str = Field(..., repr=False)
    model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536  # Fixed for ada-002
    max_retries: int = 3
    timeout: int = 30

class CeleryConfig(BaseSettings):
    """Celery configuration with Redis defaults"""
    model_config = SettingsConfigDict(env_prefix="CELERY_")

    broker_url: str = "redis://localhost:6379/0"
    result_backend: Optional[str] = None
    concurrency: int = 4
    task_time_limit: int = 300

class FAISSConfig(BaseSettings):
    """Vector store configuration"""
    model_config = SettingsConfigDict(env_prefix="FAISS_")

    index_path: str = Field("news_index.faiss", min_length=3)
    payload_path: Optional[str] = None
    index_dim: int = 1536  # Should match OpenAI embedding_dim
    save_interval: int = Field(1000, ge=100, le=10000)


class ProcessingConfig(BaseSettings):
    """News processing pipeline configuration"""
    model_config = SettingsConfigDict(env_prefix="PROCESSING_")

    batch_size: int = Field(50, ge=1, le=100)
    interval_minutes: int = Field(5, ge=1, le=60)
    max_tokens: int = Field(512, ge=256, le=2048)
    overlap_percent: float = Field(0.1, ge=0.0, le=1.0)
    embedding_batch_size: int = Field(50, ge=1, le=100)

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field("INFO", min_length=3)
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", min_length=5)
    file: Optional[str] = Field(None, min_length=3)

class ConfigManager(BaseSettings):
    """Main configuration aggregator"""
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    crawler: NewsCrawlerConfig = Field(default_factory=NewsCrawlerConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls):
        """Load configuration with environment variables"""
        return cls.model_validate({})

# Singleton instance
config = ConfigManager.load()