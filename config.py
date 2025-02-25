import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Configuration settings for the PostgreSQL database."""
    hostname: str = Field(..., env="DB_HOSTNAME")
    name: str = Field(..., env="DB_NAME")
    user: str = Field(..., env="DB_USER")
    password: str = Field(..., env="DB_PASSWORD")
    port: int = Field(..., env="DB_PORT")

    @classmethod
    def load_from_env(cls):
        return cls(
            hostname=os.getenv("DB_HOSTNAME", "localhost"),
            name=os.getenv("DB_NAME", "news_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            port=int(os.getenv("DB_PORT", 5432)),
        )

    def get_database_url(self) -> str:
        """Constructs and returns the database connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.hostname}:{self.port}/{self.name}"


class NewsCrawlerConfig(BaseModel):
    base_url: str = Field(..., env="NEWS_CRAWLER_BASE_URL")

    @classmethod
    def load_from_env(cls):
        return cls(
            base_url=os.getenv("NEWS_CRAWLER_BASE_URL", "")
        )


class Config:
    """Main configuration class that holds all sub-configs."""

    def __init__(self):
        self.database = DatabaseConfig.load_from_env()
        self.news_crawler = NewsCrawlerConfig.load_from_env()


# Create a single config object
config = Config()
