import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Iterator

from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Boolean, func, Index
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import expression

from collectors.schema import News

Base = declarative_base()

class RawNews(Base):
    """Raw news articles table schema for PostgreSQL database"""
    __tablename__ = 'raw_news'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    published_date = Column(DateTime(timezone=True), nullable=False)
    has_processed = Column(Boolean, server_default=expression.false(), nullable=False)

    # PostgreSQL-specific indexes
    __table_args__ = (
        Index('idx_news_published_date', published_date),
        Index('idx_news_published_date_processed', published_date, has_processed,
              postgresql_where=(has_processed == False)),
    )

class DBManager:
    def __init__(
            self,
            user: str,
            password: str,
            host: str,
            port: int,
            database: str,
            connector: str = "psycopg2",
            pool_size: int = 10,
            max_overflow: int = 20,
            logger: logging.Logger = logging.getLogger(__name__)
    ):
        """
        Initialize PostgreSQL database connection manager

        Args:
            user: Database username
            password: Database password
            host: Database host
            port: Database port
            database: Database name
            connector: PostgreSQL connector type (default: psycopg2)
            pool_size: Connection pool size
            max_overflow: Maximum number of connections to overflow
            logger: Logger instance
        """
        if connector not in ["psycopg2", "psycopg", "asyncpg", "pg8000"]:
            raise ValueError(f"Unsupported PostgreSQL connector: {connector}")

        self.connection_string = (
            f"postgresql+{connector}://{user}:{password}@{host}:{port}/{database}"
        )

        self.engine = create_engine(
            self.connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            poolclass=QueuePool
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        self.logger = logger

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """
        Provide a database session with automatic cleanup

        Yields:
            SQLAlchemy Session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Session rollback due to error: {str(e)}")
            raise
        finally:
            session.close()

    def close_connections(self) -> None:
        """Close all database connections in the pool"""
        try:
            self.engine.dispose()
            self.logger.info("Successfully closed all database connections")
        except SQLAlchemyError as e:
            self.logger.error(f"Error closing database connections: {str(e)}")
            raise

    def get_unprocessed_news(self, limit: int = 100) -> List[RawNews]:
        """
        Retrieve unprocessed news articles with limit

        Args:
            limit: Maximum number of articles to retrieve

        Returns:
            List of unprocessed RawNews objects (detached from session)
        """
        with self.get_session() as session:
            news_items = session.query(RawNews) \
                .filter_by(has_processed=False) \
                .order_by(RawNews.published_date.desc()) \
                .limit(limit) \
                .with_for_update() \
                .all()

            # Detach objects to avoid session-bound errors
            for item in news_items:
                session.expunge(item)

            return news_items

    def mark_news_as_processed(self, news_ids: List[int]) -> None:
        """
        Mark multiple news articles as processed

        Args:
            news_ids: List of news article IDs to mark as processed

        Raises:
            SQLAlchemyError: If database update fails
        """
        if not news_ids:
            return

        with self.get_session() as session:
            try:
                session.query(RawNews) \
                    .filter(RawNews.id.in_(news_ids)) \
                    .with_for_update() \
                    .update(
                    {
                        RawNews.has_processed: True
                    },
                    synchronize_session=False
                )
            except SQLAlchemyError as e:
                session.rollback()
                raise RuntimeError(f"Failed to mark news as processed: {str(e)}") from e

    def initialize_database(self) -> None:
        """Initialize PostgreSQL database by creating required tables and indexes"""
        try:
            self.logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def get_unprocessed_count(self) -> int:
        """Get count of unprocessed news articles"""
        with self.get_session() as session:
            return session.query(func.count(RawNews.id)) \
                .filter_by(has_processed=False) \
                .scalar()

    def store_raw_news(self, news_items: List[News]) -> List[int]:
        """
        Store a batch of raw news items in the database

        Args:
            news_items: List of News objects containing:
                - source: str
                - content: str
                - published_date: datetime or str (ISO format)
                - category: str (optional)

        Returns:
            List of inserted news IDs

        Raises:
            ValueError: If required fields are missing
            SQLAlchemyError: If database operation fails
        """
        with self.get_session() as session:
            try:
                news_objects = []
                for item in news_items:
                    if not all(k in item for k in ['source', 'content', 'published_date']):
                        raise ValueError(f"Missing required fields in news item: {item}")

                    # Handle datetime conversion
                    if isinstance(item['published_date'], str):
                        try:
                            published_date = datetime.fromisoformat(
                                item['published_date'].replace('Z', '+00:00')
                            )
                        except ValueError as e:
                            raise ValueError(f"Invalid published_date format: {item['published_date']}") from e
                    else:
                        published_date = item['published_date']

                    news = RawNews(
                        source=item['source'],
                        content=item['content'],
                        published_date=published_date,
                        has_processed=False
                    )
                    news_objects.append(news)

                self.logger.info(f"Inserting {len(news_objects)} news items")
                session.add_all(news_objects)
                session.commit()

                return [news.id for news in news_objects]

            except (ValueError, SQLAlchemyError) as e:
                session.rollback()
                self.logger.error(f"Failed to store {len(news_items)} news items: {str(e)}")
                raise

    def migrate_database(self) -> None:
        """
        Perform database migrations (placeholder for Alembic or manual migrations)

        Note: Currently initializes tables; replace with proper migration logic.
        """
        try:
            self.logger.info("Running database migrations...")
            self.initialize_database()  # Temporary; replace with Alembic
            self.logger.info("Database migrations completed successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Database migration failed: {str(e)}")
            raise