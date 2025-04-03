import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List

from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Boolean, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from collectors.schema import News

Base = declarative_base()


class RawNews(Base):
    """Raw news articles table schema matching the existing database"""
    __tablename__ = 'raw_news'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Text, nullable=False)
    content = Column(Text, nullable=False)  # Changed from body to content
    published_date = Column(DateTime, nullable=False)  # Changed from timestamp
    has_processed = Column(Boolean, default=False, nullable=True)  # tinyint(1) default 0

    # Define indexes matching the SQL
    __table_args__ = (
        Index('idx_news_published_date', published_date),
        Index('idx_news_published_date_processed', published_date, has_processed),
        Index('idx_raw_news_published_date', published_date),
        Index('idx_raw_news_published_date_processed', published_date, has_processed),
    )


class DBManager:
    def __init__(
            self,
            user: str,
            password: str,
            host: str,
            port: int,
            database: str,
            connector: str = "mysqlconnector",
            pool_size: int = 10,
            max_overflow: int = 20,
            logger: logging.Logger = logging.getLogger(__name__)
    ):
        """
        Initialize database connection manager

        Args:
            user: Database username
            password: Database password
            host: Database host
            port: Database port
            database: Database name
            connector: MySQL connector type
            pool_size: Connection pool size
            max_overflow: Maximum number of connections to overflow
        """
        self.connection_string = (
            f"mysql+{connector}://{user}:{password}@{host}:{port}/{database}"
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
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup
        
        Yields:
            SQLAlchemy Session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_unprocessed_news(self, limit: int = 100) -> List[RawNews]:
        """
        Retrieve unprocessed news articles with limit
        
        Args:
            limit: Maximum number of articles to retrieve

        Returns:
            List of unprocessed RawNews objects
        """
        with self.get_session() as session:
            return session.query(RawNews) \
                .filter_by(has_processed=False) \
                .order_by(RawNews.published_date.desc()) \
                .limit(limit) \
                .with_for_update() \
                .all()

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
                session.commit()
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Failed to mark news as processed: {str(e)}") from e

    def create_tables(self) -> None:
        """Create all database tables if they don't exist"""
        Base.metadata.create_all(self.engine)

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
            news_items: List of dictionaries containing news data
                Each dict should have:
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
                # Prepare news items for bulk insert
                news_objects = []

                for item in news_items:
                    # Validate required fields
                    if not all(k in item for k in ['source', 'content', 'published_date']):
                        raise ValueError(f"Missing required fields in news item: {item}")

                    # Handle datetime conversion if string is provided
                    if isinstance(item['published_date'], str):
                        published_date = datetime.fromisoformat(item['published_date'].replace('Z', '+00:00'))
                    else:
                        published_date = item['published_date']

                    # Create RawNews object
                    news = RawNews(
                        source=item['source'],
                        content=item['content'],
                        published_date=published_date,
                        has_processed=False
                    )

                    news_objects.append(news)

                # Bulk insert
                session.bulk_save_objects(news_objects)
                session.flush()

                # Collect inserted IDs
                inserted_ids = [news.id for news in news_objects]

                session.commit()
                return inserted_ids

            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to store news batch: {str(e)}")
                raise
