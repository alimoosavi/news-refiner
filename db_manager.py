from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import List, Optional
from datetime import datetime

Base = declarative_base()

class RawNews(Base):
    """Raw news articles table schema"""
    __tablename__ = 'raw_news'

    id = Column(Integer, primary_key=True)
    source = Column(String(255), nullable=False)
    title = Column(Text, nullable=False)
    body = Column(Text, nullable=False)
    url = Column(String(512), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    has_processed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


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
        max_overflow: int = 20
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
                # Update with pessimistic locking
                session.query(RawNews)\
                    .filter(RawNews.id.in_(news_ids))\
                    .with_for_update()\
                    .update(
                        {
                            RawNews.has_processed: True,
                            RawNews.updated_at: datetime.utcnow()
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
            return session.query(func.count(RawNews.id))\
                .filter_by(has_processed=False)\
                .scalar()

    def get_unprocessed_news(self, limit: int = 100) -> List[RawNews]:
        """
        Retrieve unprocessed news articles with limit
        
        Args:
            limit: Maximum number of articles to retrieve

        Returns:
            List of unprocessed RawNews objects
        """
        with self.get_session() as session:
            return session.query(RawNews)\
                .filter_by(has_processed=False)\
                .order_by(RawNews.timestamp.desc())\
                .limit(limit)\
                .with_for_update()\
                .all()
