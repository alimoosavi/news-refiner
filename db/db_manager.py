import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
import uuid

from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Boolean, func, Index
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import expression

from collectors.schema import News
from db.base import Base
from db.chat_models import ChatSession, ChatMessage, ChatReference

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

    # Chat Session Operations
    def create_chat_session(self) -> Dict[str, Any]:
        """Create a new chat session"""
        with self.get_session() as session:
            chat_session = ChatSession()
            session.add(chat_session)
            session.commit()
            
            # Return dictionary instead of session object
            return {
                "id": chat_session.id,
                "created_at": chat_session.created_at
            }

    def get_chat_session(self, session_id: uuid.UUID) -> Optional[ChatSession]:
        """Get chat session by ID"""
        with self.get_session() as session:
            return session.query(ChatSession).get(session_id)

    def update_chat_session_activity(self, session_id: uuid.UUID) -> None:
        """Update last active timestamp of a chat session"""
        with self.get_session() as session:
            chat_session = session.query(ChatSession).get(session_id)
            if chat_session:
                chat_session.last_active = datetime.utcnow()
                session.commit()

    # Chat Message Operations
    def create_chat_message(self, session_id: uuid.UUID, role: str, content: str) -> Dict[str, Any]:
        """Create a new chat message"""
        with self.get_session() as session:
            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            
            # Return dictionary instead of session object
            return {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp
            }

    def get_chat_history(self, session_id: uuid.UUID, limit: int) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        with self.get_session() as session:
            messages = session.query(ChatMessage)\
                .filter(ChatMessage.session_id == session_id)\
                .order_by(ChatMessage.timestamp.desc())\
                .limit(limit)\
                .all()
            
            # Convert to dictionaries before returning
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in messages
            ]

    # Chat Reference Operations
    def create_chat_references(self, references: List[Dict]) -> List[ChatReference]:
        """Create multiple chat references"""
        with self.get_session() as session:
            chat_refs = []
            for ref in references:
                chat_ref = ChatReference(
                    session_id=ref['session_id'],
                    message_id=ref['message_id'],
                    news_id=ref['news_id'],
                    relevance_score=ref['relevance_score'],
                    content_snippet=ref['content_snippet']
                )
                session.add(chat_ref)
                chat_refs.append(chat_ref)
            session.commit()
            return chat_refs

    def get_message_references(self, message_id: uuid.UUID) -> List[ChatReference]:
        """Get references for a specific message"""
        with self.get_session() as session:
            return session.query(ChatReference) \
                .filter(ChatReference.message_id == message_id) \
                .order_by(ChatReference.relevance_score.desc()) \
                .all()

    def get_session_references(self, session_id: uuid.UUID) -> List[ChatReference]:
        """Get all references for a chat session"""
        with self.get_session() as session:
            return session.query(ChatReference) \
                .filter(ChatReference.session_id == session_id) \
                .order_by(ChatReference.relevance_score.desc()) \
                .all()

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete chat sessions older than specified days"""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted = session.query(ChatSession) \
                .filter(ChatSession.last_active < cutoff_date) \
                .delete(synchronize_session=False)
            session.commit()
            return deleted

    def get_session_info(self, session_id: uuid.UUID) -> Dict[str, Any]:
        """Get comprehensive session information"""
        with self.get_session() as session:
            chat_session = session.query(ChatSession).get(session_id)
            if not chat_session:
                return None

            messages = session.query(ChatMessage) \
                .filter(ChatMessage.session_id == session_id) \
                .order_by(ChatMessage.timestamp) \
                .all()

            return {
                "session_id": str(chat_session.id),
                "created_at": chat_session.created_at.isoformat(),
                "last_active": chat_session.last_active.isoformat(),
                "messages": [
                    {
                        "id": str(msg.id),
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "references": [
                            {
                                "news_id": ref.news_id,
                                "relevance_score": ref.relevance_score,
                                "content_snippet": ref.content_snippet
                            }
                            for ref in msg.references
                        ]
                    }
                    for msg in messages
                ]
            }
