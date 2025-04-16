import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base

class ChatSession(Base):
    """Chat session table for tracking conversations"""
    __tablename__ = 'chat_sessions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_active = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    title = Column(String(255), nullable=True)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    references = relationship("ChatReference", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_chat_session_last_active', last_active),
    )


class ChatMessage(Base):
    """Individual chat messages within a session"""
    __tablename__ = 'chat_messages'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")
    references = relationship("ChatReference", back_populates="message", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_chat_message_session_timestamp', session_id, timestamp),
    )


class ChatReference(Base):
    """References to news articles used in chat responses"""
    __tablename__ = 'chat_references'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey('chat_messages.id'), nullable=False)
    news_id = Column(String(255), nullable=False)
    relevance_score = Column(Float, nullable=False)
    content_snippet = Column(Text, nullable=False)

    session = relationship("ChatSession", back_populates="references")
    message = relationship("ChatMessage", back_populates="references")

    __table_args__ = (
        Index('idx_chat_reference_session_message', session_id, message_id),
    )
