import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from retriever.reranker import RankedResult
from services.chat_service import ChatService
from processors.news_preprocessor import QueryPreprocessor
from vector_database.vector_database import VectorDatabaseManager
from retriever.retriever import Retriever
from db.db_manager import DBManager
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="News API",
    description="API for searching and chatting with news database",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared components
db_manager = DBManager(
    user=config.database.user,
    password=config.database.passkey,
    host=config.database.hostname,
    port=config.database.port,
    database=config.database.name
)

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


# Request/Response Models for Search
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    source: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    source: str
    published_date: str
    keywords: List[str]
    relevance_explanation: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    query_time: float


# Request/Response Models for Chat
class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")


class ChatResponse(BaseModel):
    response: str
    references: List[Dict[str, Any]]


class SessionInfoResponse(BaseModel):
    session_id: str
    created_at: str
    last_active: str
    messages: List[Dict[str, Any]]


# Search Routes
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search endpoint that performs hybrid search on news articles"""
    try:
        start_time = datetime.now()

        if request.source:
            results = await retriever.search_by_source(
                query=request.query,
                source=request.source,
                top_k=request.top_k
            )
        elif request.start_date and request.end_date:
            results = await retriever.search_by_date_range(
                query=request.query,
                start_date=request.start_date,
                end_date=request.end_date,
                top_k=request.top_k
            )
        else:
            results = await retriever.search(
                query=request.query,
                top_k=request.top_k
            )

        formatted_results = [
            SearchResult(
                id=result.content[:8] if isinstance(result, RankedResult) else str(result.get('id', '')),
                score=result.score if isinstance(result, RankedResult) else result.get('score', 0.0),
                content=result.content if isinstance(result, RankedResult) else result.get('content', ''),
                source=result.metadata.get('source', 'Unknown') if isinstance(result, RankedResult) else result.get(
                    'metadata', {}).get('source', 'Unknown'),
                published_date=result.metadata.get('published_date', '') if isinstance(result,
                                                                                       RankedResult) else result.get(
                    'metadata', {}).get('published_date', ''),
                keywords=result.metadata.get('keywords', []) if isinstance(result, RankedResult) else result.get(
                    'metadata', {}).get('keywords', []),
                relevance_explanation=result.relevance_explanation if isinstance(result, RankedResult) else None
            )
            for result in results
        ]

        return SearchResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time=(datetime.now() - start_time).total_seconds()
        )

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Chat Routes
@app.post("/chat/sessions", response_model=CreateSessionResponse)
async def create_chat_session():
    """Create a new chat session"""
    try:
        chat_service = ChatService(
            db_manager=db_manager,
            retriever=retriever
        )
        session_data = await chat_service.create_session()

        return CreateSessionResponse(
            session_id=str(session_data["id"]),
            created_at=session_data["created_at"].isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to create chat session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create chat session")


@app.post("/chat/sessions/{session_id}/completion", response_model=ChatResponse)
async def chat_completion(session_id: UUID, request: ChatRequest):
    """Send a message in an existing chat session"""
    try:
        chat_service = ChatService(
            db_manager=db_manager,
            retriever=retriever
        )

        response = await chat_service.process_message(
            session_id=session_id,
            query=request.message
        )

        return ChatResponse(
            response=response["response"],
            references=response["references"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Chat session not found")
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Chat completion failed")


@app.get("/chat/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session_history(session_id: UUID):
    """Get chat history for a session"""
    try:
        chat_service = ChatService(
            db_manager=db_manager,
            retriever=retriever
        )

        session_info = await chat_service.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return session_info
    except Exception as e:
        logger.error(f"Failed to get session history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session history")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
