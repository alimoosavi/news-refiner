import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from processors.news_preprocessor import QueryPreprocessor
from vector_database.vector_database import VectorDatabaseManager
from retriever.retriever import Retriever
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="News Search API",
    description="API for searching news articles using hybrid search and reranking",
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

# Initialize components
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

# Request/Response models
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

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search endpoint that performs hybrid search on news articles
    """
    try:
        start_time = datetime.now()

        # Perform search based on filters
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

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                SearchResult(
                    id=result.get("id", ""),
                    score=result.get("score", 0.0),
                    content=result.get("content", ""),
                    source=result.get("metadata", {}).get("source", "Unknown"),
                    published_date=result.get("metadata", {}).get("published_date", ""),
                    keywords=result.get("metadata", {}).get("keywords", []),
                    relevance_explanation=getattr(result, "relevance_explanation", None)
                )
            )

        query_time = (datetime.now() - start_time).total_seconds()

        return SearchResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time=query_time
        )

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)