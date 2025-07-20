"""
This module defines the FastAPI routes for querying the AML
(Anti-Money Laundering) document database using
Retrieval-Augmented Generation (RAG).

Main Components:
- QueryRequest: Pydantic model for incoming query requests with
optional filtering
- Query Source: Pydantic model for source document metadata.
- QueryResponse: Pydantic model for structured API responses with
answers and sources
- query_aml_documents: Main endpoint that processes AML compliance queries
- health_check: Simple health monitoring endpoint
- get_api_info: API capabilities and feature documentation endpoint

Usage:
    POST /api/v1/query - Submit AML compliance questions
    GET /api/v1/health - Check API health status
    GET /api/v1/info - Get API information and capabilities
"""
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from backend.agents.rag_agent import AMLRagAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])

try:
    rag_agent = AMLRagAgent()
    logger.info("RAG Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Agent: {str(e)}")
    rag_agent = None


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description=(
            "The AML/CFT compliance question to query the document database."
        )
    )


class Source(BaseModel):
    """
    Information about the source document used in the answer.
    """
    filename: str
    jurisdiction: str
    language: str
    chunk_text: str
    confidence: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    detected_language: str
    relevant_jurisdictions: list[str] = []
    confidence: float = 0.0


@router.post("/query", response_model=QueryResponse)
async def query_aml_documents(request: QueryRequest):
    """
    Query the AML document database using RAG.
    """
    logger.info(f"Received query request: {request.question[:100]}...")

    if rag_agent is None:
        logger.error("RAG Agent is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Agent is not available. Please try again later."
        )

    try:
        logger.info("Processing query with RAG Agent...")
        result = await rag_agent.query(request.question)

        sources = [
            Source(
                filename=doc['filename'],
                jurisdiction=doc['region'],
                language=doc['language'],
                chunk_text=(
                    doc['content'][:500] + "..."
                    if len(doc['content']) > 500
                    else doc['content']
                ),
                confidence=doc.get('score', 0.0)
            )
            for doc in result['sources']
        ]

        response = QueryResponse(
            answer=result['answer'],
            sources=sources,
            detected_language=result['detected_language'],
            relevant_jurisdictions=result.get('relevant_jurisdictions', []),
            confidence=result.get('confidence', 0.0)
        )

        logger.info(
            f"Query processed successfully - "
            f"Language: {response.detected_language}, "
            f"Sources: {response.sources}"
            f"Jurisdictions: {response.relevant_jurisdictions}"
        )

        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)

        error_message = (
            "I apologize, but I encountered an error while processing your "
            "question. Please try rephrasing your question."
        )

        return QueryResponse(
            answer=error_message,
            sources=[],
            detected_language="English",
            relevant_jurisdictions=[],
            confidence=0.0
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify if API is running.
    """
    logger.debug("Health check requested")
    agent_status = "available" if rag_agent else "unavailable"

    agent_details = {}
    if rag_agent:
        try:
            agent_details = {
                "openai_client": (
                    "connected" if rag_agent.openai_client else "disconnected"
                ),
                "qdrant_client": (
                    "connected" if rag_agent.qdrant_client else "disconnected"
                ),
                "collection_name": rag_agent.collection_name
            }
        except Exception as e:
            logger.error(f"Error retrieving agent details: {str(e)}")
            agent_details = {"error": "Failed to retrieve agent details"}

    health_data = {
        "status": "healthy" if agent_status == "available" else "unhealthy",
        "service": "AML MultiAgent RAG API",
        "version": "1.0.0",
        "rag_agent": {
            "status": agent_status,
            "details": agent_details
        }
    }
    logger.info(
        f"Health check completed - Status: {health_data['status']}, "
    )
    return health_data


@router.get("/info")
async def get_api_info():
    """
    Get information about the API capabilities and supported features.

    Returns:
        dict: Information about the API
    """
    return {
        "name": "AML MultiAgent RAG API",
        "version": "1.0.0",
        "description": "AI-powered AML/CFT regulatory compliance assistant",
        "supported_languages": ["English", "Portuguese"],
        "supported_jurisdictions": ["USA", "EU", "Brazil"],
        "features": [
            "Natural language queries",
            "Automatic language detection",
            "Multi-jurisdictional search",
            "Source citation",
            "Confidence scoring"
        ],
        "endpoints": [
            "POST /api/v1/query - Submit compliance questions",
            "GET /api/v1/health - Check service health",
            "GET /api/v1/info - Get API information"
        ]
    }
