"""
Enhanced API routes with Multi-Agent support for AML RAG System.

This module provides both single-agent (backward compatible) and multi-agent
endpoints for processing AML compliance queries with enhanced quality control.

New Multi-Agent Endpoints:
- POST /api/v1/multi-agent/query - Enhanced query processing with all agents
- GET /api/v1/multi-agent/status - Multi-agent system health status

Features:
- Consistency validation via ConsistencyAgent
- Confidence calibration via ConfidenceAgent
- Quality gates and comprehensive recommendations
- Detailed analysis and processing metrics
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.agents.orchestrator import MultiAgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for enhanced multi-agent endpoints
multi_agent_router = APIRouter(
    prefix="/multi-agent",
    tags=["Multi-Agent RAG System"]
)

try:
    orchestrator = MultiAgentOrchestrator()
    logger.info("Multi-Agent Orchestrator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Multi-Agent Orchestrator: {str(e)}")
    orchestrator = None


class MultiAgentQueryRequest(BaseModel):
    """Request model for multi-agent query processing."""

    question: str = Field(
        ...,
        description="The AML compliance question to process",
        min_length=3,
        max_length=1000,
        example=(
            "What are the customer identification requirements for "
            "high-risk customers in Brazil?"
        )
    )


class QualityGates(BaseModel):
    """Model for quality gate results."""

    consistency_passed: bool = Field(
        description="Whether consistency validation passed"
    )
    confidence_passed: bool = Field(
        description="Whether confidence threshold was met"
    )
    overall_passed: bool = Field(
        description="Whether overall quality gates passed"
    )


class MultiAgentQueryResponse(BaseModel):
    """Enhanced response model with multi-agent analysis."""

    success: bool = Field(
        description="Whether the query was processed successfully"
    )
    query: str = Field(description="The original user question")
    answer: str = Field(description="The comprehensive answer")
    sources: list = Field(description="Source documents with metadata")

    quality_score: float = Field(
        description="Overall quality score (0.0-1.0)"
    )
    confidence_score: float = Field(
        description="Calibrated confidence score"
    )
    confidence_level: str = Field(
        description="Confidence level category"
    )
    consistency_score: float = Field(
        description="Consistency validation score"
    )
    is_consistent: bool = Field(
        description="Whether response is consistent"
    )
    quality_gates: QualityGates = Field(
        description="Quality gate results"
    )
    quality_assessment: str = Field(
        description="Human-readable quality assessment"
    )

    recommendations: list = Field(description="System recommendations")

    metadata: dict = Field(description="Processing metadata")

    detailed_analysis: Optional[dict] = Field(
        default=None,
        description="Detailed analysis results"
    )


@multi_agent_router.post(
    "/query",
    response_model=MultiAgentQueryResponse,
    summary="Process AML compliance queries with multi-agent validation",
    description="""
    Process AML compliance queries using a multi-agent system:

    Three-Agent Architecture:
    - RAG Agent: Retrieves relevant documents and generates initial answers
    - Consistency Agent: Validates response coherence and citation accuracy
    - Confidence Agent: Evaluates reliability and calibrates confidence scores

    Quality Assurance:
    - Automatic quality gates (Consistency ≥60%, Confidence ≥40%)
    - Source attribution and citation validation
    - Multi-language support (English and Portuguese) with automatic detection
    - Real-time quality assessment with actionable recommendations

    Enterprise Features:
    - Processing time metrics and performance monitoring
    - Detailed analysis breakdown for more transparency
    - Jurisdiction-specific compliance checking
    - Professional regulatory language validation
    """
)
async def query_aml_documents(request: MultiAgentQueryRequest):
    """
    Process AML compliance query through multi-agent pipeline.

    This endpoint provides enhanced query processing with quality validation,
    confidence calibration, and comprehensive recommendations.
    """
    if orchestrator is None:
        logger.error("Multi-Agent Orchestrator not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-agent system not available. Please try again later."
        )

    try:
        logger.info(
            f"Processing multi-agent query: '{request.question[:100]}...'"
        )

        result = await orchestrator.process_query(
            query=request.question
        )

        if not result.get("success", False):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Multi-agent processing failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {error_msg}"
            )

        quality_score = result.get('quality_score', 0)
        logger.info(
            f"Multi-agent query completed - Quality: {quality_score:.3f}"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in multi-agent query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during query processing: {str(e)}"
        )


@multi_agent_router.get(
    "/status",
    summary="System health and capabilities",
    description=(
        "Get the health status, capabilities, and configuration of the AML "
        "multi-agent system"
    )
)
async def get_system_status():
    """Get multi-agent system health status and capabilities."""
    if orchestrator is None:
        return {
            "overall_status": "unavailable",
            "error": "Multi-Agent Orchestrator not initialized",
            "agents": {
                "rag_agent": "unavailable",
                "consistency_agent": "unavailable",
                "confidence_agent": "unavailable"
            }
        }

    try:
        status_info = await orchestrator.get_system_status()
        return status_info

    except Exception as e:
        logger.error(f"Error getting multi-agent status: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": "unknown"
        }


@multi_agent_router.get(
    "/",
    summary="API Information",
    description="Get information about the AML Multi-Agent RAG API"
)
async def get_api_info():
    """Get API information and capabilities."""
    return {
        "name": "AML Multi-Agent RAG API",
        "version": "1.0.0",
        "description": (
            "Advanced AML compliance query system with multi-agent validation"
        ),
        "features": [
            "Semantic document retrieval from AML regulatory sources",
            "Consistency validation and citation checking",
            "Confidence calibration and uncertainty analysis",
            (
                "Multi-language support (English and Portuguese) with"
                " automatic detection"
            ),
            "Quality gates and automated recommendations",
            "Enterprise-grade performance monitoring"
        ],
        "endpoints": {
            "POST /api/v1/query": "Fast single-agent AML queries",
            "POST /api/v1/multi-agent/query": (
                "Enhanced multi-agent queries with validation"
            ),
            "GET /api/v1/multi-agent/status": (
                "Multi-agent system health and capabilities"
            ),
            "GET /api/v1/multi-agent/": "Multi-agent API information"
        },
        "documentation": "Visit /docs for interactive API documentation",
        "architecture": "Three-agent system: RAG + Consistency + Confidence"
    }


__all__ = ["multi_agent_router"]
