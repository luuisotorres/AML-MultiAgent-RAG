"""
Multi-Agent Orchestrator for AML RAG System.

This orchestrator coordinates the RAG, Consistency, and Confidence agents to
provide comprehensive, validated, and calibrated responses to AML compliance
queries.

Flow:
1. RAG Agent generates initial response with sources
2. Consistency Agent validates response coherence and accuracy
3. Confidence Agent evaluates and calibrates confidence scores
4. Orchestrator combines results into final response with quality metrics
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from backend.agents.confidence_agent import ConfidenceAgent
from backend.agents.consistency_agent import ConsistencyAgent
from backend.agents.rag_agent import AMLRagAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents for enhanced AML compliance query processing.

    This orchestrator coordinates:
    - AMLRagAgent: Core RAG functionality with document retrieval
    - ConsistencyAgent: Response validation and coherence checking
    - ConfidenceAgent: Confidence scoring and calibration

    Provides comprehensive responses with quality metrics and recommendations.
    """

    def __init__(self):
        """Initialize the Multi-Agent Orchestrator."""
        logger.info("Initializing Multi-Agent Orchestrator...")

        try:
            self.rag_agent = AMLRagAgent()
            self.consistency_agent = ConsistencyAgent()
            self.confidence_agent = ConfidenceAgent()

            self.min_consistency_threshold = 0.6
            self.min_confidence_threshold = 0.4

            logger.info("Multi-Agent Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing orchestrator: {e}")
            raise

    async def process_query(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Process a query through the multi-agent pipeline.

        Args:
            query (str): The user's AML compliance question

        Returns:
            Dict[str, Any]: Comprehensive response with quality metrics
        """
        logger.info(
            f"Processing query through multi-agent pipeline: "
            f"'{query[:100]}...'"
        )

        start_time = datetime.now()

        try:
            logger.info("Step 1: Generating RAG response...")
            rag_response = await self.rag_agent.query(query)

            if not rag_response or not rag_response.get("answer"):
                logger.warning("RAG agent failed to generate response")
                return self._create_error_response(
                    "Failed to generate initial response",
                    "RAG agent returned empty or invalid response",
                )

            answer = rag_response.get("answer", "")
            sources = rag_response.get("sources", [])
            base_confidence = rag_response.get("confidence", 0.0)

            logger.info("Step 2: Validating response consistency...")

            logger.info("Step 3: Evaluating response confidence...")

            consistency_task = self.consistency_agent.validate_response(
                query, answer, sources
            )
            confidence_task = self.confidence_agent.evaluate_confidence(
                query, answer, sources, base_confidence
            )

            consistency_result, confidence_result = await asyncio.gather(
                consistency_task, confidence_task
            )

            logger.info(
                "Step 4: Combining results and applying quality gates...")

            final_response = await self._combine_agent_results(
                query=query,
                rag_response=rag_response,
                consistency_result=consistency_result,
                confidence_result=confidence_result
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            final_response["metadata"]["processing_time"] = processing_time

            logger.info(
                f"Multi-agent processing completed in {processing_time:.2f}s"
                f" - Quality: {final_response['quality_score']:.3f}"
            )

            return final_response

        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return self._create_error_response(
                "Multi-agent processing failed",
                str(e),
                processing_time
            )

    async def _combine_agent_results(
        self,
        query: str,
        rag_response: Dict[str, Any],
        consistency_result: Dict[str, Any],
        confidence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine results from all agents into a comprehensive response.

        Args:
            query (str): Original query
            rag_response (Dict[str, Any]): RAG agent response
            consistency_result (Dict[str, Any]): Consistency validation results
            confidence_result (Dict[str, Any]): Confidence evaluation results

        Returns:
            Dict[str, Any]: Combined response with quality metrics
        """
        logger.debug("Combining agent results...")

        is_consistent = consistency_result.get("is_consistent", False)
        consistency_score = consistency_result.get("overall_score", 0.0)
        confidence_score = confidence_result.get("confidence_score", 0.0)
        confidence_level = confidence_result.get("confidence_level", "low")

        quality_score = self._calculate_quality_score(
            consistency_score, confidence_score
        )

        quality_gates = {
            "consistency_passed": (
                consistency_score >= self.min_consistency_threshold
            ),
            "confidence_passed": (
                confidence_score >= self.min_confidence_threshold
            ),
            "overall_passed": quality_score >= 0.5
        }

        quality_assessment = self._generate_quality_assessment(
            quality_score, consistency_result, confidence_result, quality_gates
        )

        all_recommendations = []
        all_recommendations.extend(
            consistency_result.get("recommendations", []))
        all_recommendations.extend(
            confidence_result.get("recommendations", []))

        if not quality_gates["overall_passed"]:
            all_recommendations.insert(
                0,
                "⚠️ Response quality below threshold - "
                "manual review recommended"
            )

        response = {
            "success": True,
            "query": query,
            "answer": rag_response.get("answer", ""),
            "sources": rag_response.get("sources", []),

            "quality_score": quality_score,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "consistency_score": consistency_score,
            "is_consistent": is_consistent,
            "quality_gates": quality_gates,
            "quality_assessment": quality_assessment,

            "recommendations": list(set(all_recommendations)),

            "metadata": {
                "detected_language": rag_response.get("detected_language", ""),
                "relevant_jurisdictions": rag_response.get(
                    "relevant_jurisdictions", []
                ),
                "agent_versions": {
                    "rag": "1.0",
                    "consistency": "1.0",
                    "confidence": "1.0",
                    "orchestrator": "1.0"
                },
                "processing_steps": [
                    "RAG response generation",
                    "Consistency validation",
                    "Confidence evaluation",
                    "Result orchestration"
                ]
            }
        }

        return response

    def _calculate_quality_score(
        self,
        consistency_score: float,
        confidence_score: float
    ) -> float:
        """
        Calculate overall quality score from component scores.

        Args:
            consistency_score (float): Consistency validation score
            confidence_score (float): Confidence evaluation score

        Returns:
            float: Overall quality score (0.0 - 1.0)
        """
        quality_score = (consistency_score * 0.6) + (confidence_score * 0.4)
        return min(max(quality_score, 0.0), 1.0)

    def _calculate_avg_source_score(
        self, sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate average score from sources."""
        if not sources:
            return 0.0

        scores = [source.get("score", 0.0) for source in sources]
        valid_scores = [
            score for score in scores if isinstance(score, (int, float))
        ]

        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    def _generate_quality_assessment(
        self,
        quality_score: float,
        consistency_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        quality_gates: Dict[str, bool]
    ) -> str:
        """Generate human-readable quality assessment."""

        if quality_score >= 0.8:
            base_assessment = "🟢 Excellent - High quality response"
        elif quality_score >= 0.6:
            base_assessment = "🟡 Good - Acceptable quality with minor concerns"
        elif quality_score >= 0.4:
            base_assessment = "🟠 Fair - Moderate quality, review recommended"
        else:
            base_assessment = "🔴 Poor - Low quality, manual review required"

        issues = []
        if not quality_gates["consistency_passed"]:
            issues.append("consistency concerns")
        if not quality_gates["confidence_passed"]:
            issues.append("low confidence")

        if issues:
            base_assessment += f" ({', '.join(issues)})"

        return base_assessment

    def _create_error_response(
        self,
        error_message: str,
        error_details: str,
        processing_time: float = None
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_message,
            "error_details": error_details,
            "answer": "",
            "sources": [],
            "quality_score": 0.0,
            "confidence_score": 0.0,
            "confidence_level": "very_low",
            "consistency_score": 0.0,
            "is_consistent": False,
            "quality_gates": {
                "consistency_passed": False,
                "confidence_passed": False,
                "overall_passed": False
            },
            "quality_assessment": "🔴 Error - Unable to process query",
            "recommendations": [
                "Please try rephrasing your question or contact support"
            ],
            "metadata": {
                "processing_time": processing_time,
                "agent_versions": {
                    "orchestrator": "1.0"
                },
                "error_timestamp": datetime.now().isoformat()
            }
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents in the system."""
        logger.info("Checking multi-agent system status...")

        try:
            agents_status = {
                "rag_agent": (
                    "healthy" if hasattr(self, 'rag_agent')
                    else "not_initialized"
                ),
                "consistency_agent": (
                    "healthy" if hasattr(self, 'consistency_agent')
                    else "not_initialized"
                ),
                "confidence_agent": (
                    "healthy" if hasattr(self, 'confidence_agent')
                    else "not_initialized"
                )
            }

            overall_status = "healthy" if all(
                status == "healthy" for status in agents_status.values()
            ) else "degraded"

            return {
                "overall_status": overall_status,
                "agents": agents_status,
                "thresholds": {
                    "min_consistency": self.min_consistency_threshold,
                    "min_confidence": self.min_confidence_threshold
                },
                "capabilities": [
                    "Multi-language AML query processing",
                    "Consistency validation",
                    "Confidence calibration",
                    "Quality gate enforcement",
                    "Comprehensive recommendations"
                ],
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error checking system status: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
