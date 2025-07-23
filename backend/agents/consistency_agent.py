"""
Consistency Agent for AML Multi-Agent RAG System.

This agent validates the consistency and coherence of RAG responses by:
1. Cross-referencing citations against retrieved documents
2. Checking for contradictory information within sources
3. Validating jurisdiction-specific consistency
4. Identifying gaps between question and retrieved context
"""

import logging
from typing import Any, Dict, List
import re
from openai import OpenAI
from backend.core.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsistencyAgent:
    """
    Agent responsible for validating consistency in AML compliance responses.

    This agent analyzes RAG outputs to ensure:
    - Citations match retrieved documents
    - No contradictory information within sources
    - Jurisdiction-specific requirements are consistent
    - Answer addresses the asked question appropriately
    """

    def __init__(self):
        """Initialize the Consistency Agent."""
        logger.info("Initializing Consistency Agent...")
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please configure it in your "
                "environment."
            )
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized for Consistency Agent")
        self.jurisdictions = ["usa", "eu", "brazil"]
        logger.info("Consistency Agent initialized successfully")

    async def _get_llm_evaluation(
        self, prompt: str, max_tokens: int = 200
    ) -> str:
        """Get evaluation from the LLM."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in AML compliance "
                            "and consistency checking."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting LLM evaluation: {e}")
            return ""

    async def validate_response(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate the consistency of a RAG response.

        Args:
            question (str): The original user question
            answer (str): The generated answer
            sources (List[Dict[str, Any]]): Retrieved source documents

        Returns:
            Dict[str, Any]: Consistency validation results
        """
        logger.info("Validating response consistency...")

        try:
            # Run all consistency checks
            citation_check = await self._validate_citations(
                answer, sources
            )
            jurisdiction_check = self._validate_jurisdictions(
                question, sources
            )
            contradiction_check = await self._check_contradictions(
                question, answer, sources
            )
            relevance_check = await self._validate_relevance(
                question, sources
            )

            checks = (
                [
                    citation_check,
                    jurisdiction_check,
                    contradiction_check,
                    relevance_check
                ]
            )
            overall_score = (
                sum(check["score"] for check in checks) / len(checks)
            )

            is_consistent = overall_score >= 0.6  # 60% threshold

            result = {
                "is_consistent": is_consistent,
                "overall_score": overall_score,
                "checks": {
                    "citations": citation_check,
                    "jurisdictions": jurisdiction_check,
                    "contradictions": contradiction_check,
                    "relevance": relevance_check
                },
                "recommendations": self._generate_recommendations(
                    checks, overall_score
                )
            }

            logger.info(
                f"Consistency validation completed - "
                f"Overall score: {overall_score:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error during consistency validation: {e}")
            return {
                "is_consistent": False,
                "overall_score": 0.0,
                "error": str(e),
                "checks": {},
                "recommendations": (
                    [
                        "Manual review recommended due to validation error"
                    ]
                )
            }

    async def _validate_citations(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that the answer properly references the provided sources
        using an LLM.

        Args:
            answer (str): Generated answer text
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Citation validation results
        """
        logger.debug("Validating citations with LLM...")

        try:
            prompt = f"""
            Please evaluate if the answer properly cites the provided sources.
            Provide a score from 0.0 to 1.0, where 1.0 means all claims are
            well-supported by citations.
            Also, list any claims in the answer that are not
            supported by the sources.

            Answer: {answer}
            Sources: {[source['content'] for source in sources]}

            Score:
            Unsupported Claims:
            """
            llm_response = await self._get_llm_evaluation(prompt)

            score_match = re.search(r"Score:\s*(\d\.\d+)", llm_response)
            score = float(score_match.group(1)) if score_match else 0.5

            unsupported_claims_match = re.search(
                r"Unsupported Claims:\s*(.*)", llm_response, re.DOTALL
            )
            if unsupported_claims_match:
                issues = unsupported_claims_match.group(1).strip().split('\n')
            else:
                issues = []

            return {
                "score": score,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Error validating citations: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "issues": ["Citation validation failed"]
            }

    def _validate_jurisdictions(
        self, question: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate jurisdiction consistency in sources.

        Args:
            question (str): User question
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Jurisdiction validation results
        """
        logger.debug("Validating jurisdictions...")

        try:
            source_jurisdictions = (
                [
                    doc.get("region", "").lower() for doc in sources
                    if doc.get("region")
                ]
            )
            unique_jurisdictions = list(set(source_jurisdictions))

            question_lower = question.lower()
            mentioned_jurisdictions = []

            jurisdiction_keywords = {
                "usa": [
                    "usa",
                    "united states",
                    "america",
                    "patriot act",
                    "bsa",
                    "fincen"
                ],
                "eu": [
                    "eu",
                    "european",
                    "europe",
                    "amld",
                    "directive"
                ],
                "brazil": [
                    "brazil",
                    "brasil",
                    "bcb",
                    "banco central",
                    "circular"
                ]
            }

            for jurisdiction, keywords in jurisdiction_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    mentioned_jurisdictions.append(jurisdiction)

            issues = []
            score = 1.0

            if mentioned_jurisdictions:
                matching_sources = (
                    [
                        j for j in source_jurisdictions
                        if j in mentioned_jurisdictions
                    ]
                )
                if not matching_sources:
                    issues.append(
                        f"Question mentions {mentioned_jurisdictions} "
                        f"but sources are from {unique_jurisdictions}"
                    )
                    score *= 0.5

            if len(unique_jurisdictions) > 2:
                issues.append(
                    f"Multiple jurisdictions present: {unique_jurisdictions}"
                )
                score *= 0.8

            return {
                "score": score,
                "source_jurisdictions": unique_jurisdictions,
                "mentioned_jurisdictions": mentioned_jurisdictions,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Error validating jurisdictions: {e}")
            return (
                {
                    "score": 0.0, "error": str(e),
                    "issues": ["Jurisdiction validation failed"]
                }
            )

    async def _check_contradictions(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for contradictions within the agent's answer and sources
        using an LLM.

        Args:
            question (str): User question
            answer (str): Agent's answer
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Contradiction analysis results
        """
        logger.debug("Checking for contradictions with LLM...")

        try:
            prompt = f"""
            Please check for any contradictions in the provided answer or
            between the answer and the sources.
            Provide a score from 0.0 to 1.0, where 1.0 means no contradictions
            were found.
            Also, list any contradictions you find.

            Answer: {answer}
            Sources: {[source['content'] for source in sources]}

            Score:
            Contradictions:
            """
            llm_response = await self._get_llm_evaluation(prompt)

            score_match = re.search(r"Score:\s*(\d\.\d+)", llm_response)
            score = float(score_match.group(1)) if score_match else 0.5

            contradictions_match = re.search(
                r"Contradictions:\s*(.*)", llm_response, re.DOTALL
            )
            issues = (
                contradictions_match.group(1).strip().split('\n')
                if contradictions_match else []
            )

            return {
                "score": score,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Error checking contradictions: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "issues": ["Contradiction check failed"]
            }

    async def _validate_relevance(
        self, question: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that sources are relevant to the question using an LLM.

        Args:
            question (str): User question
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Relevance validation results
        """
        logger.debug("Validating relevance with LLM...")

        try:
            prompt = f"""
            Please evaluate the relevance of the provided
            sources to the question.
            Provide a score from 0.0 to 1.0, where 1.0 means all sources
            are highly relevant.
            Also, list any sources that are not relevant.

            Question: {question}
            Sources: {[source['content'] for source in sources]}

            Score:
            Irrelevant Sources:
            """
            llm_response = await self._get_llm_evaluation(prompt)

            score_match = re.search(r"Score:\s*(\d\.\d+)", llm_response)
            score = float(score_match.group(1)) if score_match else 0.5

            irrelevant_sources_match = re.search(
                r"Irrelevant Sources:\s*(.*)", llm_response, re.DOTALL
            )
            issues = (
                irrelevant_sources_match.group(1).strip().split('\n')
                if irrelevant_sources_match else []
            )

            return {
                "score": score,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Error validating relevance: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "issues": ["Relevance validation failed"]
            }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text for relevance analysis."""
        # Common AML/compliance keywords
        important_terms = [
            "kyc", "customer", "identification", "due diligence", "aml", "cft",
            "suspicious", "transaction", "reporting", "compliance", "risk",
            "beneficial", "ownership", "enhanced", "simplified", "monitoring"
        ]

        words = text.split()
        keywords = []

        for word in words:
            clean_word = word.strip(".,!?:;\"'()[]{}").lower()
            if clean_word in important_terms:
                keywords.append(clean_word)

        for word in words:
            clean_word = word.strip(".,!?:;\"'()[]{}").upper()
            if (
                len(clean_word) >= 2
                and len(clean_word) <= 4
                and clean_word.isalpha()
            ):
                keywords.append(clean_word.lower())

        return list(set(keywords))

    def _generate_recommendations(
        self, checks: List[Dict[str, Any]], overall_score: float
    ) -> List[str]:
        """Generate recommendations based on consistency analysis."""
        recommendations = []

        if overall_score < 0.7:
            recommendations.append(
                "Consider manual review due to low consistency score"
            )

        # Specific recommendations based on individual checks
        for check in checks:
            if check.get("score", 0) < 0.6:
                issues = check.get("issues", [])
                recommendations.extend(issues)

        if not recommendations:
            recommendations.append("Response passes consistency validation")

        return recommendations
