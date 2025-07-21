"""
Consistency Agent for AML Multi-Agent RAG System.

This agent validates the consistency and coherence of RAG responses by:
1. Cross-referencing citations against retrieved documents
2. Checking for contradictory information within sources
3. Validating jurisdiction-specific consistency
4. Identifying gaps between question and retrieved context
"""

import logging
from typing import List, Dict, Any

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
        self.jurisdictions = ["usa", "eu", "brazil"]
        logger.info("Consistency Agent initialized successfully")

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
            citation_check = self._validate_citations(
                answer, sources
            )
            jurisdiction_check = self._validate_jurisdictions(
                question, sources
            )
            contradiction_check = self._check_contradictions(
                question, answer, sources
            )
            relevance_check = self._validate_relevance(
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

    def _validate_citations(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that the answer properly references the provided sources.

        Args:
            answer (str): Generated answer text
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Citation validation results
        """
        logger.debug("Validating citations...")

        try:
            source_filenames = [doc.get("filename", "") for doc in sources]

            citations_found = []
            for filename in source_filenames:
                if filename and (
                    filename.lower() in answer.lower() or
                    any(
                        part in answer.lower()
                        for part in filename.lower().split('.')
                    )
                ):
                    citations_found.append(filename)

            # Score based on citation coverage
            citation_ratio = (
                len(citations_found) / len(source_filenames) if
                source_filenames else 0
            )

            return {
                "score": min(citation_ratio + 0.3, 1.0),
                "citations_found": citations_found,
                "total_sources": len(source_filenames),
                "issues": (
                    [] if citation_ratio > 0.3
                    else ["Low citation coverage in response"]
                )
            }

        except Exception as e:
            logger.error(f"Error validating citations: {e}")
            return (
                {
                    "score": 0.0,
                    "error": str(e),
                    "issues": ["Citation validation failed"]
                }
            )

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

    def _check_contradictions(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for contradictions within the agent's answer.

        Args:
            question (str): User question
            answer (str): Agent's answer
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Contradiction analysis results
        """
        logger.debug("Checking for contradictions...")

        try:
            sentences = answer.split('. ')

            if len(sentences) < 2:
                logger.debug("Answer too short for contradiction check")

                return {
                    "score": 1.0,
                    "contradictions": [],
                    "issues": []
                }

            contradictions = []

            contradiction_patterns = [
                (
                    ["required", "mandatory", "must"],
                    ["optional", "voluntary", "may"]
                ),
                (
                    ["prohibited", "forbidden", "banned"],
                    ["allowed", "permitted", "can"]
                ),
                (
                    ["minimum of", "at least"],
                    ["maximum of", "no more than"]
                ),
            ]

            for pattern in contradiction_patterns:
                positive_sentences = []
                negative_sentences = []

                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    if any(
                        term in sentence_lower for term in pattern[0]
                    ):
                        positive_sentences.append(i)
                    if any(
                        term in sentence_lower for term in pattern[1]
                    ):
                        negative_sentences.append(i)

                if positive_sentences and negative_sentences:
                    logger.debug(
                        f"Found potential contradiction: {pattern[0][0]} vs "
                        f"{pattern[1][0]} "
                        f"in sentences {positive_sentences} and "
                        f"{negative_sentences}"
                    )
                    contradictions.append(
                        {
                            "type": f"{pattern[0][0]} vs {pattern[1][0]}",
                            "positive_sentences": positive_sentences,
                            "negative_sentences": negative_sentences
                        }
                    )

            score = max(1.0 - (len(contradictions) * 0.4), 0.0)

            logger.debug(
                f"Contradiction check completed - "
                f"{len(contradictions)} contradictions found - "
                f"Score: {score}"
            )

            return {
                "score": score,
                "contradictions": contradictions,
                "issues": (
                    [
                        f"Potential contradiction: {c['type']}"
                        for c in contradictions
                    ]
                )
            }

        except Exception as e:
            logger.error(f"Error checking contradictions: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "issues": ["Contradiction check failed"]
            }

    def _validate_relevance(
        self, question: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that sources are relevant to the question.

        Args:
            question (str): User question
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            Dict[str, Any]: Relevance validation results
        """
        logger.debug("Validating relevance...")

        try:
            question_keywords = self._extract_keywords(question.lower())

            relevant_sources = 0
            low_relevance_sources = []

            for i, doc in enumerate(sources):
                content = doc.get("content", "").lower()
                score = doc.get("score", 0.0)

                keyword_matches = sum(
                    1 for keyword in question_keywords if keyword in content
                )
                keyword_ratio = (
                    keyword_matches / len(question_keywords)
                    if question_keywords else 0
                )

                combined_relevance = (score * 0.7) + (keyword_ratio * 0.3)

                if combined_relevance >= 0.5:
                    relevant_sources += 1
                else:
                    low_relevance_sources.append({
                        "source": i,
                        "filename": doc.get("filename", "unknown"),
                        "relevance_score": combined_relevance
                    })

            relevance_ratio = relevant_sources / len(sources) if sources else 0
            score = min(relevance_ratio + 0.2, 1.0)

            issues = []
            if relevance_ratio < 0.6:
                issues.append(
                    f"Low relevance: only {relevant_sources}/{len(sources)} "
                    f"sources highly relevant"
                )

            return {
                "score": score,
                "relevant_sources": relevant_sources,
                "total_sources": len(sources),
                "low_relevance_sources": low_relevance_sources,
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
