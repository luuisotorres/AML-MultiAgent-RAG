"""
Confidence Agent for AML Multi-Agent RAG System.

This agent evaluates and adjusts confidence scores for RAG responses by:
1. Analyzing source document quality and relevance scores
2. Evaluating answer completeness and specificity
3. Cross-referencing with domain expertise patterns
4. Providing confidence calibration based on uncertainty indicators
"""

import logging
import re
from statistics import mean, stdev
from typing import Any, Dict, List

from openai import OpenAI

from backend.core.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceAgent:
    """
    Agent responsible for evaluating and calibrating confidence
    in RAG responses.

    This agent analyzes multiple factors to provide accurate confidence scores:
    - Source document quality and semantic similarity scores
    - Answer completeness and specificity
    - Uncertainty indicators in the response
    - Domain-specific confidence patterns
    """

    def __init__(self):
        """Initialize the Confidence Agent."""
        logger.info("Initializing Confidence Agent...")

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please configure it in your "
                "environment."
            )
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized for Confidence Agent")

        self.uncertainty_indicators = [
            "may", "might", "could", "possibly", "potentially", "unclear",
            "depends", "varies", "generally", "typically", "usually",
            "appears", "seems", "suggests", "indicates", "likely"
        ]

        self.confidence_indicators = [
            "required", "mandatory", "must", "shall", "always", "never",
            "specific", "clearly", "defined", "established", "regulated"
        ]

        logger.info("Confidence Agent initialized successfully")

    async def _get_llm_evaluation(
        self, prompt: str, max_tokens: int = 150
    ) -> str:
        """Get evaluation from the LLM."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for "
                            "evaluating text quality."
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

    async def evaluate_confidence(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        base_confidence: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate confidence level for a RAG response.

        Args:
            question (str): Original user question
            answer (str): Generated answer
            sources (List[Dict[str, Any]]): Retrieved source documents
            base_confidence (float, optional): Base confidence from RAG system

        Returns:
            Dict[str, Any]: Confidence evaluation results
        """
        logger.info("Evaluating response confidence...")

        try:
            source_confidence = self._analyze_source_quality(sources)
            answer_confidence = await self._analyze_answer_quality(answer)
            semantic_confidence = await self._analyze_semantic_alignment(
                question, answer, sources
            )
            uncertainty_analysis = self._analyze_uncertainty_indicators(answer)

            confidence_components = {
                "source_quality": source_confidence,
                "answer_quality": answer_confidence,
                "semantic_alignment": semantic_confidence,
                "uncertainty_adjustment": uncertainty_analysis[
                    "confidence_factor"
                ]
            }

            weights = {
                "source_quality": 0.3,
                "answer_quality": 0.25,
                "semantic_alignment": 0.3,
                "uncertainty_adjustment": 0.15
            }

            calibrated_confidence = sum(
                confidence_components[key] * weights[key]
                for key in weights.keys()
            )

            if base_confidence is not None:
                calibrated_confidence = (
                    calibrated_confidence * 0.7 + base_confidence * 0.3
                )

            calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

            confidence_level = self._categorize_confidence(
                calibrated_confidence)

            result = {
                "confidence_score": calibrated_confidence,
                "confidence_level": confidence_level,
                "base_confidence": base_confidence,
                "components": confidence_components,
                "analysis": {
                    "source_analysis": source_confidence,
                    "answer_analysis": answer_confidence,
                    "semantic_analysis": semantic_confidence,
                    "uncertainty_analysis": uncertainty_analysis
                },
                "recommendations": self._generate_confidence_recommendations(
                    calibrated_confidence, uncertainty_analysis
                )
            }

            logger.info(
                f"Confidence evaluation completed - "
                f"Score: {calibrated_confidence:.3f} ({confidence_level})"
            )
            return result

        except Exception as e:
            logger.error(f"Error during confidence evaluation: {e}")
            return {
                "confidence_score": 0.5,
                "confidence_level": "medium",
                "error": str(e),
                "components": {},
                "analysis": {},
                "recommendations": [
                    "Manual review recommended due to evaluation error"
                ]
            }

    def _analyze_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """
        Analyze the quality and relevance of source documents.

        Args:
            sources (List[Dict[str, Any]]): Source documents with scores

        Returns:
            float: Source quality confidence score (0.0 - 1.0)
        """
        logger.debug("Analyzing source quality...")

        if not sources:
            return 0.1

        try:
            scores = []
            for source in sources:
                score = source.get("score", 0.0)
                if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
                    scores.append(score)

            if not scores:
                return 0.3

            avg_score = mean(scores)
            score_consistency = (
                1.0 - (stdev(scores) / avg_score if avg_score > 0 else 0)
            )

            source_count_factor = min(len(sources) / 3.0, 1.0)

            high_quality_sources = sum(1 for score in scores if score >= 0.8)
            quality_ratio = high_quality_sources / len(scores)

            source_confidence = (
                avg_score * 0.4 +
                score_consistency * 0.2 +
                source_count_factor * 0.2 +
                quality_ratio * 0.2
            )

            logger.debug(f"Source quality analysis: {source_confidence:.3f}")
            return min(source_confidence, 1.0)

        except Exception as e:
            logger.error(f"Error analyzing source quality: {e}")
            return 0.3

    async def _analyze_answer_quality(self, answer: str) -> float:
        """
        Analyze the quality and completeness of the generated answer using
        an LLM.

        Args:
            answer (str): Generated answer text

        Returns:
            float: Answer quality confidence score (0.0 - 1.0)
        """
        logger.debug("Analyzing answer quality with LLM...")

        if not answer or len(answer.strip()) < 10:
            return 0.1

        try:
            prompt = f"""
            Please evaluate the quality of the following answer.
            Consider factors like clarity, structure, specificity, and
            professionalism.
            Provide a score from 0.0 to 1.0, where 1.0 is a high-quality
            answer.
            Also, provide a brief rationale for your score.

            Answer: {answer}

            Score:
            Rationale:
            """
            llm_response = await self._get_llm_evaluation(prompt)

            score_match = re.search(r"Score:\s*(\d\.\d+)", llm_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            logger.error(f"Error analyzing answer quality: {e}")
            return 0.3

    async def _analyze_semantic_alignment(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Analyze semantic alignment between question, answer, and sources
        using an LLM.

        Args:
            question (str): Original question
            answer (str): Generated answer
            sources (List[Dict[str, Any]]): Source documents

        Returns:
            float: Semantic alignment confidence score (0.0 - 1.0)
        """
        logger.debug("Analyzing semantic alignment with LLM...")

        try:
            prompt = f"""
            Please evaluate the semantic alignment between the question
            , answer, and sources.
            Provide a score from 0.0 to 1.0, where 1.0 is perfect alignment.
            Also, provide a brief rationale for your score.

            Question: {question}
            Answer: {answer}
            Sources: {[source['content'] for source in sources]}

            Score:
            Rationale:
            """
            llm_response = await self._get_llm_evaluation(prompt)

            score_match = re.search(r"Score:\s*(\d\.\d+)", llm_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            logger.error(f"Error analyzing semantic alignment: {e}")
            return 0.5

    def _analyze_uncertainty_indicators(self, answer: str) -> Dict[str, Any]:
        """
        Analyze uncertainty indicators in the answer text.

        Args:
            answer (str): Generated answer text

        Returns:
            Dict[str, Any]: Uncertainty analysis results
        """
        logger.debug("Analyzing uncertainty indicators...")

        try:
            answer_lower = answer.lower()

            uncertainty_count = sum(
                1 for indicator in self.uncertainty_indicators
                if indicator in answer_lower
            )

            confidence_count = sum(
                1 for indicator in self.confidence_indicators
                if indicator in answer_lower
            )

            total_indicators = uncertainty_count + confidence_count
            uncertainty_ratio = (
                uncertainty_count / total_indicators
                if total_indicators > 0 else 0
            )

            if uncertainty_ratio > 0.6:
                confidence_factor = 0.5
            elif uncertainty_ratio > 0.3:
                confidence_factor = 0.7
            elif confidence_count > uncertainty_count:
                confidence_factor = 1.0
            else:
                confidence_factor = 0.8

            return {
                "uncertainty_indicators": uncertainty_count,
                "confidence_indicators": confidence_count,
                "uncertainty_ratio": uncertainty_ratio,
                "confidence_factor": confidence_factor,
                "analysis": (
                    f"Found {uncertainty_count} uncertainty and "
                    f"{confidence_count} confidence indicators"
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing uncertainty: {e}")
            return {
                "uncertainty_indicators": 0,
                "confidence_indicators": 0,
                "uncertainty_ratio": 0.5,
                "confidence_factor": 0.5,
                "analysis": f"Error in analysis: {e}"
            }

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms for semantic analysis."""
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        words = clean_text.split()

        key_terms = []
        important_patterns = [
            r'\b(?:kyc|aml|cft|cdd|edd|pep|beneficial|ownership)\b',
            r'\b(?:due\s+diligence|customer\s+identification)\b',
            r'\b(?:risk\s+assessment|transaction\s+monitoring)\b',
            r'\b(?:suspicious\s+activity|unusual\s+transaction)\b',
            r'\b(?:compliance|regulatory|requirement|threshold)\b'
        ]

        for pattern in important_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_terms.extend([
                match.lower().replace(' ', '_') for match in matches
            ])

        significant_words = [
            word for word in words
            if len(word) > 4 and word not in [
                "customer", "requirements", "procedures",
                "following", "information"
            ]
        ][:5]

        return list(set(key_terms + significant_words))

    def _check_question_type_addressed(
        self, question: str, answer: str
    ) -> float:
        """Check if the answer addresses the type of question asked."""
        question_lower = question.lower()
        answer_lower = answer.lower()

        question_patterns = [
            (["what", "define", "definition"],
             ["is", "are", "means", "refers"]),
            (["how", "process", "procedure"],
             ["step", "process", "procedure", "method"]),
            (["when", "timeline", "deadline"],
             ["day", "month", "time", "period", "deadline"]),
            (["who", "responsibility"],
             ["responsible", "authority", "entity", "person"]),
            (["where", "jurisdiction"],
             ["country", "region", "jurisdiction", "location"]),
            (["why", "reason", "purpose"],
             ["reason", "purpose", "because", "due to"])
        ]

        for question_words, answer_words in question_patterns:
            if any(word in question_lower for word in question_words):
                if any(word in answer_lower for word in answer_words):
                    return 1.0
                else:
                    return 0.3

        return 0.7

    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence score into levels."""
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"

    def _generate_confidence_recommendations(
        self,
        confidence_score: float,
        uncertainty_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on confidence analysis."""
        recommendations = []

        if confidence_score < 0.4:
            recommendations.append(
                "Very low confidence - consider alternative sources or "
                "expert review"
            )
        elif confidence_score < 0.6:
            recommendations.append(
                "Low confidence - verify with additional sources"
            )
        elif confidence_score < 0.8:
            recommendations.append(
                "Medium confidence - acceptable for general guidance"
            )
        else:
            recommendations.append(
                "High confidence - reliable for decision making"
            )

        if uncertainty_analysis.get("uncertainty_ratio", 0) > 0.5:
            recommendations.append(
                "High uncertainty indicators detected - "
                "consider seeking clarification"
            )

        if uncertainty_analysis.get("confidence_indicators", 0) > 3:
            recommendations.append(
                "Strong regulatory language detected - high reliability"
            )

        return recommendations
