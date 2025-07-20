"""
Core AI Agent that handles:
1. Language detection from user queries
2. Vector similarity search in Qdrant
3. Context preparation and answer generation via OpenAI
4. Source citation and confidence scoring
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from backend.core.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AMLRagAgent:
    """
    Main RAG agent for AML compliance queries.

    This agent orchestrates the entire RAG pipeline:
    - Detects query language (Portuguese/English)
    - Searches Qdrant vector database for relevant documents
    - Generates contextual answers using OpenAI GPT models
    - Provides source citations and confidence scoring
    """

    def __init__(self):
        """
        Initialize the AML RAG Agent with OpenAI and Qdrant clients.

        Raises:
            ValueError: If required API keys or configuration are missing
        """
        logger.info("Initializing AML RAG Agent...")

        try:
            # Initialize OpenAI client
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required")

            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")

            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url
            )
            qdrant_host = getattr(settings, 'QDRANT_HOST', 'localhost')
            qdrant_port = getattr(settings, 'QDRANT_PORT', 6333)
            logger.info(
                f"Qdrant client initialized - Host: {qdrant_host}:"
                f"{qdrant_port}"
            )

            self.collection_name = "aml-documents"

        except Exception as e:
            logger.error(f"Failed to initialize AML RAG Agent: {e}")
            raise

    def detect_language(self, question: str) -> str:
        """
        Detect if the question is in Portuguese or English based on keywords.

        Args:
            question (str): The user's query

        Returns:
            str: "Portuguese" or "English"
        """
        logger.debug(f"Detecting language for question: {question[:50]}...")

        portuguese_keywords = [
            "lavagem", "dinheiro", "bcb", "circular", "banco central",
            "brasil", "clientes", "devido", "controle", "suspeitas",
            "regulamentação", "compliance", "prevenção", "identificação"
        ]

        question_lower = question.lower()
        portuguese_matches = [
            keyword for keyword in portuguese_keywords
            if keyword in question_lower
        ]

        if portuguese_matches:
            logger.info(
                f"Language detected: Portuguese "
                f"(matched keywords: {portuguese_matches})"
            )
            return "Portuguese"
        else:
            logger.info("Language detected: English")
            return "English"

    async def search_documents(
        self, question: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents in Qdrant vector database.

        Args:
            question (str): The user's query
            limit (int, optional): Maximum number of documents to retrieve.
                Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of relevant document chunks with
                metadata
        """
        logger.info(
            f"Searching for documents related to: '{question[:100]}...'"
        )

        try:
            # Generate embedding for the question
            logger.debug("Generating embedding for user query...")
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
            query_vector = embedding_response.data[0].embedding
            logger.debug(
                f"Generated embedding with {len(query_vector)} dimensions"
            )

            # Search in Qdrant
            logger.debug(
                f"Searching Qdrant collection '{self.collection_name}' "
                f"with limit {limit}"
            )
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            # Process search results
            documents = []
            for i, hit in enumerate(search_result):
                doc = {
                    "content": hit.payload.get("content", ""),
                    "filename": hit.payload.get("filename", ""),
                    "language": hit.payload.get("language", ""),
                    "region": hit.payload.get("source_region", ""),
                    "score": hit.score
                }
                documents.append(doc)
                logger.debug(
                    f"Result {i+1}: {doc['filename']} "
                    f"(score: {doc['score']:.3f})"
                )

            logger.info(f"Found {len(documents)} relevant documents")
            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def generate_answer(
        self, question: str, context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate an answer using OpenAI with retrieved context documents.

        Args:
            question (str): The user's query
            context_documents (List[Dict[str, Any]]): Retrieved document chunks

        Returns:
            Dict[str, Any]: Complete response with answer, sources, and
                metadata
        """
        logger.info("Generating answer using OpenAI with retrieved context")

        # Prepare context from retrieved documents
        context_parts = []
        for doc in context_documents:
            context_part = (
                f"Source: {doc['filename']} ({doc['region']} - "
                f"{doc['language']})\n{doc['content']}"
            )
            context_parts.append(context_part)

        context = "\n\n".join(context_parts)
        logger.debug(
            f"Prepared context with {len(context)} characters from "
            f"{len(context_documents)} documents"
        )

        # Detect language for appropriate prompting
        detected_language = self.detect_language(question)

        # Create system prompt based on language
        if detected_language == "Portuguese":
            system_prompt = (
                "Você é um especialista em regulamentações AML "
                "(Anti-Money Laundering) e CFT "
                "(Counter-Financing of Terrorism). "
                "Baseando-se estritamente nos documentos fornecidos "
                "como contexto, "
                "responda à pergunta de forma precisa e profissional "
                "em português. "
                "Se a informação não estiver disponível nos documentos "
                "fornecidos, "
                "indique isso claramente. "
                "Cite sempre as fontes específicas quando possível."
            )
        else:
            system_prompt = (
                "You are an expert in AML (Anti-Money Laundering) "
                "and CFT "
                "(Counter-Financing of Terrorism) regulations. "
                "Based strictly on the provided documents as context, "
                "answer the "
                "question accurately and professionally in English. "
                "If information is not available in the provided "
                "documents, "
                "clearly indicate this. "
                "Always cite specific sources when possible."
            )

        try:
            logger.debug(f"Calling OpenAI API with {detected_language} prompt")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\nQuestion: {question}"
                        )
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            logger.info("Answer generated successfully")
            logger.debug(f"Answer length: {len(answer)} characters")

            # Extract relevant jurisdictions from context
            relevant_jurisdictions = list(set([
                doc["region"] for doc in context_documents if doc["region"]
            ]))

            # Calculate average confidence score
            confidence = (
                sum([doc["score"] for doc in context_documents]) /
                len(context_documents)
                if context_documents else 0.0
            )

            result = {
                "answer": answer,
                "detected_language": detected_language,
                "relevant_jurisdictions": relevant_jurisdictions,
                "sources": context_documents,
                "confidence": confidence
            }

            logger.info(
                f"Generated response - "
                f"Jurisdictions: {relevant_jurisdictions}, "
                f"Confidence: {confidence:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            error_message = (
                "Desculpe, mas encontrei um erro ao processar "
                "sua pergunta."
                if detected_language == "Portuguese"
                else "I apologize, but I encountered an error while "
                "processing your request."
            )

            return {
                "answer": error_message,
                "detected_language": detected_language,
                "relevant_jurisdictions": [],
                "sources": [],
                "confidence": 0.0
            }

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Main query method that orchestrates the complete RAG process.

        This method:
        1. Validates the input question
        2. Searches for relevant documents in the vector database
        3. Generates a contextual answer using retrieved information
        4. Returns a complete response with sources and metadata

        Args:
            question (str): The user's AML compliance query

        Returns:
            Dict[str, Any]: Complete response containing:
                - answer: Generated response text
                - detected_language: Question language (Portuguese/English)
                - relevant_jurisdictions: List of applicable jurisdictions
                - sources: List of source documents used
                - confidence: Average relevance score
        """
        logger.info(f"Processing new query: '{question[:100]}...'")

        # Validate input
        if not question or not question.strip():
            logger.warning("Empty or invalid question received")
            return {
                "answer": "Please provide a valid question.",
                "detected_language": "English",
                "relevant_jurisdictions": [],
                "sources": [],
                "confidence": 0.0
            }

        try:
            # Step 1: Search for relevant documents
            logger.info("Step 1: Searching for relevant documents...")
            documents = await self.search_documents(question.strip())

            if not documents:
                logger.warning("No relevant documents found for the query")
                detected_language = self.detect_language(question)
                no_results_message = (
                    "Desculpe, não encontrei documentos relevantes "
                    "para sua pergunta."
                    if detected_language == "Portuguese"
                    else "Sorry, I couldn't find relevant documents "
                    "for your question."
                )

                return {
                    "answer": no_results_message,
                    "detected_language": detected_language,
                    "relevant_jurisdictions": [],
                    "sources": [],
                    "confidence": 0.0
                }

            # Step 2: Generate answer with context
            logger.info("Step 2: Generating answer with retrieved context...")
            result = await self.generate_answer(question.strip(), documents)

            logger.info("Query processing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Unexpected error during query processing: {e}")
            return {
                "answer": (
                    "An unexpected error occurred while processing your "
                    "question. Please try again."
                ),
                "detected_language": "English",
                "relevant_jurisdictions": [],
                "sources": [],
                "confidence": 0.0
            }
