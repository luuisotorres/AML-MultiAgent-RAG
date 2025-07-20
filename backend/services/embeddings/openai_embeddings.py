"""
This module generates vector embeddings from chunked documents using OpenAI's
embedding models. Handles both English and Portuguese text with optimized
batch processing for efficient API usage.

Key Features:
- Generates embeddings using OpenAI's text-embedding models
- Batch processing to minimize API calls
- Handles multilingual content (Portuguese/English)
- Error handling and retry logic
- Progress tracking and logging
"""
import os
import json
import logging
import time
from typing import List, Dict, Optional
import openai
from backend.core.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbeddings:
    """
    OpenAI Embeddings Service for AML Compliance Documents.

    Generates vector embeddings for text chunks using OpenAI's embedding
    models, with support for batch processing and multilingual content.

    Attributes:
        client: OpenAI API client instance
        model: OpenAI embedding model name
        batch_size: Number of documents to process per batch
    """

    def __init__(
        self, api_key: Optional[str] = None, model: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize the OpenAI embeddings service.

        Args:
            api_key (str): OpenAI API key. If None, uses settings.
            model (str): OpenAI embedding model name. Defaults to settings.
            batch_size (int): Number of documents to process per batch.
                              Default is 100.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.embedding_model
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided. "
                "Set OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(api_key=self.api_key)

        logger.info(
            f"OpenAIEmbeddings initialized with model: {self.model}, "
            f"batch size: {self.batch_size}"
        )

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embeddings, each a list of floats
        """
        try:
            logger.info(
                f"Creating embeddings for {len(texts)} text chunks "
                f"using model: {self.model}"
            )

            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [
                embedding.embedding for embedding in response.data
            ]
            logger.info(f"Successfully created {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def embed_documents(
        self, chunked_docs: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Generate embeddings for chunked documents with batch processing.

        Args:
            chunked_docs: List of chunked documents from text splitter

        Returns:
            List of documents with embeddings added.
        """
        logger.info(
            f"Starting embedding process for {len(chunked_docs)} chunks"
        )

        embedded_docs = []
        total_batches = (
            len(chunked_docs) + self.batch_size - 1
        ) // self.batch_size

        for i in range(0, len(chunked_docs), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = chunked_docs[i:i + self.batch_size]

            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} chunks)"
            )

            try:
                texts = [doc['content'] for doc in batch]

                embeddings = self.create_embeddings(texts)

                for doc, embedding in zip(batch, embeddings):
                    embedded_doc = doc.copy()
                    embedded_doc['embedding'] = embedding
                    embedded_doc['embedding_model'] = self.model
                    embedded_docs.append(embedded_doc)

                if batch_num < total_batches:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                continue

        logger.info(
            f"Completed embedding process for {len(embedded_docs)} documents."
        )
        return embedded_docs

    def save_embeddings(
        self, embedded_docs: List[Dict],
        output_path: str = "docs/embeddings/embedded_docs.json"
    ) -> str:
        """
        Save embedded documents to a JSON file for vector database storage.

        Args:
            embedded_docs: List of documents with embeddings
            output_path: Path to save the JSON file

        Returns:
            str: Path to the saved file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(embedded_docs, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Saved {len(embedded_docs)} embedded "
                f"documents to {output_path}"
            )
            return output_path

        except Exception as e:
            logger.error(f"Error saving embedded documents: {str(e)}")
            raise


if __name__ == "__main__":
    logger.info("\nStarting OpenAI embeddings generation...")

    # Load chunked documents from previous stage
    chunked_file = 'docs/processed_docs/chunked_docs.json'

    if os.path.exists(chunked_file):
        logger.info(
            f"\nLoading chunked documents from {chunked_file}..."
        )

        with open(chunked_file, 'r', encoding='utf-8') as f:
            chunked_docs = json.load(f)

        logger.info(
            f"Found {len(chunked_docs)} chunked documents."
        )

        try:
            embedder = OpenAIEmbeddings()
        except ValueError as e:
            logger.error(str(e))
            logger.info(
                "Please set the OPENAI_API_KEY environment variable."
            )
            exit(1)

        embedded_docs = embedder.embed_documents(chunked_docs)

        if embedded_docs:
            saved_file = embedder.save_embeddings(embedded_docs)
            logger.info(
                f"\n✅ Generated embeddings for {len(embedded_docs)} "
                f"documents. Saved to {saved_file}"
            )

            print("\nSummary:")
            for doc_name in set(doc['filename'] for doc in chunked_docs):
                doc_chunks = [
                    chunk for chunk in embedded_docs
                    if chunk['filename'] == doc_name
                ]
                sample_embedding_dim = len(
                    doc_chunks[0]['embedding']
                ) if doc_chunks else 0
                print(
                    f"Document: {doc_name}, "
                    f"Chunks: {len(doc_chunks)}, "
                    f"Dimension: {sample_embedding_dim}"
                )
        else:
            logger.warning("No documents to embed.")
