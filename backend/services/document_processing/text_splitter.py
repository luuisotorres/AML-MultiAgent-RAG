"""
This module splits large PDF documents into smaller, manageable chunks for
embedding generation and vector database storage. Uses LangChain's recursive
character text splitter.

Key Features:
- Splits documents into configurable chunk sizes with overlap
- Preserves document metadata (filename, language, source region)
- Creates unique chunk IDs for tracking and retrieval
- Maintains context continuity between chunks
"""
import os
import json
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.core.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Text chunking service for AML compliance documents.

    Splits large documents into smaller chunks while preserving context
    and metadata for optimal embedding generation and retrieval.

    Attributes:
        chunk_size (int): Maximum characters per chunk
        chunk_overlap (int): Character overlap between chunks
        splitter: LangChain RecursiveCharacterTextSplitter instance
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the text chunker with configurable parameters.

        Args:
            chunk_size: Maximum characters per chunk (default from settings)
            chunk_overlap: Character overlap between chunks
                          (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", "", ". "]
        )

        logging.info(
            f"TextChunker initialized with chunk size: {self.chunk_size}, "
            f"chunk overlap: {self.chunk_overlap}"
        )

    def chunk_documents(
        self, documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Split documents into smaller chunks for embedding generation.

        This method takes processed documents from the PDF processor and splits
        them into smaller chunks while preserving all metadata. Each chunk gets
        a unique ID and maintains reference to its source document.

        Args:
            documents: List of processed documents from PDF processor

        Returns:
            List[Dict[str, str]]: List of chunked documents with preserved
                                 metadata and unique IDs. Each chunk contains:
                                 - chunk_id: Unique identifier
                                 - content: Text content of the chunk
                                 - filename: Original document filename
                                 - language: Detected language
                                 - original_path: Path to source PDF
                                 - source_region: Document region
                                 - chunk_index: Sequential index within
                                 document
        """
        logger.info(
            f"Chunking {len(documents)} documents"
        )
        chunked_docs = []

        for doc in documents:
            logger.info(
                f"Chunking document: {doc['filename']}"
                f" - ({len(doc['content'])} characters)"
            )

            try:
                text_chunk = self.splitter.split_text(doc['content'])
                logger.info(
                    f"Generated {len(text_chunk)} chunks for {doc['filename']}"
                )

                for i, chunk in enumerate(text_chunk):
                    chunked_doc = {
                        "chunk_id": f"{doc['filename']}_chunk_{i}",
                        "content": chunk,
                        "filename": doc['filename'],
                        "language": doc['language'],
                        "original_path": doc['path'],
                        "source_region": doc['source_region'],
                        "chunk_index": i
                    }
                    chunked_docs.append(chunked_doc)
            except Exception as e:
                logger.error(
                    f"Error chunking document {doc['filename']}: {str(e)}"
                )
                continue
        logger.info(
            f"Chunking completed: {len(chunked_docs)} total chunks generated"
        )
        return chunked_docs

    def save_chunks(
        self, chunked_docs: List[Dict[str, str]],
        output_path: str = "docs/processed_docs/chunked_docs.json"
    ) -> str:
        """
        Save chunked documents to JSON file for next pipeline stage.

        This method saves the chunked documents to a JSON file that can be
        consumed by the embedding generation module.

        Args:
            chunked_docs: List of chunked document dictionaries
            output_path: Full path where to save the JSON file

        Returns:
            str: Path to the saved file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunked_docs, f, indent=2, ensure_ascii=False)

            logger.info(f"Chunked documents saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save chunked documents: {str(e)}")
            raise RuntimeError(f"Could not save chunked documents: {str(e)}")


if __name__ == "__main__":
    logger.info("\nStarting text chunking...")

    processed_file = 'docs/processed_docs/processed_docs.json'

    if os.path.exists(processed_file):
        logger.info(
            f"\nLoading processed documents from {processed_file}..."
        )

        with open(processed_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        logger.info(
            f"Found {len(documents)} processed documents."
        )

        chunker = TextChunker()
        chunked_docs = chunker.chunk_documents(documents)

        logger.info(
            f"\nGenerated {len(chunked_docs)} text chunks."
        )

        saved_file = chunker.save_chunks(chunked_docs)

        print("\nSummary:")
        for doc in documents:
            doc_chunks = [
                chunk for chunk in chunked_docs
                if chunk['filename'] == doc['filename']
            ]
            print(
                f"\nDocument: {doc['filename']}"
                f"\nContent Length: {len(doc['content'])} characters"
                f"\nSource Region: {doc['source_region']}"
                f"\nLanguage: {doc['language']}"
                f"\nChunks: {len(doc_chunks)}"
            )

        print("\nSample Chunks:")
        for chunk in chunked_docs[:3]:
            print(
                f"\nChunk ID: {chunk['chunk_id']}"
                f"\nContent: {chunk['content'][:100]}..."
                f"\nLength: {len(chunk['content'])} characters"
            )
    else:
        print(f"\n❌ No processed documents found at {processed_file}.")
