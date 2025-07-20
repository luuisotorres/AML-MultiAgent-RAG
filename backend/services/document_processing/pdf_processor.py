"""
This module provides functionality to extract and process PDF documents
containing AML/FT (Anti-Money Laundering/Financial Terrorism) compliance
documentation.

Key Features:
- Extracts text content from PDF files
- Automatically detects document language (Portuguese/English)
- Processes multiple PDFs in batch
- Prepares documents for embedding generation and vector storage
"""
import os
import json
from typing import List, Dict
import pymupdf as fitz
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF Document Processor for AML Compliance Documents.

    This class handles the extraction and preprocessing of PDF documents
    in the AML Multi-Agent RAG system. It's designed to work with both
    Portuguese and English compliance documents.

    Attributes:
        raw_docs_path (Path): Directory containing raw PDF documents
    """

    def __init__(self, raw_docs_path: str = "docs/raw_docs"):
        """
        Initialize the PDF processor.

        Args:
            raw_docs_path (str): Path to directory containing PDF files.
                                Default is "docs/raw_docs"
        """
        self.raw_docs_path = Path(raw_docs_path)

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, str]:
        """
        Extract text content from a single PDF file.

        This method opens a PDF file, extracts text from all pages,
        detects the document language, and returns structured data
        ready for the next pipeline stage (text chunking).

        Args:
            pdf_path (Path): Path to the PDF file to process

        Returns:
            Dict[str, str]: Document data containing:
                - filename: Name of the PDF file
                - content: Extracted text content
                - language: Detected language ('portuguese' or 'english')
                - path: Full path to the original file
                - source_region: Directory name indicating document region

        Returns None if processing fails.
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()

            doc.close()

            return {
                "filename": pdf_path.name,
                "content": text_content,
                "language": self.detect_language(text_content),
                "path": str(pdf_path),
                "source_region": pdf_path.parent.name
            }
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None

    def detect_language(self, text: str) -> str:
        """
        Detect document language using common word frequency analysis.

        This method uses a simple but effective approach to distinguish
        between Portuguese and English documents by counting occurrences
        of common words in each language.

        Args:
            text (str): Text content to analyze

        Returns:
            str: Detected language ('portuguese' or 'english')]
        """
        portuguese_words = (
            'e', 'de', 'do', 'da', 'em', 'para', 'com', 'não', 'que', 'uma'
        )
        english_words = (
            'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that',
            'this', 'for', 'was', 'on', 'with', 'as', 'by', 'at', 'an'
        )

        text_lower = text.lower()
        pt_count = sum(
            1 for word in portuguese_words if f' {word} ' in text_lower
        )
        en_count = sum(
            1 for word in english_words if f' {word} ' in text_lower
        )

        return 'portuguese' if pt_count > en_count else 'english'

    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """
        Process all PDF files in the configured directory.

        This method scans the raw_docs_path directory for PDF files,
        processes each one to extract text and detect language,
        and returns a list of processed documents ready for the
        next stage in the RAG pipeline (text chunking).

        Returns:
            List[Dict[str, str]]: List of processed documents, where each
                                 document is a dictionary with keys:
                                 - filename, content, language, path,
                                   source_region
        Note:
            Failed document processing is logged but doesn't stop
            the batch processing of other documents.
        """
        documents = []

        if not self.raw_docs_path.exists():
            logger.error(f"Directory {self.raw_docs_path} does not exist.")
            return documents

        pdf_files = list(self.raw_docs_path.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process.")

        for pdf_file in pdf_files:
            logger.info(
                f"Processing {pdf_file.relative_to(self.raw_docs_path)}..."
            )
            doc_data = self.extract_text_from_pdf(pdf_file)
            if doc_data:
                documents.append(doc_data)

        return documents

    def save_processed_documents(
        self, documents: List[Dict[str, str]],
        output_path: str = "docs/processed_docs"
    ):
        """
        Save processed documents to JSON file for next pipeline stage.

        This method saves the processed PDF documents to a JSON file
        that can be consumed by the text chunking module.

        Args:
            documents: List of processed document dictionaries
            output_path: Directory where to save the JSON file

        Returns:
            str: Path to the saved file
        """
        os.makedirs(output_path, exist_ok=True)

        output_file = os.path.join(output_path, "processed_docs.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved {len(documents)} processed documents to {output_file}"
        )
        return output_file


if __name__ == "__main__":
    print("\nStarting PDF processing...")
    processor = PDFProcessor()
    documents = processor.process_all_pdfs()

    if documents:
        saved_file = processor.save_processed_documents(documents)
        print(
            f"\n✅ Processed {len(documents)} documents. Saved to {saved_file}"
        )
        for doc in documents:
            print(
                f"\nDocument: {doc['filename']} ({doc['source_region']}) - "
                f"\nLanguage: {doc['language']} - "
                f"\nContent Length: {len(doc['content'])} characters"
            )
    else:
        print("\n❌ No documents processed.")
