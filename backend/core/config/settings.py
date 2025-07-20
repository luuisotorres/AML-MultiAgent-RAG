"""
Configuration settings for AML Multi-Agent RAG System.

Manages environment variables and application settings for:
- OpenAI API integration
- Vector database configuration
- Document processing parameters
- Model specifications
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application configuration settings with environment variable support.

    This class manages all configuration parameters for the AML Multi-Agent RAG
    system, automatically loading values from environment variables
    or .env file.
    """
    # OpenAI API key
    openai_api_key: Optional[str] = None

    # Vector DB configuration
    qdrant_url: str = "http://localhost:6333"

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding Model
    embedding_model: str = "text-embedding-3-small"

    # Collection names
    collection_name: str = "aml-documents"

    class Config:
        env_file = ".env"


settings = Settings()
