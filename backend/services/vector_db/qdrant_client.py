"""
This module handles storage and retrieval of document embeddings using Qdrant
vector database. Provides functionality for creating collections, storing
embeddings, and performing similarity searches for AML compliance queries.

Key Features:
- Creates and manages Qdrant collections
- Stores document embeddings with metadata
- Performs similarity searches for efficient retrieval
- Handles multilingual content (Portuguese/English)
- Supports batch processing for efficient API usage
"""

import os
import json
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from backend.core.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantVectorDB:
    """
    Qdrant vector database service for AML compliance documents.

    Handles storage, inexing, and retrieval of document embeddings
    with metadata preservation for the RAG system.

    Attributes:
        client (str): QdrantClient client instance
        collection_name (str): Name of the Qdrant collection
        vector_size (int): Dimension of the embedding vectors
    """

    def __init__(
        self, url: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 1536  # Default OpenAI model embedding size
    ):
        """
        Initialize the Qdrant vector database service.

        Args:
            url (str): Qdrant server URL. If None, uses settings.
            collection_name (str): Name of the collection to use.
                                   Use settings if not provided.
            vector_size (int): Dimension of the embedding vectors.
                               Default is 1536 for OpenAI's
                               text-embedding-3-small model.
        """
        self.url = url or settings.qdrant_url
        self.collection_name = (
            collection_name or settings.collection_name
        )
        self.vector_size = vector_size

        try:
            self.client = QdrantClient(url=self.url)
            logger.info(f"Connected to Qdrant at {self.url}")

            collections = self.client.get_collections()
            logger.info(
                f"Found {len(collections.collections)} existing collections"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create a new collection for storing document embeddings.

        Args:
            recreate (bool): Whether to recreate the collection if it already
                            exists. Defaults to False.
        Returns:
            bool: True if collection was created or recreated.
        """
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if collection_exists:
                if recreate:
                    logger.info(
                        "Recreating existing collection: "
                        f"{self.collection_name}"
                    )
                    self.client.delete_collection(self.collection_name)

                else:
                    logger.info(
                        f"Collection {self.collection_name} already exists."
                    )
                    return True

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

            logger.info(
                f"Created collection: {self.collection_name} with"
                f" {self.vector_size} dimensions and "
                f"{Distance.COSINE} distance."
            )
            return True

        except Exception as e:
            logger.error(
                f"Error creating collection: {e}"
            )
            return False

    def store_embeddings(
        self, embedded_docs: List[Dict]
    ) -> bool:
        """
        Store document embeddings in the Qdrant collection with metadata.

        Args:
            embedded_docs (List[Dict]): List of documents with embeddings
                                        and metadata.

        Returns:
            bool: True if embeddings were stored successfully.
        """
        logging.info(
            f"Storing {len(embedded_docs)} document embeddings in Qdrant"
        )

        try:
            self.create_collection(recreate=True)

            points = []

            for i, doc in enumerate(embedded_docs):
                point = PointStruct(
                    id=i,
                    vector=doc['embedding'],
                    payload={
                        "chunk_id": doc['chunk_id'],
                        "content": doc['content'],
                        "filename": doc['filename'],
                        "language": doc['language'],
                        "source_region": doc['source_region'],
                        "original_path": doc['original_path'],
                        "chunk_index": doc['chunk_index'],
                        "embedding_model": doc['embedding_model']
                    }
                )
                points.append(point)

            batch_size = 100
            total_batches = (
                (len(points) + batch_size - 1) // batch_size
            )

            for i in range(0, len(points), batch_size):
                batch_num = i // batch_size + 1
                batch = points[i:i + batch_size]

                logger.info(
                    f"Uploading batch {batch_num}/{total_batches} "
                    f"({len(batch)} documents)"
                )

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )

            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            logger.info(
                f"Stored {collection_info.points_count} embeddings in "
                "Qdrant."
            )

            return True

        except Exception as e:
            logger.error(
                f"Error storing embeddings in Qdrant: {e}"
            )
            return False

    def search_similar(
        self, query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_vector (List[float]): Query embedding vector.
            limit (int): Maximum number of results to return.
            score_threshold (float): Minimum similarity score (0 to 1).

        Returns:
            List[Dict]: List of matching documents with metadata.
        """
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )

            results = []
            for result in search_results:
                doc = {
                    "score": result.score,
                    "chunk_id": result.payload.get("chunk_id"),
                    "content": result.payload.get("content"),
                    "filename": result.payload.get("filename"),
                    "language": result.payload.get("language"),
                    "source_region": result.payload.get("source_region"),
                }
                results.append(doc)

            logger.info(
                f"Found {len(results)} similar documents."
            )
            return results

        except Exception as e:
            logger.error(
                f"Error searching similar documents: {e}"
            )
            return []

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        """
        try:
            collection_info = self.client.get_collection(
                self.collection_name
            )

            stats = {
                "total_documents": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": (
                    collection_info.config.params.vectors.distance.name
                ),
                "collection_name": self.collection_name
            }

            return stats

        except Exception as e:
            logger.error(
                f"Error retrieving collection stats: {e}"
            )
            return {}


if __name__ == "__main__":
    logger.info("Qdrant Vector DB service initialized.")

    embedded_file = 'docs/embeddings/embedded_docs.json'

    if not os.path.exists(embedded_file):
        logger.error(f"Embedded documents file not found: {embedded_file}")
        exit(1)

    logging.info(f"Loading embedded documents from {embedded_file}")

    with open(embedded_file, 'r', encoding='utf-8') as f:
        embedded_docs = json.load(f)

    logger.info(f"Loaded {len(embedded_docs)} embedded documents.")

    # Initialize Qdrant client
    try:
        qdrant = QdrantVectorDB()
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        exit(1)

    # Store embeddings in Qdrant
    success = qdrant.store_embeddings(embedded_docs)
    if success:
        logger.info("\n✅ Vector Database Storage Complete!")

        stats = qdrant.get_collection_stats()
        print("\nCollection Statistics:")
        print(f" Collection Name: {stats['collection_name']}")
        print(f" Total Documents: {stats['total_documents']}")
        print(f" Vector Size: {stats['vector_size']}")
        print(f" Distance Metric: {stats['distance_metric']}")

        print("\n Documents by Region:")
        regions = {}
        for doc in embedded_docs:
            region = doc['source_region']
            if region not in regions:
                regions[region] = 0
            regions[region] += 1

        for region, count in regions.items():
            print(f" {region}: {count} documents")
    else:
        logger.error("Failed to store embeddings in Qdrant.")
