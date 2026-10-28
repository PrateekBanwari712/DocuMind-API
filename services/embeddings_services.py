"""
Embeddings service using HuggingFace models
"""

import asyncio
from typing import List, Optional
import numpy as np

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingsService:
    def __init__(self):
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model"""
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "LangChain HuggingFace library is required. Install with: pip install langchain-huggingface"
            )

        try:
            # Initialize HuggingFace embeddings with the specified model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},  # can use 'cuda if GPU is available
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                f"HuggingFace embeddings initialized with model: {settings.EMBEDDING_MODEL}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text using Google's Gemini embedding model.
        """
        try:
            # Run the synchronous embedding in a thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.embeddings.embed_query(text)
            )

            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ValueError(f"Failed to generate embeddings: {str(e)}")

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        """
        try:
            # Run the synchronous batch embedding in a thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: self.embeddings.embed_documents(texts)
            )

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.info(f"Error generating batch embeddings: {str(e)}")
            raise

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query
        """
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.embeddings.embed_query(query)
            )

            logger.debug(f"Generated query embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise ValueError(f"Failed to generate query embedding: {str(e)}")

    def cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar embeddings to query
        Returns list of (index, similarity_score) tuples
        """
        try:
            similarities = []

            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.cosine_similarity(query_embedding, candidate)
                similarities.append((i, similarity))

            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top k results
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar embeddings: {str(e)}")
            return []
