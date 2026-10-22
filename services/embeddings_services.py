"""
Embeddings service using HuggingFace models
"""

import asyncio
from typing import List, Optional

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

    async def generate_embeddings(self, text:str) -> List[float]:
      """ 
      Generate embeddings for a single text using HuggingFace
      """        
      try:
          # Run the synchronous embedding in a thread pool
          loop = asyncio.get_event_loop()
          embedding = await loop.run_in_executor(
              None,
              lambda: self.embeddings.embed_query(text)
          ) 

          logger.debug(f"Generated embedding wih dimension : {len(embedding)}")
          return embedding

      except Exception as e:
          logger.error(f"Error generating embeddings: {str(e)}")
          raise ValueError(f"Failed to generate embeddings: {str(e)}")