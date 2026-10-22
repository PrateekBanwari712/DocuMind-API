"""
Vector database service using Pinecone
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from utils.logger import get_logger
from core.config import settings

# Pinecone imports
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    pinecone = None
    Pinecone = None
    ServerlessSpec = None

logger = get_logger(__name__)


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations"""

    @abstractmethod
    async def store_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    async def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_by_file_id(self, file_id: str) -> bool:
        pass

    @abstractmethod
    async def list_files(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        pass


class PineconeVectorDB(VectorDBInterface):
    """Pinecone vector database implementation with comprehensive error handling"""

    def __init__(self):
        if Pinecone is None or pinecone is None:
            raise ImportError(
                "Pinecone is required. Insall with: pip install pinecone-client"
            )

        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for Pinecone")

        if not settings.PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_ENVIRONMENT is required for Pinecone")

        try:
            # Initialize Pinecone client with new SDK
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self.embedding_dimension = 768  # Gemini embedding dimension
            self._initialize_index()

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

    def _initialize_index(self):
        """Initialize or create the Pinecone Index"""

        try:
            # List existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone Index: {self.index_name}")
                # Creating index with serverless specification
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",  # or "gcp", "Azure"
                        region=settings.PINECONE_ENVIRONMENT,
                    ),
                )
                # Wait for index to be ready
                import time

                while self.index_name not in [
                    index.name for index in self.pc.list_indexes()
                ]:
                    time.sleep(1)
                logger.info(f"Created Pinecone index: {self.index_name}")

            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists")

            # Connect to the index
            index_stats = self.index.describe_index_stats()
            logger.info(
                f"Pinecone index '{self.index_name}' initialized. Total vectors: {index_stats.total_vector_count}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise

    async def store_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        """Store embeddings in Pinecone with proper metadata handling"""
        if not embeddings:
            logger.warning("No embeddings provided to store")
            return True

        try:
            vectors = []
            for emb in embeddings:
                # Validate embedding structure
                if not all(
                    key in emb for key in ["id", "embedding", "text", "metadata"]
                ):
                    logger.error(f"Invalid embedding structure: {list(emb.keys())}")

                # Prepare metadata with size limitations
                metadata = dict(emb["metadata"]) if emb["metadata"] else {}

                # Store text content in metadata (truncate if too long)
                text_content = emb["text"][:40000]  # Pinecone metadata size limit
                metadata["text"] = text_content

                # Ensure all metadata values are JSON serializable
                for key, value in metadata.items():
                    if isinstance(value, datetime):
                        metadata[key] = value.isoformat()
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        metadata[key] = str(value)

                    vectors.append(
                        {
                            "id": str(emb["id"]),
                            "values": emb["embedding"],
                            "metadata": metadata,
                        }
                    )
                if not vectors:
                    logger.error("No valid vectors to store")
                    return False
                # Batch upsert with error handling
                batch_size = 100
                total_stored = 0

                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i : i + batch_size]
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda b=batch: self.index.upsert(vectors=b)
                    )
                    total_stored += len(batch)
                    logger.debug(
                        f"Stored batch {i//batch_size + 1}: {len(batch)} vectors"
                    )
                except Exception as batch_error:
                    logger.error(
                        f"Error storing batch {i//batch_size + 1}: {str(batch_error)}"
                    )
                    # Continue with next batch instead of failing completely
                    continue
                logger.info(
                    f"Successfully stored {total_stored}/{len(embeddings)} embeddings in Pinecone"
                )
                return total_stored > 0
        except Exception as e:
            logger.error(f"Error storing embeddings in Pinecone :{str(e)}")
            return False

    async def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Pinecone with comprehensive error handling"""
        if not query_embedding:
            logger.error("No query embedding provided")
            return []

        if len(query_embedding) != self.embedding_dimension:
            logger.error(
                f"Query embedding dimension {len(query_embedding)} doesn't match index dimension {self.embedding_dimension}"
            )
            return []

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.index_query(
                    vector=query_embedding,
                    top_k=min(top_k, 1000),  # Pinecone limit
                    include_metadata=True,
                    include_values=False,  # Excluding embedding values in response
                ),
            )

            formatted_results = []
            if "matches" in results:
                for match in results["matches"]:
                    # Extract text from metadata
                    metadata = match.get("metadata", {})
                    text = metadata.get("text", "")

                    # Create a clean metadata dic without the text field
                    clean_metadata = {k: v for k, v in metadata.items() if k != "text"}

                    formatted_results.append(
                        {
                            "id": match["id"],
                            "text": text,
                            "metadata": clean_metadata,
                            "score": match.get("score", 0.0),
                        }
                    )

            logger.info(
                f"Found {len(formatted_results)} similar documents with scores: {[r['score'] for r in formatted_results[:3]] }"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching in Pinecone: {str(e)}")
            return []

    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete all embeddings for a specific file using metadata filtering"""
        if not file_id:
            logger.error("No file_id provided for deletion")
            return False

        try:
            loop = asyncio.get_event_loop()

            # Use delete with filter (requires Pinecone namespaec feature)
            # First, try direct filter based deletion
            try:
                await loop.run_in_executor(
                    None, lambda: self.index.delete(filter={"file_id": file_id})
                )
                logger.info(f"Deleted all embeddings for file {file_id} using filter")
                return True
            except Exception as filter_error:
                logger.warning(
                    f"Filter-based deletion failed: {str(filter_error)}. Trying query-based deletion."
                )

            # Fallback : Query first, then delete by IDs
            # Use a dummy vector for querying
            dummy_vector = [0.0] * self.embedding_dimension

            query_results = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=dummy_vector,
                    filter={"file_id": file_id},
                    top_k=1000,  # to get all matches using large number
                    include_metadata=False,
                ),
            )

            if query_results.get("matches"):
                ids_to_delete = [match["id"] for match in query_results["matches"]]

                # Delete in batches to avoid API limits
                batch_size = 1000
                total_deleted = 0

                for i in range(0, len(ids_to_delete), batch_size):
                    batch_ids = ids_to_delete[i : i + batch_size]
                    await loop.run_in_executor(
                        None, lambda b=batch_ids: self.index.delete(ids=b)
                    )
                    total_deleted += len(batch_ids)
                    logger.debug(
                        f"Deleted batch {i//batch_size + 1} : {len(batch_ids)} vectors"
                    )

                logger.info(f"Deleted {total_deleted} embeddings for file {file_id}")
                return True
            else:
                logger.warning(f"No embeddings found for file {file_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting file from Pinecone: {str(e)}")
            return False

    async def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the database using index statistics"""
        try:
            loop = asyncio.get_event_loop()

            # Get index statistics
            stats = await loop.run_in_executor(
                None, lambda: self.index.describe_index_stats()
            )

            # this is a basic implementation Pinecone doesn't have native file listing

            files_info = []

            # If there will be namespaces, iterating through them
            if hasattr(stats, "namesapce") and stats.namespace:
                for namespace, namespace_stats in stats.namespace.items():
                    files_info.append(
                        {
                            "file_id": namespace,
                            "file_info": f"Namespace: {namespace}",
                            "upload_date": "",
                            "status": "active",
                            "chunk_count": namespace_stats.vector_count,
                        }
                    )
            else:
                # If no namespace, provide general index info
                if stats.total_vector_count > 0:
                    files_info.append(
                        {
                            "file_id": "unknown",
                            "filename": "All documents",
                            "upload_date": "",
                            "status": "active",
                            "chunk_count": stats.total_vector_count,
                        }
                    )

            logger.info(f"Listed {len(files_info)} file entries from Pinecone")
            return files_info

        except Exception as e:
            logger.error(f"Error listing files from Pinecone: {str(e)}")
            return []

    async def get_file_info(self, file_id: str) -> Optional[Dict[str:Any]]:
        """Get information about a specific file using query sampling"""
        if not file_id:
            logger.error("No file_id provided")
            return None

        try:
            loop = asyncio.get_event_loop()

            # Using a dummy vector to query for vectors with the specific file_id
            dummy_vector = [0.0] * self.embedding_dimension

            query_results = await loop.run_in_executor(
                None,
                lambda: self.index_query(
                    vector=dummy_vector,
                    filter={"file_id": file_id},
                    top_k=1,  # Just 1 to check existance and get metadata
                    include_metadata=True,
                ),
            )

            if query_results.get("matches"):
                match = query_results["matches"][0]
                metadata = match.get("metadata", {})

                # Count total chunks for this file
                count_results = await loop.run_in_executor(
                    None,
                    lambda: self.index_query(
                        vector=dummy_vector,
                        filter={"file_id": file_id},
                        top_k=1000,  # Large number to count all
                        include_metadata=False,
                    ),
                )

                chunk_count = len(count_results.get("matches", []))

                file_info = {
                    "file_id": file_id,
                    "filename": metadata.gett("filename", "Unknown"),
                    "upload_date": metadata.get("upload_date", ""),
                    "status": "active",
                    "chunk_count": chunk_count,
                }

                logger.info(f"Retrieved info for file {file_id}: {chunk_count} chunks")
                return file_info
            else:
                logger.warning(f"No embeddings found for file {file_id}")
                return None

        except Exception as e:
            logger.error(f"Error getting file info from Pinecone: {str(e)}")
            return None


class VectorDBService:
    """Vector database service using Pinecone as the sole backend"""

    def __init__(self):
        "Initialize the Pinecone vector database service"
        try:
            self.db = PineconeVectorDB()
            logger.info("VectorDBService initialized with Pinecone backend")
        except Exception as e:
            logger.error(f"Failed to initialize VectorDBService: {str(e)}")
            raise

    async def store_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        """Store embeddings in Pinecone"""
        return await self.db.store_embeddings(embeddings)

    async def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Pinecone"""
        return await self.db.search_similar(query_embedding, top_k)

    async def delete_file(self, file_id: str) -> bool:
        """Delete all embeddings for a specific file from Pinecone"""
        return await self.db.delete_by_file_id(file_id)

    async def list_files(self) -> List[Dict[str, Any]]:
        """list all files in Pinecone"""
        return await self.db.list_files()

    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific file from Pinecone"""
        return await self.db.get_file_info(file_id)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.db.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "index_fullness": getattr(stats, "index_fullness", 0.0),
                "namespace": (
                    dict(stats.namespace)
                    if hasattr(stats, "namespace") and stats.namespace
                    else {}
                ),
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"total_vectors": 0, "index_fullness": 0.0, "namespace": {}}
