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
    async def store_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> bool:
        pass

    @abstractmethod
    async def search_similar(
        self, query_text: str, top_k: int = 5
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

        if not settings.PINECONE_DB_API_KEY:
            raise ValueError("PINECONE_DB_API_KEY is required for Pinecone")

        if not settings.PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_ENVIRONMENT is required for Pinecone")

        try:
            # Initialize Pinecone client with new SDK
            self.pc = Pinecone(api_key=settings.PINECONE_DB_API_KEY)
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
            self.index = self.pc.Index(self.index_name)

            # Verify index configuration
            index_stats = self.index.describe_index_stats()
            logger.info(
                f"Pinecone index '{self.index_name}' initialized. Total vectors: {index_stats.total_vector_count}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise

    async def store_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> bool:
        """Store embeddings in Pinecone with proper metadata handling"""
        if not embeddings:
            logger.warning("No embeddings provided to store")
            return True

        try:
            documents = []
            for emb in embeddings:
                # Validate embedding structure
                if not all(key in emb for key in ["id", "text", "metadata"]):
                    logger.error(f"Invalid embedding structure: {list(emb.keys())}")
                    continue

                metadata = dict(emb["metadata"] or {})
                text_content = str(emb["text"])[:4000]
                metadata["chunk_text"] = text_content

                clean_metadata = {}

                # Ensure all metadata values are JSON serializable
                for key, value in metadata.items():
                    if value is None:
                        continue
                    elif isinstance(value, datetime):
                        clean_metadata[key] = value.isoformat()
                    elif isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    elif isinstance(value, list) and all(
                        isinstance(item, str) for item in value
                    ):
                        clean_metadata[key] = value
                    else:
                        clean_metadata[key] = str(value)

                documents.append(
                    {
                        "_id": str(emb["id"]),
                        **clean_metadata,
                    }
                )

            if not documents:
                logger.error("No valid documents to store")
                return False

            # Batch upsert with error handling
            batch_size = 100
            total_stored = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                try:
                    response = await asyncio.to_thread(
                        self.index.documents.upsert,
                        documents=batch,
                        namespace=namespace,
                    )

                    total_stored += response.upserted_count
                    logger.info(
                        "Stored batch %s: %s documents",
                        i // batch_size + 1,
                        response.upserted_count,
                    )

                except Exception:
                    logger.exception(
                        "Error storing batch %s",
                        i // batch_size + 1,
                    )
            logger.info(
                f"Successfully stored {total_stored}/{len(documents)} embeddings in Pinecone"
            )
            return total_stored > 0

        except Exception as e:
            logger.error(f"Error storing embeddings in Pinecone: {str(e)}")
            return False

    async def search_similar(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Pinecone with comprehensive error handling"""
        if not query_text or not query_text.strip():
            logger.error("No query text provided")
            return []

        if top_k < 1:
            logger.error("top_k must be at least 1")
            return []

        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            namespaces = getattr(stats, "namespaces", None) or {}
            namespace_names = list(namespaces) or [""]
            limit = min(top_k, 1000)
            query_tasks = [
                asyncio.to_thread(
                    self.index.documents.search,
                    score_by=[
                        {
                            "type": "embed",
                            "field": "chunk_text",
                            "query": query_text,
                        }
                    ],
                    top_k=limit,
                    namespace=namespace,
                    include_fields=["*"],
                )
                for namespace in namespace_names
            ]
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            successful_results = [
                result for result in results if not isinstance(result, Exception)
            ]
            if not successful_results:
                raise RuntimeError("Pinecone search failed for every namespace") from next(
                    result for result in results if isinstance(result, Exception)
                )

            formatted_results = []
            for namespace, result in zip(namespace_names, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Error searching Pinecone namespace %s: %s", namespace, result
                    )
                    continue

                for document in getattr(result, "matches", []) or []:
                    fields = document.to_dict()
                    metadata = {
                        key: value
                        for key, value in fields.items()
                        if key not in {"_id", "_score", "chunk_text"}
                    }
                    formatted_results.append(
                        {
                            "_id": document.id,
                            "text": document.get("chunk_text", ""),
                            "metadata": metadata,
                            "score": document.score or 0.0,
                        }
                    )

            formatted_results.sort(
                key=lambda result: result["score"], reverse=True
            )
            formatted_results = formatted_results[:top_k]
            logger.info(
                f"Found {len(formatted_results)} similar documents with scores: {[r['score'] for r in formatted_results[:3]] }"
            )
            return formatted_results

        except Exception as e:
            logger.exception("Error searching in Pinecone: %s", e)
            raise

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

    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
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

    async def store_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> bool:
        """Store embeddings in Pinecone"""
        return await self.db.store_embeddings(embeddings, namespace)

    async def search_similar(
        self, query_text: str, top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone"""
        return await self.db.search_similar(query_text, top_k)

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
