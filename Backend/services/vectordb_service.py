"""
Vector database service using Pinecone (Documents API — schema-based index)
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
from utils.logger import get_logger
from core.config import settings

# Pinecone imports
try:
    import pinecone
    from pinecone import Pinecone

    # SchemaBuilder is available in pinecone >= 5.x (Documents API)
    try:
        from pinecone import SchemaBuilder

        SCHEMA_API_AVAILABLE = True
    except ImportError:
        SCHEMA_API_AVAILABLE = False
        SchemaBuilder = None
except ImportError:
    pinecone = None
    Pinecone = None
    SCHEMA_API_AVAILABLE = False
    SchemaBuilder = None

logger = get_logger(__name__)


# Formatting search field for llm
def _format_search_match(match: Any) -> Dict[str, Any]:
    """Normalize Pinecone Documents API hits for the RAG pipeline."""
    if isinstance(match, dict):
        match_data = match
    else:
        to_dict = getattr(match, "to_dict", None)
        match_data = to_dict() if callable(to_dict) else {}

    fields = match_data.get("fields") or getattr(match, "fields", None) or {}
    metadata = match_data.get("metadata") or getattr(match, "metadata", None) or {}
    if not isinstance(fields, dict):
        fields = {}
    if not isinstance(metadata, dict):
        metadata = {}

    text = (
        fields.get("chunk_text")
        or match_data.get("chunk_text")
        or metadata.get("chunk_text")
        or getattr(match, "chunk_text", "")
    )
    normalized_metadata = {
        key: value
        for key, value in match_data.items()
        if key
        not in {"_id", "id", "_score", "score", "chunk_text", "fields", "metadata"}
    }
    normalized_metadata.update(fields)
    normalized_metadata.update(
        {key: value for key, value in metadata.items() if key != "chunk_text"}
    )

    document_id = (
        match_data.get("_id") or match_data.get("id") or getattr(match, "id", "")
    )
    score = match_data.get("_score")
    if score is None:
        score = match_data.get("score")
    if score is None:
        score = getattr(match, "score", 0.0)

    return {
        "_id": document_id,
        "text": text or "",
        "metadata": normalized_metadata,
        "score": score or 0.0,
    }


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations"""

    @abstractmethod
    async def store_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> bool:
        pass

    @abstractmethod
    async def search_similar(
        self, query_embedding: List[float], query_text: str, top_k: int = 5
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


# Pinecone Documents-API implementation


class PineconeVectorDB(VectorDBInterface):
    """
    Pinecone vector database implementation with error handling.
    """

    # Name of the dense-vector field declared in the index schema
    VECTOR_FIELD = "embedding"

    def __init__(self):
        if Pinecone is None or pinecone is None:
            raise ImportError(
                "Pinecone package is required. Install with: pip install pinecone"
            )

        if not settings.PINECONE_DB_API_KEY:
            raise ValueError("PINECONE_DB_API_KEY is required for Pinecone")

        if not settings.PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_ENVIRONMENT is required for Pinecone")

        # HuggingFace sentence-transformers/all-mpnet-base-v2 produces 768-dim vectors
        self.embedding_dimension = 768

        try:
            self.pc = Pinecone(api_key=settings.PINECONE_DB_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self._initialize_index()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

    # Index bootstrap
    def _initialize_index(self):
        """Create (if needed) a schema-based Pinecone index and connect to it."""
        try:
            existing_names = [idx.name for idx in self.pc.list_indexes()]

            if self.index_name not in existing_names:
                logger.info(f"Creating schema-based Pinecone index: {self.index_name}")
                self._create_schema_index()
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists")

            # Connect via host URL — required by the Documents API
            index_description = self.pc.describe_index(self.index_name)
            host = index_description.host
            self.index = self.pc.Index(host=host)

            # Quick connectivity check
            stats = self.index.describe_index_stats()
            total = getattr(stats, "total_vector_count", "n/a")
            logger.info(
                f"Pinecone index '{self.index_name}' ready. Total vectors: {total}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise

    def _create_schema_index(self):
        """
        Create a schema-based index using SchemaBuilder.
        Falls back to a legacy ServerlessSpec index if SchemaBuilder is unavailable.

        """
        from pinecone import ServerlessSpec  # type: ignore

        if SCHEMA_API_AVAILABLE and SchemaBuilder is not None:
            schema = (
                SchemaBuilder()
                .add_dense_vector_field(
                    name=self.VECTOR_FIELD,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                )
                .build()
            )
            self.pc.indexes.create(
                name=self.index_name,
                schema=schema,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT,
                ),
            )
        else:
            # Fallback for older SDK versions that lack SchemaBuilder
            logger.warning(
                "SchemaBuilder not available — creating a legacy dimension-based index. "
                "Upgrade the pinecone package to use the Documents API."
            )
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT,
                ),
            )

        # Poll until the index appears in the list
        for _ in range(60):
            names = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name in names:
                break
            time.sleep(2)
        logger.info(f"Pinecone index '{self.index_name}' created successfully")

    # Storing embedding in Pinecone

    async def store_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> bool:
        """
        Store embeddings in Pinecone with proper metadata handling.
        """
        if not embeddings:
            logger.warning("No embeddings provided to store")
            return True

        try:

            def _sanitise(val):
                """Coerce a value to a JSON-serialisable primitive."""
                if val is None:
                    return ""
                if isinstance(val, datetime):
                    return val.isoformat()
                if isinstance(val, (str, int, float, bool)):
                    return str(val)
                return str(val)

            documents = []
            for emb in embeddings:
                required_fields = {"id", "text", "values", "metadata"}
                missing = required_fields.difference(emb)
                if missing:
                    raise ValueError(
                        f"Embedding record missing required fields: {sorted(missing)}"
                    )

                values = [float(v) for v in emb["values"]]
                if len(values) != self.embedding_dimension:
                    raise ValueError(
                        f"Expected {self.embedding_dimension}-dim vector, got {len(values)}"
                    )

                text_content = str(emb["text"])[:4000]
                meta = dict(emb["metadata"] or {})

                document = {
                    "_id": str(emb["id"]),
                    self.VECTOR_FIELD: values,
                    "chunk_text": text_content,
                    "file_id": _sanitise(meta.get("file_id", "")),
                    "filename": _sanitise(meta.get("filename", "")),
                    "chunk_index": _sanitise(meta.get("chunk_index", 0)),
                    "upload_date": _sanitise(meta.get("upload_date", "")),
                    "custom_metadata": _sanitise(meta.get("custom_metadata", "")),
                }
                documents.append(document)

            if not documents:
                logger.error("No valid documents to store")
                return False

            # Batch upsert via Documents API
            batch_size = 100
            total_stored = 0
            total_batches = -(-len(documents) // batch_size)  # ceiling division

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                try:
                    await asyncio.to_thread(
                        self.index.documents.upsert,
                        documents=batch,
                        namespace=namespace,
                    )
                    total_stored += len(batch)
                    logger.info(
                        "Stored batch %s/%s: %s documents (namespace=%s)",
                        i // batch_size + 1,
                        total_batches,
                        len(batch),
                        namespace,
                    )
                except Exception:
                    logger.exception(
                        "Error storing document batch %s", i // batch_size + 1
                    )
                    raise

            logger.info(
                f"Successfully stored {total_stored}/{len(documents)} documents in Pinecone"
            )
            return total_stored == len(documents)

        except Exception as e:
            logger.error(f"Error storing embeddings in Pinecone: {str(e)}")
            return False

    async def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the index vector query API.
        Queries every namespace (file) and merges results.
        """
        if not query_embedding:
            logger.error("No query embedding provided")
            return []

        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension}-dim query vector, "
                f"got {len(query_embedding)}"
            )

        if top_k < 1:
            logger.error("top_k must be at least 1")
            return []

        try:
            # Discover all namespaces (one per file)
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            namespaces = getattr(stats, "namespaces", None) or {}
            namespace_names = list(namespaces) if namespaces else [""]

            limit = min(top_k, 1000)

            # Fan out a vector query per namespace
            query_tasks = [
                asyncio.to_thread(
                    self.index.documents.search,
                    score_by=[
                        {
                            "type": "dense_vector",
                            "field": self.VECTOR_FIELD,
                            "values": query_embedding,
                        }
                    ],
                    top_k=limit,
                    namespace=ns,
                    include_fields=["*"],
                )
                for ns in namespace_names
            ]
            results = await asyncio.gather(*query_tasks, return_exceptions=True)

            successful = [r for r in results if not isinstance(r, Exception)]
            if not successful:
                first_err = next(r for r in results if isinstance(r, Exception))
                raise RuntimeError(
                    "Pinecone vector search failed for all namespaces"
                ) from first_err

            formatted: List[Dict[str, Any]] = []
            for ns, result in zip(namespace_names, results):
                if isinstance(result, Exception):
                    logger.error("Search failed for namespace %s: %s", ns, result)
                    continue

                matches = getattr(result, "matches", []) or []
                for match in matches:
                    formatted.append(_format_search_match(match))

            formatted.sort(key=lambda r: r["score"], reverse=True)
            formatted = formatted[:top_k]
            logger.info(
                "Found %s similar documents (top scores: %s)",
                len(formatted),
                [r["score"] for r in formatted[:3]],
            )
            return formatted

        except Exception as e:
            logger.exception("Error searching in Pinecone: %s", e)
            raise

    async def delete_by_file_id(self, file_id: str) -> bool:
        """
        Delete all documents for a file by purging its namespace.
        Each file is stored in namespace = file_id, so one call removes everything.
        """
        if not file_id:
            logger.error("No file_id provided for deletion")
            return False

        try:
            await asyncio.to_thread(
                self.index.documents.delete,
                namespace=file_id,
                delete_all=True,
            )
            logger.info(f"Deleted all documents for file '{file_id}' (namespace purge)")
            return True
        except Exception as e:
            logger.error(
                f"Error deleting namespace '{file_id}' from Pinecone: {str(e)}"
            )
            return False

    async def list_files(self) -> List[Dict[str, Any]]:
        """List all files by inspecting index namespaces."""
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            namespaces = getattr(stats, "namespaces", None) or {}

            files_info = []
            if namespaces:
                for ns, ns_stats in namespaces.items():
                    vector_count = getattr(ns_stats, "vector_count", 0)
                    files_info.append(
                        {
                            "file_id": ns,
                            "filename": ns,
                            "upload_date": "",
                            "status": "active",
                            "chunk_count": vector_count,
                        }
                    )
            else:
                total = getattr(stats, "total_vector_count", 0)
                if total > 0:
                    files_info.append(
                        {
                            "file_id": "unknown",
                            "filename": "All documents",
                            "upload_date": "",
                            "status": "active",
                            "chunk_count": total,
                        }
                    )

            logger.info(f"Listed {len(files_info)} file entries from Pinecone")
            return files_info

        except Exception as e:
            logger.error(f"Error listing files from Pinecone: {str(e)}")
            return []

    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific file by querying its namespace."""
        if not file_id:
            logger.error("No file_id provided")
            return None

        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            namespaces = getattr(stats, "namespaces", None) or {}

            if file_id not in namespaces:
                logger.warning(f"No namespace (file) found for file_id '{file_id}'")
                return None

            ns_stats = namespaces[file_id]
            chunk_count = getattr(ns_stats, "vector_count", 0)

            # Fetch one sample document to retrieve filename / upload_date
            dummy = [0.0] * self.embedding_dimension
            sample = await asyncio.to_thread(
                self.index.documents.query,
                vector={self.VECTOR_FIELD: dummy},
                top_k=1,
                namespace=file_id,
                include_values=False,
            )
            matches = getattr(sample, "matches", []) or []
            filename = file_id
            upload_date = ""
            if matches:
                fields = getattr(matches[0], "fields", {}) or {}
                filename = fields.get("filename", file_id)
                upload_date = fields.get("upload_date", "")

            file_info = {
                "file_id": file_id,
                "filename": filename,
                "upload_date": upload_date,
                "status": "active",
                "chunk_count": chunk_count,
            }
            logger.info(f"Retrieved info for file '{file_id}': {chunk_count} chunks")
            return file_info

        except Exception as e:
            logger.error(f"Error getting file info from Pinecone: {str(e)}")
            return None


class VectorDBService:
    """Vector database service — thin facade over PineconeVectorDB."""

    def __init__(self):
        """Initialize the Pinecone vector database service."""
        try:
            self.db = PineconeVectorDB()
            logger.info(
                "VectorDBService initialized with Pinecone Documents API backend"
            )
        except Exception as e:
            logger.error(f"Failed to initialize VectorDBService: {str(e)}")
            raise

    async def store_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> bool:
        """Store embeddings in Pinecone via the Documents API."""
        return await self.db.store_embeddings(embeddings, namespace)

    async def search_similar(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Embed *query_text* and search for similar documents in Pinecone."""
        from services.embeddings_services import get_embeddings_service

        query_embedding = await get_embeddings_service().generate_query_embedding(
            query_text
        )
        return await self.db.search_similar(query_embedding, top_k=top_k)

    async def delete_file(self, file_id: str) -> bool:
        """Delete all documents for a specific file from Pinecone."""
        return await self.db.delete_by_file_id(file_id)

    async def list_files(self) -> List[Dict[str, Any]]:
        """List all files stored in Pinecone."""
        return await self.db.list_files()

    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific file from Pinecone."""
        return await self.db.get_file_info(file_id)

    def get_index_stats(self) -> dict:
        """Get Pinecone index statistics."""
        try:
            stats = self.db.index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", None) or {}
            return {
                "total_vectors": getattr(stats, "total_vector_count", 0),
                "index_fullness": getattr(stats, "index_fullness", 0.0),
                "namespaces": {
                    ns: getattr(ns_stats, "vector_count", 0)
                    for ns, ns_stats in namespaces.items()
                },
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"total_vectors": 0, "index_fullness": 0.0, "namespaces": {}}
