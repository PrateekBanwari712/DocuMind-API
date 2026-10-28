"""
Tools for the LangGraph RAG agent
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback if Langchain is not available
    BaseTool = None
    BaseModel = object
    Field = lambda **kwargs: None

from services.vectordb_service import VectorDBService
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetrieverInput(BaseModel):
    """Input schema for document retriever tool"""

    query: str = Field(description="The search query to find relevant documents")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(
        default=0.05, description="Minimum similarity threshold"
    )


class DocumentRetrieverTool(BaseTool if BaseTool else object):
    """Tool for retrieving relevant documents from the vector database"""

    name: str = "document_retriever"
    description: str = (
        "Retrieve relevant documents from the knowledge base based on a query"
    )
    args_schema: type = DocumentRetrieverInput
    vectorbd_service: Any = None
    embeddings_service: Any = None

    def __init__(self):
        super().__init__() if BaseTool else None
        self.vectorbd_service = None

    async def _arun(
        self, query: str, top_k: int = 5, similarity_threshold: float = 0.05
    ) -> str:
        """Async implementation of the tool"""

        try:
            if self.vectorbd_service is None:
                self.vectorbd_service = VectorDBService()

            # Pinecone generates query embeddings from the indexed text field.
            similar_docs = await self.vectorbd_service.search_similar(
                query_text=query, top_k=top_k
            )

            best_score = max(
                (doc.get("score", 0.0) for doc in similar_docs), default=None
            )
            logger.info(
                "Pinecone returned %s candidates; best score: %s",
                len(similar_docs),
                best_score,
            )

            # Filter by similarity threshold if using distance-based results
            filtered_docs = []
            logger.info(
                f"Filtering {len(similar_docs)} documents with threshold {similarity_threshold}"
            )
            logger.info(
                f"DEBUG: similarity_threshold type: {type(similarity_threshold)}, value: {similarity_threshold}"
            )

            for doc in similar_docs:
                # Convert distance to similarity score if needed (depends on vector DB)
                if "score" in doc:
                    similarity = doc["score"]
                elif "distance" in doc:
                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1 - doc["distance"]
                else:
                    similarity = 1.0  # Default if no score available

                logger.info(
                    f"DEBUG: Document score: {similarity} (type: {type(similarity)}), threshold: {similarity_threshold}, passes: {similarity >= similarity_threshold}"
                )

                if similarity >= similarity_threshold:
                    doc["similarity_score"] = similarity
                    filtered_docs.append(doc)

            logger.info(
                f"Retrieved {len(filtered_docs)} relevant documents for query: {query}"
            )

            return json.dumps(
                {
                    "documents": filtered_docs,
                    "count": len(filtered_docs),
                    "query": query,
                }
            )

        except Exception as e:
            logger.error(f"Error in document retriever tool: {str(e)}")
            return json.dumps({"error": str(e), "documents": [], "count": 0})

    def _run(
        self, query: str, top_k: int = 5, similarity_threshold: float = 0.05
    ) -> str:
        """Sync implementation (fallback)"""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._arun(query, top_k, similarity_threshold)
            )
        finally:
            loop.close()


class ContextSummarizerInput(BaseModel):
    """Input schema for context summarizer tool"""

    documents: List[Dict[str, Any]] = Field(description="List of retrieved documents")
    query: str = Field(description="Original user query")
    max_context_length: int = Field(default=4000, description="Maximum context length")


class ContextSummarizerTool(BaseTool if BaseTool else object):
    """Tool for summarizing and preparing content from retrieved documents"""

    name: str = "context_summarizer"
    description: str = "Summarize and prepare context from retrieved documents"
    args_schema: type = ContextSummarizerInput

    def _run(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        max_context_length: int = 4000,
    ) -> str:
        """Create summarized context from documents"""
        try:
            if not documents:
                return json.dumps(
                    {
                        "context": "no relevant documents found.",
                        "sources": [],
                        "total_chars": 0,
                    }
                )

            # Sort documents by similrity score if available
            sorted_docs = sorted(
                documents, key=lambda x: x.get("similarity_score", 0), reverse=True
            )

            context_parts = []
            sources = []
            total_chars = 0

            for i, doc in enumerate(sorted_docs):
                doc_text = doc.get("text", "")
                doc_source = doc.get("metadata", {}).get(
                    "filename", f"Document {i + 1}"
                )

                # Check if adding this document would exceed the limit
                if total_chars + len(doc_text) > max_context_length and context_parts:
                    break

                context_parts.append(f"[Source: {doc_source}\n{doc_text}]")
                sources.append(doc_source)
                total_chars += len(doc_text)

            context = "\n\n---\n\n".join(context_parts)

            logger.info(
                f"Created context from {len(context_parts)} documents ({total_chars} chars)"
            )

            return json.dumps(
                {
                    "context": context,
                    "sources": list(set(sources)),  # Remove duplicates
                    "total_chars": total_chars,
                    "documents_used": len(context_parts),
                }
            )

        except Exception as e:
            logger.error(f"Error in context summarizer tool: {str(e)}")
            return json.dumps(
                {"error": str(e), "context": "", "sources": [], "total_chars": 0}
            )

    async def _arun(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        max_context_length: int = 4000,
    ) -> str:
        """Async implementation"""
        return self._run(documents, query, max_context_length)


class RelevanceCheckerInput(BaseModel):
    """Input schema for relevance checker tool"""

    query: str = Field(description="User query")
    context: str = Field(description="Retrieved context")
    threshold: float = Field(default=0.5, description="Relevance threshold")


class RelevanceCheckerTool(BaseTool if BaseTool else object):
    """Tool for checking relevance of retrieved context to user query"""

    name: str = "relevance_checker"
    description: str = "Check if retrieved context is relevant to the user query"
    args_schema: type = RelevanceCheckerInput

    def _run(self, query: str, context: str, threshold: float = 0.5) -> str:
        """Check relevance of context to query"""
        try:
            # Simple keyword based relevance check
            # In a production system, you might use a more sophisticated approach
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())

            if not query_words:
                return json.dumps(
                    {"relevant": False, "score": 0.0, "reason": "Empty query"}
                )

            if not context_words:
                return json.dumps(
                    {"relevant": False, "score": 0.0, "reason": "Empty context"}
                )

            # Calculate word overlap
            overlap = len(query_words.intersection(context_words))
            relevance_score = overlap / len(query_words)

            is_relevant = relevance_score >= threshold

            logger.info(
                f"Relevance check: score ={relevance_score:.2f}, relevant={is_relevant}"
            )

            return json.dumps(
                {
                    "relevant": is_relevant,
                    "score": relevance_score,
                    "reason": f"Word overlap score: {relevance_score:.2f}",
                    "threshold": threshold,
                }
            )

        except Exception as e:
            logger.error((f"Error in relevance checker tool: {str(e)}"))
            return json.dumps({"error": str(e), "relevant": False, "score": 0.0})

    async def _arun(self, query: str, context: str, threshold: float = 0.5) -> str:
        """Async implementation"""
        return self._run(query, context, threshold)


# Tool instances
document_retriever = DocumentRetrieverTool()
context_summarizer = ContextSummarizerTool()
relevance_checker = RelevanceCheckerTool()

# Tool list for LangGraph
AVAILABLE_TOOLS = [document_retriever, context_summarizer, relevance_checker]


def get_tool_by_name(name: str):
    """Get a tool by its name"""
    for tool in AVAILABLE_TOOLS:
        if tool.name == name:
            return tool

    return None
