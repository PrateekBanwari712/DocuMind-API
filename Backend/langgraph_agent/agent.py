"""
Main langGraph agent for RAG functionality
"""

import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from google import genai
except ImportError:
    StateGraph = None
    END = None
    MemorySaver = None
    genai = None


from langgraph_agent.state import AgentState, state_manager
from langgraph_agent.tools import (
    document_retriever,
    context_summarizer,
    relevance_checker,
)
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGAgent:
    """Main RAG agent using LangGraph for orchestration"""

    def __init__(self):
        self._initialize_llm()
        self.graph = self._build_graph()

    def _initialize_llm(self):
        """Initialize the Gemini Client"""
        if genai is None:
            raise ImportError(
                "Gemini is required. Install with pip install -U google-genai "
            )

        if not settings.GOOGLE_GENERATIVE_AI_API_KEY :
            raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY is required")

        try:
            self.client = genai.Client(api_key=settings.GOOGLE_GENERATIVE_AI_API_KEY)
            logger.info(
                f"Gemini client initialized successfully with model: {settings.LLM_MODEL}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise

    def _build_graph(self):
        """Build the LangGraph workflow"""
        if StateGraph is None:
            logger.warning("LangGraph not available, using simplified workflow")
            return None

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("check_relevance", self._check_relevance)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("handle_no_context", self._handle_no_context)

        # Add edges
        workflow.set_entry_point("retrieve_documents")

        workflow.add_conditional_edges(
            "retrieve_documents",
            self._should_check_relevance,
            {"check_relevance": "check_relevance", "no_context": "handle_no_context"},
        )

        workflow.add_conditional_edges(
            "check_relevance",
            self._should_generate_response,
            {"generate": "generate_response", "no_context": "handle_no_context"},
        )

        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_no_context", END)

        # Add memory
        memory = MemorySaver() if MemorySaver else None
        return workflow.compile(checkpointer=memory)

    async def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from the vector database"""
        logger.info(f"Retrieving documents for query: {state['query']}")

        try:
            # using the document retrieval tool
            result = await document_retriever._arun(
                query=state["query"],
                top_k=state["top_k"],
                similarity_threshold=state["similarity_threshold"],
            )

            result_data = json.loads(result)

            if "error" in result_data:
                state["error_message"] = result_data["error"]
                state["retrieved_documents"] = []
            else:
                state["retrieved_documents"] = result_data["documents"]
                state["relevance_scores"] = [
                    doc.get("similarity_score", 0) for doc in result_data["documents"]
                ]

            state["current_step"] = "retrieval_complete"
            logger.info(f"Retrieved {len(state['retrieved_documents'])} documents")

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            state["error_message"] = str(e)
            state["retrieved_documents"] = []

        return state

    async def _check_relevance(self, state: AgentState) -> AgentState:
        """Check relevance of retrieved documents"""
        logger.info("Checking relevance of retrieved documents")

        try:
            # Prepare context from retrieved documents
            context_result = context_summarizer._run(
                documents=state["retrieved_documents"], query=state["query"]
            )

            context_data = json.loads(context_result)
            state["context"] = context_data.get("context", "")
            state["sources"] = context_data.get("sources", [])

            # Check relevance
            if state["context"]:
                relevance_result = relevance_checker._run(
                    query=state["query"], context=state["context"]
                )

                relevance_data = json.loads(relevance_result)
                state["confidence_score"] = relevance_data.get("score", 0)

            state["current_step"] = "relevance_checked"

        except Exception as e:
            logger.error(f"Error checking relevance: {str(e)}")
            state["error_message"] = str(e)
            state["context"] = ""

        return state

    async def _generate_response(self, state: AgentState) -> AgentState:
        """Generate response using the LLM"""
        logger.info("Generating response")

        try:
            # Prepare the prompt
            conversation_context = ""
            if state["conversation_history"]:
                # Include recent conversation history
                recent_history = state["conversation_history"][-6:]
                for msg in recent_history:
                    role = msg["role"].capitalize()
                    conversation_context += f"{role}: {msg['content']}\n"

            prompt = f"""You are a helpful AI assistant. Answer the user's question ONLY using the information provided in the context below.
            
IMPORTANT RULES:
1. ONLY use information from the context provided below.
2. DO NOT use any external knowledge or make assumptions.
3. If the context doesn't contain the answer, say that the database doesn't contain the information.
4. DO NOT cite sources or mention document names.
5. Be direct and concise
6. Answer in a friendly, professional tone.

{f"Previous conversation:{conversation_context}" if conversation_context else ""}

Context from knowledge base:
{state['context']}

User question: {state['query']}

Answer based ONLY on the context above:"""

            # Generate response using Gemini
            response = await self._generate_with_llm(prompt)

            state["generated_response"] = response
            state["final_answer"] = response
            state["current_step"] = "response_generated"

            # Calculate confidence based on context relevance
            if state["confidence_score"] is None:
                state["confidence_score"] = 0.8 if state["context"] else 0.3

            logger.info("Response generated successfully")

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["error_message"] = str(e)
            state["final_answer"] = f"I apologize, but I encountered an error: {str(e)}"

        return state

    async def _handle_no_context(self, state: AgentState) -> AgentState:
        """Handle cases where no relevant context is found"""
        logger.info("Handling query with no relevant context")

        if state.get("error_message"):
            state["final_answer"] = (
                "I couldn't search the knowledge base right now. Please try again later."
            )
            state["confidence_score"] = 0.0
            state["current_step"] = "retrieval_error"
            return state

        # Direct friendly message - no LLM call needed
        state["final_answer"] = (
            "I apologize, but I don't have any information about this topic in our database. "
            "Please try asking about topics related to the documents that have been uploaded to the system."
        )
        state["confidence_score"] = 0.0
        state["current_step"] = "no_context_handled"

        return state

    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate a response using the configured Gemini model."""
        try:
            import asyncio

            logger.info("Invoking Gemini with prompt length: %s chars", len(prompt))
            completion = await asyncio.to_thread(
                self.client.models.generate_content,
                model=settings.LLM_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                    "top_p": 1,
                },
            )
            response_text = (completion.text or "").strip()
            if not response_text:
                logger.warning("Gemini returned an empty response")
                return "I couldn't generate a response. Please try rephrasing your question."

            logger.info("Gemini response length: %s chars", len(response_text))
            return response_text

        except Exception as e:
            logger.exception("Error generating response with Gemini: %s", e)
            return "I encountered an error while generating a response. Please try again."

    def _should_check_relevance(self, state: AgentState) -> str:
        """Decide whether to check relevance or handle no context"""
        if state["retrieved_documents"]:
            return "check_relevance"
        else:
            return "no_context"

    def _should_generate_response(self, state: AgentState) -> str:
        """Decide whether to generate response or handle no context"""    
        if state["context"].strip():
            return "generate"
        else:
            return "no_context"

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Process a user query and return response"""
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            logger.info(f"Processing query for session {session_id}: {query}")

            # Create initial state
            initial_state = state_manager.create_initial_state(
                query=query,
                session_id=session_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            # Process with LangGraph if available
            if self.graph:
                config = {"configurable": {"thread_id": session_id}}
                final_state = await self.graph.ainvoke(initial_state, config)
            else:
                # Fallback to simple workflow
                final_state = await self._simple_workflow(initial_state)

            # Add to conversation history
            state_manager.add_conversation_turn(
                session_id=session_id,
                user_message=query,
                assistant_message=final_state["final_answer"],
                sources=final_state.get("sources", []),
                confidence_score=final_state.get("confidence_score"),
            )

            return {
                "answer": final_state["final_answer"],
                "session_id": session_id,
                "sources": final_state.get("sources", []),
                "confidence_score": final_state.get("confidence_score"),
                "error": final_state.get("error_message"),
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "session_id": session_id or str(uuid.uuid4()),
                "sources": [],
                "confidence_score": 0.0,
                "error": str(e),
            }

    async def _simple_workflow(self, state: AgentState) -> AgentState:
        """Simple workflow fallback when LangGraph is not available"""
        logger.info("Using simple workflow (LangGraph not available)")

        # Retrieve documents
        state = await self._retrieve_documents(state)

        # Check relevance and generate response
        if state["retrieved_documents"]:
            state = await self._check_relevance(state)
            if state["context"]:
                state = await self._generate_response(state)
            else:
                state = await self._handle_no_context(state)

        else:
            state = await self._handle_no_context(state)

        return state

    async def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        history = state_manager.get_conversation_history(session_id)
        return history

    async def clear_session(self, session_id: str):
        """Clear chat history for a session"""
        state_manager.clear_session(session_id)
