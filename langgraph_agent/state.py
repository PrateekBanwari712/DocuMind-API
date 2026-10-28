"""
State management for LangGraph RAG agent
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass


class AgentState(TypedDict):
    """
    State structure for the RAG agent
    """

    # User query and conversation
    query: str
    session_id: str
    conversation_history: List[Dict[str, str]]

    # Retrieved documents and context
    retrieved_documents: List[Dict[str, Any]]
    context: str
    relevance_scores: List[float]

    # Generation and response
    generated_response: str
    final_answer: str
    sources: List[str]

    # Agent state and metadata
    current_step: str
    confidence_score: Optional[float]
    error_message: Optional[str]
    tool_calls: List[Dict[str, Any]]

    # Search and retrieval parameters
    search_query: str
    top_k: int
    similarity_threshold: float


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""

    user_message: str
    assistant_message: str
    timestamp: str
    sources: List[str]
    confidence_score: Optional[float] = None


class StateManager:
    """Manages agent state and  conversation history"""

    def __init__(self):
        self.sessions: Dict[str, List[ConversationTurn]] = {}

    def create_initial_state(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.05,
    ) -> AgentState:
        """Create initial state for a new query"""
        return AgentState(
            query=query,
            session_id=session_id,
            conversation_history=self.get_conversation_history(session_id),
            retrieved_documents=[],
            context="",
            relevance_scores=[],
            generated_response="",
            final_answer="",
            sources=[],
            current_step="retrieval",
            confidence_score=None,
            error_message=None,
            tool_calls=[],
            search_query="",
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            return []

        history = []
        for turn in self.sessions[session_id]:
            history.extend(
                [
                    {"role": "user", "content": turn.user_message},
                    {"role": "assistant", "content": turn.assistant_message},
                ]
            )

        return history

    def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        sources: List[str],
        confidence_score: Optional[float] = None,
    ):
        """Add a new conversation turn to the session"""
        from datetime import datetime

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=datetime.now().isoformat(),
            sources=sources,
            confidence_score=confidence_score,
        )

        self.sessions[session_id].append(turn)

        # Keep only last N turns to prevent memory issues
        max_turns = 20
        if len(self.sessions[session_id]) > max_turns:
            self.sessions[session_id] = self.sessions[session_id][-max_turns:]

    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def update_state(self, state: AgentState, **updates) -> AgentState:
        """Update state with new values"""
        for key, value in updates.items():
            if key in state:
                state[key] = value
        return state

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session"""
        if session_id not in self.sessions:
            return {"session_id": session_id, "turns": 0, "last_activity": None}

        turns = self.sessions[session_id]
        return {
            "session_id": session_id,
            "turns": len(turns),
            "last_activity": turns[-1].timestamp if turns else None,
            "average_confidence": (
                (
                    sum(
                        turn.confidence_score for turn in turns if turn.confidence_score
                    )
                    / len([turn for turn in turns if turn.confidence_score])
                )
                if any(turn.confidence_score for turn in turns)
                else None
            ),
        }


# Global state manager instance
state_manager = StateManager()
