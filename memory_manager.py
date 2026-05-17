from typing import Dict, List
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class WindowedMemory:
    """
    A windowed conversation memory that retains the last k exchanges.
    Each exchange = 1 human message + 1 AI message.

    Drop-in replacement for the removed ConversationBufferWindowMemory.
    Provides the same interface used by chain.py and other modules.
    """

    def __init__(self, k=10, memory_key="chat_history",
                 return_messages=True, output_key="output"):
        self.k = k
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.output_key = output_key
        self._history = InMemoryChatMessageHistory()

    @property
    def buffer_as_messages(self):
        """Return the last k exchanges (2*k messages) from history."""
        messages = self._history.messages
        max_messages = self.k * 2
        if len(messages) > max_messages:
            return messages[-max_messages:]
        return list(messages)

    def save_context(self, inputs, outputs):
        """
        Save a single conversation turn (user input + agent output).

        Args:
            inputs: Dict with 'input' key containing the user message.
            outputs: Dict with 'output' key containing the agent response.
        """
        user_text = inputs.get("input", "")
        agent_text = outputs.get(self.output_key, outputs.get("output", ""))
        self._history.add_message(HumanMessage(content=user_text))
        self._history.add_message(AIMessage(content=agent_text))

        # Trim to keep only last k exchanges
        messages = self._history.messages
        max_messages = self.k * 2
        if len(messages) > max_messages:
            trimmed = messages[-max_messages:]
            self._history.clear()
            for msg in trimmed:
                self._history.add_message(msg)

    def clear(self):
        """Clear all messages from this memory."""
        self._history.clear()


class MemoryManager:
    """
    Manages multiple conversation memory instances keyed by session_id.
    Each session retains the last k turns of conversation.
    """

    def __init__(self, k=10):
        """
        Args:
            k: Number of conversation turns to retain per session.
               OBJECTIVE 2 requires at least 10.
        """
        self.k = k
        self._sessions: Dict[str, WindowedMemory] = {}

    def get_memory(self, session_id):
        """
        Retrieve or create memory for a given session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            WindowedMemory instance for this session.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = WindowedMemory(
                k=self.k,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
            )
        return self._sessions[session_id]

    def get_chat_history(self, session_id):
        """
        Get the chat history messages for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of message objects from memory.
        """
        memory = self.get_memory(session_id)
        return memory.buffer_as_messages

    def reset_session(self, session_id):
        """
        Clear memory for a specific session.

        Args:
            session_id: Session to reset.

        Returns:
            True if session existed and was cleared, False otherwise.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            return True
        return False

    def reset_all(self):
        """Clear all sessions."""
        for memory in self._sessions.values():
            memory.clear()
        self._sessions.clear()

    @property
    def active_sessions(self):
        """Number of active memory sessions."""
        return len(self._sessions)

    @property
    def session_ids(self):
        """List of all active session IDs."""
        return list(self._sessions.keys())


# --- Global singleton instance ---
# OBJECTIVE 2: Single memory manager shared across the application
memory_manager = MemoryManager(k=10)
