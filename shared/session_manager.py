import uuid
from typing import Optional, Any
import chainlit as cl
from semantic_kernel.contents.chat_history import ChatHistory
import asyncio


class SessionManager:
    """Manages user session data and lifecycle"""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from user session"""
        item = cl.user_session.get(key)
        if item is None:
            return default
        return item

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in user session"""
        cl.user_session.set(key, value)

    @staticmethod
    def init_chat_session() -> None:
        """Initialize a new chat session"""
        SessionManager.set("chat_history", ChatHistory())
        SessionManager.set("background_task", None)
        SessionManager.set("transcription_buffer", None)

    @staticmethod
    def generate_track_id() -> str:
        """Generate a new audio track ID"""
        track_id = f"audio_track_{uuid.uuid4()}"
        SessionManager.set("track_id", track_id)
        return track_id

    @staticmethod
    def get_or_create_message(item_id: str) -> cl.Message:
        """Get existing message or create new one for an item ID"""
        message_key = f"message{item_id}"
        message = SessionManager.get(message_key)
        if not message:
            message = cl.Message(content="")
            SessionManager.set(message_key, message)
        return message

    @staticmethod
    def clear_message(item_id: str) -> None:
        """Clear message from session"""
        SessionManager.set(f"message{item_id}", None)

    @staticmethod
    async def cancel_background_task() -> None:
        """Cancel background task if running"""
        background_task: Optional[asyncio.Task] = SessionManager.get("background_task")
        if background_task and not background_task.done():
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
