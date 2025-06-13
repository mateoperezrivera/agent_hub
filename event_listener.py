"""Event listener module for background processing"""

import asyncio
import logging

from session_manager import SessionManager
from app_context import app_context

logger = logging.getLogger(__name__)


class EventListener:
    """Handles background event listening from the realtime service"""

    @staticmethod
    async def listen_for_events():
        """Background task to listen for events from the realtime service"""
        try:
            chat_history = SessionManager.get("chat_history")
            track_id = SessionManager.get("track_id")

            async for event in app_context.realtime_service.receive():
                try:
                    await app_context.event_handler.handle_event(
                        event, chat_history, track_id
                    )
                except Exception as e:
                    logger.error(f"Error handling event: {e}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Event listener cancelled")
        except Exception as e:
            logger.error(f"Error in event listener: {e}", exc_info=True)
