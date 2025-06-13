"""Message handler for processing user messages with fallback to chat completion"""

import logging
import mimetypes
import aiofiles
import chainlit as cl
from typing import Optional
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.realtime_events import RealtimeTextEvent
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from openai.types.beta.realtime import ResponseCreateEvent
from markitdown import MarkItDown

from shared.session_manager import SessionManager

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handles message processing with realtime and chat completion fallback"""

    def __init__(self, app_context):
        self.app_context = app_context
        self.markdown = MarkItDown()

    async def handle_message(self, message: cl.Message) -> None:
        """Process message using realtime if available, otherwise use chat completion"""
        try:
            # Check if realtime session is active
            if await self._is_realtime_active():
                await self._handle_realtime_message(message)
            else:
                await self._handle_chat_completion_message(message)
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await cl.Message(content=f"Error processing message: {str(e)}").send()

    async def _is_realtime_active(self) -> bool:
        """Check if realtime session is currently active"""
        try:
            # Check if realtime service exists and has an active session
            if self.app_context.realtime_service:
                # Check if there's an active background task (indicates active session)
                background_task = SessionManager.get("background_task")
                return background_task is not None and not background_task.done()
            return False
        except Exception:
            return False

    async def _handle_realtime_message(self, message: cl.Message) -> None:
        """Handle message using realtime service"""
        logger.info("Processing message with realtime service")

        # Send text to realtime service
        text = TextContent(text=message.content)
        await self.app_context.realtime_service.send(RealtimeTextEvent(text=text))
        await self.app_context.realtime_service._send(
            ResponseCreateEvent(type="response.create")
        )

        # Interrupt any ongoing audio and generate new track
        await cl.context.emitter.send_audio_interrupt()
        SessionManager.generate_track_id()

        # Add to chat history
        chat_history = SessionManager.get("chat_history")
        if message.elements:
            await self._add_message_with_elements(message, chat_history)
        else:
            chat_history.add_user_message(message.content)

    async def _handle_chat_completion_message(self, message: cl.Message) -> None:
        """Handle message using chat completion service"""
        logger.info("Processing message with chat completion service")

        chat_history = SessionManager.get("chat_history")

        # Process message with elements if any
        if message.elements:
            await self._add_message_with_elements(message, chat_history)
        else:
            chat_history.add_user_message(message.content)

        # Create response message
        answer = cl.Message(content="")

        # Stream response
        async for chunk in self.app_context.chat_completion_service.get_streaming_chat_message_contents(
            chat_history=chat_history,
            kernel=self.app_context.kernel,
            settings=OpenAIChatPromptExecutionSettings()
        ):
            for msg in chunk:
                if msg.content:
                    await answer.stream_token(msg.content)

        # Add assistant response to history
        chat_history.add_assistant_message(answer.content)

        # Send final message
        await answer.send()

    async def _add_message_with_elements(
        self, message: cl.Message, chat_history
    ) -> None:
        """Process message with elements (images, files, etc.)"""
        items = []

        for element in message.elements:
            if element.type == "image":
                # Handle image elements
                if hasattr(element, "url") and element.url:
                    # Remote image with URL
                    mime_type = (
                        getattr(element, "mime", None)
                        or mimetypes.guess_type(element.url)[0]
                        or "image/png"
                    )
                    items.append(ImageContent(uri=element.url, mime_type=mime_type))
                elif hasattr(element, "path") and element.path:
                    # Local image file
                    mime_type = (
                        getattr(element, "mime", None)
                        or mimetypes.guess_type(element.path)[0]
                        or "image/png"
                    )

                    # Read image file
                    async with aiofiles.open(element.path, "rb") as f:
                        image_data = await f.read()

                    # Create ImageContent with binary data
                    items.append(
                        ImageContent(
                            data=image_data, mime_type=mime_type, data_format="base64"
                        )
                    )
            else:
                try:
                    # Handle text/document elements
                    if hasattr(element, "url") and element.url:
                        # Remote text with URL
                        converted_text = self.markdown.convert(source=element.url).markdown
                        items.append(TextContent(text=converted_text))
                    elif hasattr(element, "path") and element.path:
                        # Local text with path
                        converted_text = self.markdown.convert(source=element.path).markdown
                        items.append(TextContent(text=converted_text))
                    elif hasattr(element, "content") and element.content:
                        items.append(TextContent(text=element.content))
                except Exception as e:
                    logger.error(
                        f"Error processing element of type {element.type}: {e}"
                    )

        # Add the message text itself
        items.append(TextContent(text=message.content))

        # Add complete message to history
        chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                items=items,
            )
        )

