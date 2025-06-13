"""Audio session management module"""

import asyncio
import logging
import chainlit as cl
from semantic_kernel.connectors.ai.open_ai import AzureRealtimeExecutionSettings
from semantic_kernel.contents.realtime_events import RealtimeAudioEvent

from shared.session_manager import SessionManager
from app_context import app_context  # Now from root

logger = logging.getLogger(__name__)


class AudioSessionManager:
    """Manages audio session lifecycle and processing"""

    @staticmethod
    async def initialize_session():
        """Initialize audio session when user starts recording"""
        try:
            chat_history = SessionManager.get("chat_history")
            track_id = SessionManager.generate_track_id()

            settings = AzureRealtimeExecutionSettings(
                modalities=["audio", "text"],
                voice=app_context.config.audio.voice,
                input_audio_format=app_context.config.audio.input_format,
                output_audio_format=app_context.config.audio.output_format,
                chat_history=chat_history,
                input_audio_noise_reduction={"type": "near_field"},
                tool_choice="auto",
                tools=app_context.openai_tools,
            )

            await app_context.realtime_service.create_session(settings=settings)

            # Start background task to listen for events
            background_task = asyncio.create_task(
                AudioSessionManager._listen_for_events()
            )
            SessionManager.set("background_task", background_task)

            logger.info("Audio session initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing audio session: {e}", exc_info=True)
            await cl.Message(content=f"Error starting audio session: {str(e)}").send()

    @staticmethod
    async def _listen_for_events():
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

    @staticmethod
    async def process_chunk(chunk: cl.InputAudioChunk):
        """Process incoming audio chunks from the user"""
        try:
            # Update transcription buffer
            transcription_buffer = SessionManager.get(
                "transcription_buffer", bytearray()
            )
            transcription_buffer.extend(chunk.data)
            SessionManager.set("transcription_buffer", transcription_buffer)

            # Convert and send audio chunk
            audio_content = app_context.audio_processor.encode_audio_chunk(chunk.data)

            await app_context.realtime_service.send(
                RealtimeAudioEvent(
                    service_type="input_audio_buffer.append", audio=audio_content
                )
            )

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    @staticmethod
    async def finalize_session():
        """Finalize audio session when user stops recording"""
        try:
            await SessionManager.cancel_background_task()
            await app_context.realtime_service.close_session()
            logger.info("Audio session finalized successfully")

        except Exception as e:
            logger.error(f"Error finalizing audio session: {e}", exc_info=True)
