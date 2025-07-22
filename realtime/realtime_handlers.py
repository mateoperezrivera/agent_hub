"""Realtime event handlers for the application"""

import asyncio
import json
import logging
from typing import Any, Optional
import numpy as np
import chainlit as cl
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import (
    AzureRealtimeWebsocket,
    ListenEvents,
)
from openai.types.beta.realtime import ResponseCreateEvent
from semantic_kernel.contents.realtime_events import (
    RealtimeTextEvent,
    RealtimeFunctionResultEvent,
)

from shared.session_manager import SessionManager
from audio.audio_processor import AudioProcessor
from shared.utils import format_error_message, format_transcription_message

logger = logging.getLogger(__name__)


class RealtimeEventHandler:
    """Handles realtime events from Azure OpenAI"""

    def __init__(
        self,
        kernel: Kernel,
        realtime_service: AzureRealtimeWebsocket,
        audio_processor: AudioProcessor,
        function_plugin_map: dict,
    ):
        self.kernel = kernel
        self.realtime_service = realtime_service
        self.audio_processor = audio_processor
        self.function_plugin_map = function_plugin_map

    async def handle_event(
        self, event: Any, chat_history: ChatHistory, track_id: str
    ) -> None:
        """Main event handler that routes to specific handlers"""
        logger.info(
            f"Received service event: {event.event_type} - {event.service_type}"
        )

        if event.service_type == ListenEvents.RESPONSE_OUTPUT_ITEM_DONE:
            await self.handle_response_complete(event, chat_history)

        elif event.service_type == ListenEvents.INPUT_AUDIO_BUFFER_COMMITTED:
            await self.handle_audio_buffer_committed(chat_history)

        elif event.service_type == ListenEvents.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            await self.handle_speech_started()

        elif isinstance(event, RealtimeTextEvent):
            await self.handle_text_event(event)

        elif event.service_type == ListenEvents.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            await self.handle_function_call(event, chat_history)

    async def handle_response_complete(
        self, event: Any, chat_history: ChatHistory
    ) -> None:
        """Handle response completion event"""
        message = SessionManager.get(f"message{event.service_event.item.id}")
        if message:
            await message.send()
            SessionManager.clear_message(event.service_event.item.id)
            chat_history.add_assistant_message(message.content)
        else:
            logger.warning("No message found to send on response done")

    async def handle_audio_buffer_committed(self, chat_history: ChatHistory) -> None:
        """Handle audio buffer committed event"""
        logger.info("Audio buffer committed")
        audio_buffer = SessionManager.get("transcription_buffer")
        SessionManager.set("transcription_buffer", None)

        if audio_buffer:
            message = cl.Message(content="Processing audio...", type="user_message")
            await message.send()
            asyncio.create_task(
                self.process_audio_in_background(audio_buffer, message, chat_history)
            )

    async def handle_speech_started(self) -> None:
        """Handle speech started event"""
        await cl.context.emitter.send_audio_interrupt()
        SessionManager.generate_track_id()

    async def handle_text_event(self, event: RealtimeTextEvent) -> None:
        """Handle text streaming event"""
        if hasattr(event.text, "text"):
            text_content = event.text.text
            message = SessionManager.get_or_create_message(event.service_event.item_id)
            await message.stream_token(text_content)

    async def handle_function_call(self, event: Any, chat_history: ChatHistory) -> None:
        """Handle function call event"""
        try:
            arguments = event.service_event.arguments
            function_name = event.service_event.name
            plugin_name = self.function_plugin_map.get(function_name)

            if not plugin_name:
                logger.error(f"Could not find plugin for function '{function_name}'")
                return

            # Execute function
            function = self.kernel.get_function(
                plugin_name=plugin_name, function_name=function_name
            )
            arguments_dict = json.loads(arguments)
            kernel_args = KernelArguments(**arguments_dict)
            result = await self.kernel.invoke(
                function=function, arguments=kernel_args, chat_history=chat_history
            )

            # Send result back
            call = FunctionResultContent(
                call_id=event.service_event.call_id,
                function_name=function_name,
                plugin_name=plugin_name,
                result=str(result),
            )
            new_event = RealtimeFunctionResultEvent(function_result=call)
            new_event.function_result.metadata["call_id"] = event.service_event.call_id

            await self.realtime_service.send(new_event)
            await self.realtime_service._send(
                ResponseCreateEvent(type="response.create")
            )
            chat_history.add_assistant_message(str(result))

        except Exception as e:
            logger.error(f"Error handling function call: {e}", exc_info=True)

    async def process_audio_in_background(
        self, audio_buffer: bytes, message: cl.Message, chat_history: ChatHistory
    ) -> None:
        """Process audio buffer in the background without blocking event processing"""
        try:
            result = await self.audio_processor.transcribe_audio(audio_buffer)

            if result["success"]:
                message.content = format_transcription_message(result["transcript"])
            else:
                message.content = format_error_message(
                    Exception(result.get("error", "Unknown error")),
                    "Transcription failed",
                )

            await message.update()
            chat_history.add_user_message(message.content)

        except Exception as e:
            logger.error(f"Error processing audio in background: {e}", exc_info=True)
            try:
                message.content = format_error_message(e, "Error processing audio")
                await message.update()
            except:
                pass


async def audio_output_callback(audio_array: np.ndarray) -> None:
    """Callback function to handle audio output from the realtime service"""
    try:
        from config import AppConfig

        config = AppConfig()

        track_id = SessionManager.get("track_id", "default_track")
        audio_bytes = audio_array.tobytes()

        await cl.context.emitter.send_audio_chunk(
            cl.OutputAudioChunk(
                mimeType=config.audio.output_format,
                data=audio_bytes,
                track=track_id,
            )
        )

        logger.debug(f"Sent audio chunk of size {len(audio_array)} samples")

    except Exception as e:
        logger.error(f"Error in audio output callback: {e}")
