import base64
import logging
import tempfile
import wave
import os
import numpy as np
from typing import Optional, Dict, Any
from semantic_kernel.contents.audio_content import AudioContent
from semantic_kernel.connectors.ai.open_ai import (
    AzureAudioToText,
    OpenAIAudioToTextExecutionSettings,
)
import asyncio

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing and transcription"""

    def __init__(self, audio_to_text_service: AzureAudioToText, config):
        self.audio_to_text_service = audio_to_text_service
        self.config = config

    @staticmethod
    def encode_audio_chunk(data: bytes) -> AudioContent:
        """Convert audio chunk to base64 AudioContent"""
        audio_data = base64.b64encode(data).decode("utf-8")
        return AudioContent(data=audio_data, data_format="base64")

    async def transcribe_audio(self, audio_buffer: bytes) -> Dict[str, Any]:
        """Transcribe audio buffer to text"""
        result = {"success": False, "transcript": None, "error": None}

        temp_file_path = None

        try:
            if not audio_buffer:
                result["error"] = "No audio data available"
                return result

            # Create temporary WAV file
            temp_file_path = await self._create_wav_file(audio_buffer)

            # Create AudioContent from file
            audio_content = AudioContent.from_audio_file(path=temp_file_path)

            # Transcribe with timeout
            settings = OpenAIAudioToTextExecutionSettings(
                language=self.config.audio.transcription_language
            )

            text_content = await asyncio.wait_for(
                self.audio_to_text_service.get_text_content(
                    audio_content, settings=settings
                ),
                timeout=self.config.audio.transcription_timeout,
            )

            if text_content and hasattr(text_content, "text"):
                result["success"] = True
                result["transcript"] = text_content.text
                logger.info(
                    f"Transcription successful: {result['transcript'][:100]}..."
                )
            else:
                result["error"] = "Could not transcribe speech"

        except asyncio.TimeoutError:
            result["error"] = "Speech recognition timed out"
            logger.error("Transcription request timed out")
        except Exception as e:
            result["error"] = f"Speech recognition error: {str(e)}"
            logger.error(f"Speech-to-text error: {str(e)}", exc_info=True)
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")

        return result

    async def _create_wav_file(self, audio_buffer: bytes) -> str:
        """Create WAV file from PCM16 audio buffer"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name

            with wave.open(temp_file_path, "wb") as wav_file:
                wav_file.setnchannels(self.config.audio.channels)
                wav_file.setsampwidth(2)  # 2 bytes for PCM16
                wav_file.setframerate(self.config.audio.sample_rate)

                if isinstance(audio_buffer, (bytes, bytearray)):
                    wav_file.writeframes(bytes(audio_buffer))
                else:
                    audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
                    wav_file.writeframes(audio_array.tobytes())

            logger.info(f"Created WAV file: {temp_file_path}")
            return temp_file_path
