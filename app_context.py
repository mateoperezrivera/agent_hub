"""Application context and initialization module"""

import logging
from typing import Optional
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureRealtimeWebsocket,
    AzureChatCompletion,
    AzureAudioToText,
)

from config import AppConfig
from audio.audio_processor import AudioProcessor
from realtime.realtime_handlers import RealtimeEventHandler
from shared.utils import convert_to_openai_function_format
from plugins.image_generation_plugin import ImageGenerationPlugin
from plugins.weather_plugin import WeatherPlugin
from shared.ai_service_factory import AIServiceFactory, ServiceType
from shared.message_handler import MessageHandler

logger = logging.getLogger(__name__)


class AppContext:
    """Application context for shared resources"""

    def __init__(self):
        self.kernel: Optional[sk.Kernel] = None
        self.realtime_service: Optional[AzureRealtimeWebsocket] = None
        self.chat_completion_service: Optional[AzureChatCompletion] = None
        self.audio_to_text_service: Optional[AzureAudioToText] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.event_handler: Optional[RealtimeEventHandler] = None
        self.openai_tools: list = []
        self.function_plugin_map: dict = {}
        self.config = AppConfig()
        self.service_factory = AIServiceFactory(self.config)
        self.message_handler: Optional[MessageHandler] = None

    async def initialize(self):
        """Initialize all services and components"""
        try:
            # Initialize Semantic Kernel
            self.kernel = sk.Kernel()

            # Add plugins
            weather_plugin = WeatherPlugin()
            image_plugin = ImageGenerationPlugin(
                self.config.azure.endpoint, self.config.azure.dalle_deployment
            )
            self.kernel.add_plugin(weather_plugin, plugin_name="weather")
            self.kernel.add_plugin(image_plugin, plugin_name="image_generation")

            # Prepare OpenAI tools format
            function_metadata_list = [
                f.model_dump() for f in self.kernel.get_full_list_of_function_metadata()
            ]
            self.openai_tools, self.function_plugin_map = (
                convert_to_openai_function_format(function_metadata_list)
            )

            # Initialize Azure services using factory
            self.realtime_service = (
                await self.service_factory.create_realtime_websocket()
            )
            self.chat_completion_service = (
                await self.service_factory.create_chat_completion()
            )
            self.audio_to_text_service = (
                await self.service_factory.create_audio_to_text()
            )

            # Add services to kernel
            self.kernel.add_service(self.realtime_service, overwrite=True)
            self.kernel.add_service(self.chat_completion_service, overwrite=True)
            self.kernel.add_service(self.audio_to_text_service, overwrite=True)

            # Initialize audio processor
            self.audio_processor = AudioProcessor(
                self.audio_to_text_service, self.config
            )

            # Initialize event handler
            self.event_handler = RealtimeEventHandler(
                kernel=self.kernel,
                realtime_service=self.realtime_service,
                audio_processor=self.audio_processor,
                function_plugin_map=self.function_plugin_map,
            )

            # Initialize message handler
            self.message_handler = MessageHandler(self)

            logger.info("Application context initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize application context: {e}", exc_info=True
            )
            raise


# Global application context instance
app_context = AppContext()
