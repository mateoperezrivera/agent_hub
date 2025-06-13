import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AzureConfig:
    """Azure service configuration"""

    endpoint: str = os.getenv("AZURE_ENDPOINT", "https://aoai-mateop.openai.azure.com")
    realtime_deployment: str = os.getenv(
        "AZURE_REALTIME_DEPLOYMENT", "gpt-4o-realtime-preview"
    )
    chat_deployment: str = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
    transcribe_deployment: str = os.getenv(
        "AZURE_TRANSCRIBE_DEPLOYMENT", "gpt-4o-mini-transcribe"
    )
    dalle_deployment: str = os.getenv("AZURE_DALLE_DEPLOYMENT", "dall-e-3")
    api_version: str = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")

    # APIM configuration
    apim_enabled: bool = os.getenv("APIM_ENABLED", "false").lower() == "true"
    apim_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_APIM_ENDPOINT")
    apim_key_secret_name: str = os.getenv("APIM_KEY_SECRET_NAME", "apimKey")

    @property
    def websocket_url(self) -> str:
        base_endpoint = (
            self.apim_endpoint
            if self.apim_enabled and self.apim_endpoint
            else self.endpoint
        )
        return base_endpoint.replace("https://", "wss://") + "/openai"


@dataclass
class AudioConfig:
    """Audio processing configuration"""

    sample_rate: int = 16000
    channels: int = 1
    input_format: str = "pcm16"
    output_format: str = "pcm16"
    voice: str = "alloy"
    transcription_language: str = "es"
    transcription_timeout: int = 30


@dataclass
class AppConfig:
    """Application configuration"""

    azure: AzureConfig = field(default_factory=AzureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
