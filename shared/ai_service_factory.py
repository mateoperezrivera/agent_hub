"""AI Service Factory with configuration integration"""

import os
import logging
from enum import Enum
from typing import Optional, TYPE_CHECKING, Any
from azure.identity.aio import (
    ManagedIdentityCredential,
    AzureCliCredential,
    ChainedTokenCredential,
)
from azure.keyvault.secrets.aio import SecretClient as AsyncSecretClient

from config import AppConfig

# Type checking imports - these don't actually import at runtime
if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.open_ai import (
        AzureChatCompletion,
        AzureRealtimeWebsocket,
        AzureTextToAudio,
        AzureAudioToText,
        AzureTextToImage,
    )

logger = logging.getLogger(__name__)

# Enable Azure SDK logging
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.DEBUG)  # Set to DEBUG to see detailed logs

# Enable Semantic Kernel logging
sk_logger = logging.getLogger("semantic_kernel")
sk_logger.setLevel(logging.DEBUG)


class ModelProvider(Enum):
    AZURE = "aoai"
    AZURE_INFERENCE = "azure_inference"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AMAZON = "amazon"
    GOOGLE = "google"
    VERTEX = "vertex"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    ONNX = "onnx"
    AZURE_AUDIO = "azure_audio"
    AZURE_TEXT_TO_AUDIO = "azure_text_to_audio"
    AZURE_REALTIME = "azure_realtime"


class ServiceType(Enum):
    """Service types available in the factory"""

    CHAT_COMPLETION = "chat_completion"
    REALTIME_WEBSOCKET = "realtime_websocket"
    AUDIO_TO_TEXT = "audio_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_AUDIO = "text_to_audio"


class AIServiceFactory:
    """Factory for creating AI services with centralized configuration"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self._credential = None
        self._use_apim = self.config.azure.apim_enabled
        self._apim_key: Optional[str] = None

    async def _get_apim_key(self) -> Optional[str]:
        """Get APIM key from Key Vault if needed"""
        if self._use_apim and self._apim_key is None:
            self._apim_key = await self.get_secret(
                self.config.azure.apim_key_secret_name
            )
        return self._apim_key

    async def create_realtime_websocket(
        self, service_id: str = "realtime"
    ) -> "AzureRealtimeWebsocket":
        """Create Azure Realtime WebSocket service"""
        logger.info("Creating Azure Realtime WebSocket service")
        
        # Lazy import only when needed
        from semantic_kernel.connectors.ai.open_ai import AzureRealtimeWebsocket
        # Import here to avoid circular dependency
        from realtime.realtime_handlers import audio_output_callback

        extra_args = {}
        if self._use_apim:
            apim_key = await self._get_apim_key()
            if apim_key:
                extra_args["api_key"] = apim_key
            endpoint = self.config.azure.apim_endpoint or self.config.azure.endpoint
        else:
            endpoint = self.config.azure.endpoint

        service = AzureRealtimeWebsocket(
            service_id=service_id,
            endpoint=endpoint,
            deployment_name=self.config.azure.realtime_deployment,
            api_version=self.config.azure.api_version,
            websocket_base_url=self.config.azure.websocket_url,
            audio_output_callback=audio_output_callback,
            default_headers={"OpenAI-Beta": "realtime=v1"},
            **extra_args,
        )

        logger.info(
            f"Created Realtime WebSocket service with deployment: {self.config.azure.realtime_deployment}"
        )
        return service

    async def create_chat_completion(
        self, service_id: str = "chat_completion"
    ) -> "AzureChatCompletion":
        """Create Azure Chat Completion service"""
        logger.info("Creating Azure Chat Completion service")
        
        # Lazy import only when needed
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

        extra_args = {}
        if self._use_apim:
            apim_key = await self._get_apim_key()
            if apim_key:
                extra_args["api_key"] = apim_key
            endpoint = self.config.azure.apim_endpoint or self.config.azure.endpoint
        else:
            endpoint = self.config.azure.endpoint

        service = AzureChatCompletion(
            service_id=service_id,
            endpoint=endpoint,
            deployment_name=self.config.azure.chat_deployment,
            api_version=self.config.azure.api_version,
            **extra_args,
        )

        logger.info(
            f"Created Chat Completion service with deployment: {self.config.azure.chat_deployment}"
        )
        return service

    async def create_audio_to_text(
        self, service_id: str = "audio_to_text"
    ) -> "AzureAudioToText":
        """Create Azure Audio-to-Text service"""
        logger.info("Creating Azure Audio-to-Text service")
        
        # Lazy import only when needed
        from semantic_kernel.connectors.ai.open_ai import AzureAudioToText

        extra_args = {}
        if self._use_apim:
            apim_key = await self._get_apim_key()
            if apim_key:
                extra_args["api_key"] = apim_key
            endpoint = self.config.azure.apim_endpoint or self.config.azure.endpoint
        else:
            endpoint = self.config.azure.endpoint

        service = AzureAudioToText(
            service_id=service_id,
            endpoint=endpoint,
            deployment_name=self.config.azure.transcribe_deployment,
            api_version=self.config.azure.api_version,
            **extra_args,
        )

        logger.info(
            f"Created Audio-to-Text service with deployment: {self.config.azure.transcribe_deployment}"
        )
        return service

    async def create_text_to_image(
        self, service_id: str = "text_to_image"
    ) -> "AzureTextToImage":
        """Create Azure Text-to-Image service"""
        logger.info("Creating Azure Text-to-Image service")
        
        # Lazy import only when needed
        from semantic_kernel.connectors.ai.open_ai import AzureTextToImage

        extra_args = {}
        if self._use_apim:
            apim_key = await self._get_apim_key()
            if apim_key:
                extra_args["api_key"] = apim_key
            endpoint = self.config.azure.apim_endpoint or self.config.azure.endpoint
        else:
            endpoint = self.config.azure.endpoint

        service = AzureTextToImage(
            service_id=service_id,
            endpoint=endpoint,
            deployment_name=self.config.azure.dalle_deployment,
            api_version=self.config.azure.api_version,
            **extra_args,
        )

        logger.info(
            f"Created Text-to-Image service with deployment: {self.config.azure.dalle_deployment}"
        )
        return service

    async def create_text_to_audio(
        self, service_id: str = "text_to_audio"
    ) -> "AzureTextToAudio":
        """Create Azure Text-to-Audio service"""
        logger.info("Creating Azure Text-to-Audio service")
        
        # Lazy import only when needed
        from semantic_kernel.connectors.ai.open_ai import AzureTextToAudio

        extra_args = {}
        if self._use_apim:
            apim_key = await self._get_apim_key()
            if apim_key:
                extra_args["api_key"] = apim_key
            endpoint = self.config.azure.apim_endpoint or self.config.azure.endpoint
        else:
            endpoint = self.config.azure.endpoint

        deployment = os.getenv("AZURE_OPENAI_TEXT_TO_AUDIO_DEPLOYMENT", "tts")

        service = AzureTextToAudio(
            service_id=service_id,
            endpoint=endpoint,
            deployment_name=deployment,
            api_version=self.config.azure.api_version,
            **extra_args,
        )

        logger.info(f"Created Text-to-Audio service with deployment: {deployment}")
        return service

    async def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from Azure Key Vault"""
        key_vault_name = os.environ.get("AZURE_KEY_VAULT_NAME")
        if not key_vault_name:
            logger.warning("AZURE_KEY_VAULT_NAME not set, cannot retrieve secrets")
            return None

        kv_uri = f"https://{key_vault_name}.vault.azure.net"

        try:
            if not self._credential:
                self._credential = ChainedTokenCredential(
                    ManagedIdentityCredential(), AzureCliCredential()
                )

            async with AsyncSecretClient(
                vault_url=kv_uri, credential=self._credential
            ) as client:
                retrieved_secret = await client.get_secret(secret_name)
                return retrieved_secret.value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None


# Legacy function to maintain backward compatibility
async def create_service(model_provider_str="aoai", apim_key=None) -> Any:
    """Legacy function for creating services - maintained for backward compatibility"""
    model_provider = ModelProvider(model_provider_str)

    # Use environment variable APIM settings if not provided
    APIM_ENABLED = os.environ.get("APIM_ENABLED", "False")
    APIM_ENABLED = True if APIM_ENABLED.lower() == "true" else False

    # If no apim_key provided, try to get it from Key Vault
    if APIM_ENABLED and apim_key is None:
        factory = AIServiceFactory()
        apim_key = await factory.get_secret(
            os.getenv("APIM_KEY_SECRET_NAME", "apimKey")
        )

    switch = {
        ModelProvider.AZURE: lambda: create_azure_client(apim_key, "aoai"),
        ModelProvider.AZURE_INFERENCE: lambda: create_azure_inference_client(
            "azure_inference"
        ),
        ModelProvider.OPENAI: lambda: create_openai_client("openai"),
        ModelProvider.ANTHROPIC: lambda: create_anthropic_client("anthropic"),
        ModelProvider.AMAZON: lambda: create_amazon_client("amazon"),
        ModelProvider.GOOGLE: lambda: create_google_client("google"),
        ModelProvider.VERTEX: lambda: create_vertex_client("vertex"),
        ModelProvider.MISTRAL: lambda: create_mistral_client("mistral"),
        ModelProvider.OLLAMA: lambda: create_ollama_client("ollama"),
        ModelProvider.ONNX: lambda: create_onnx_client("onnx"),
        ModelProvider.AZURE_AUDIO: lambda: create_audio_to_text_service("aoai"),
        ModelProvider.AZURE_TEXT_TO_AUDIO: lambda: create_text_to_audio_service("aoai"),
        ModelProvider.AZURE_REALTIME: lambda: create_realtime_websocket_client(
            "realtime"
        ),
    }

    if model_provider not in switch:
        raise ValueError(f"Unknown model provider: {model_provider}")

    return await switch[model_provider]()


async def create_azure_client(apim_key, service_id):
    """Create Azure OpenAI client"""
    # Lazy import
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    
    APIM_ENABLED = os.environ.get("APIM_ENABLED", "False").lower() == "true"

    if APIM_ENABLED:
        # If no apim_key provided, try to get it from Key Vault
        if apim_key is None:
            factory = AIServiceFactory()
            apim_key = await factory.get_secret(
                os.getenv("APIM_KEY_SECRET_NAME", "apimKey")
            )

        return AzureChatCompletion(
            service_id=service_id + "_chat_completion",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            endpoint=os.getenv("AZURE_OPENAI_APIM_ENDPOINT"),
            api_key=apim_key,
        )
    else:
        return AzureChatCompletion(
            service_id=service_id + "_chat_completion",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )


async def create_azure_inference_client(service_id):
    """Create Azure AI Inference client"""
    # Lazy import
    from semantic_kernel.connectors.ai.azure_ai_inference import (
        AzureAIInferenceChatCompletion,
    )
    
    factory = AIServiceFactory()
    key = os.getenv("AZURE_AI_INFERENCE_API_KEY")
    if key is None:
        key = await factory.get_secret("azureAIInferenceApiKey")
    model_id = os.getenv("AZURE_AI_INFERENCE_MODEL_ID")
    service = AzureAIInferenceChatCompletion(
        service_id=service_id + "_chat_completion", ai_model_id=model_id, api_key=key
    )
    return service


async def create_openai_client(service_id):
    """Create OpenAI client"""
    # Lazy import
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    
    factory = AIServiceFactory()
    key = os.getenv("OPENAI_API_KEY")
    if key is None:
        key = await factory.get_secret("openAIApiKey")
    service = OpenAIChatCompletion(
        api_key=key, service_id=service_id + "_chat_completion"
    )
    return service


async def create_anthropic_client(service_id):
    """Create Anthropic client"""
    # Lazy import
    from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
    
    factory = AIServiceFactory()
    key = os.getenv("ANTHROPIC_API_KEY")
    if key is None:
        key = await factory.get_secret("anthropicApiKey")
    service = AnthropicChatCompletion(
        api_key=key, service_id=service_id + "_chat_completion"
    )
    return service


async def create_amazon_client(service_id):
    """Create Amazon Bedrock client"""
    # Lazy import
    from semantic_kernel.connectors.ai.bedrock import BedrockChatCompletion
    
    service = BedrockChatCompletion(service_id=service_id + "_chat_completion")
    return service


async def create_google_client(service_id):
    """Create Google AI client"""
    # Lazy import
    from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
    
    factory = AIServiceFactory()
    key = os.getenv("GOOGLE_AI_API_KEY")
    if key is None:
        key = await factory.get_secret("googleAiApiKey")
    service = GoogleAIChatCompletion(
        api_key=key, service_id=service_id + "_chat_completion"
    )
    return service


async def create_vertex_client(service_id):
    """Create Vertex AI client"""
    # Lazy import
    from semantic_kernel.connectors.ai.google.vertex_ai import VertexAIChatCompletion
    
    service = VertexAIChatCompletion(service_id=service_id + "_chat_completion")
    return service


async def create_mistral_client(service_id):
    """Create Mistral AI client"""
    # Lazy import
    from semantic_kernel.connectors.ai.mistral_ai import MistralAIChatCompletion
    
    factory = AIServiceFactory()
    key = os.getenv("MISTRAL_API_KEY")
    if key is None:
        key = await factory.get_secret("mistralApiKey")
    service = MistralAIChatCompletion(
        api_key=key, service_id=service_id + "_chat_completion"
    )
    return service


async def create_ollama_client(service_id):
    """Create Ollama client"""
    # Lazy import
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    
    service = OllamaChatCompletion(service_id=service_id + "_chat_completion")
    return service


async def create_onnx_client(service_id):
    """Create ONNX client"""
    # Lazy import
    from semantic_kernel.connectors.ai.onnx import OnnxGenAIChatCompletion
    
    service = OnnxGenAIChatCompletion(
        template=os.getenv("ONNX_GEN_AI_CHAT_TEMPLATE"),
        service_id=service_id + "_chat_completion",
    )
    return service


async def create_realtime_websocket_client(service_id="realtime"):
    """Create Azure Realtime WebSocket client"""
    factory = AIServiceFactory()
    return await factory.create_realtime_websocket(service_id)


async def create_audio_to_text_service(service_id):
    """Create Azure Audio-to-Text service"""
    factory = AIServiceFactory()
    return await factory.create_audio_to_text(service_id)


async def create_text_to_audio_service(service_id):
    """Create Azure Text-to-Audio service"""
    factory = AIServiceFactory()
    return await factory.create_text_to_audio(service_id)
