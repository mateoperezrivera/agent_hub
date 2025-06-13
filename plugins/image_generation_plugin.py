from typing import Annotated, Optional
from semantic_kernel.connectors.ai.open_ai import AzureTextToImage
from semantic_kernel.functions import kernel_function
from shared.ai_service_factory import AIServiceFactory
import logging

logger = logging.getLogger(__name__)


class ImageGenerationPlugin:
    """A plugin for generating images using Azure OpenAI's DALL-E model."""

    def __init__(self, azure_openai_endpoint: str, azure_openai_deployment_name: str):
        # Keep the parameters for backward compatibility
        self.endpoint = azure_openai_endpoint
        self.deployment_name = azure_openai_deployment_name
        self.image_generator: Optional[AzureTextToImage] = None
        self._factory = AIServiceFactory()

    async def _ensure_service(self):
        """Lazy initialization of the image generator service"""
        if self.image_generator is None:
            try:
                self.image_generator = await self._factory.create_text_to_image()
                logger.info("Image generator service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize image generator: {e}")
                # Fallback to direct initialization
                self.image_generator = AzureTextToImage(
                    endpoint=self.endpoint,
                    deployment_name=self.deployment_name,
                )

    @kernel_function(
        name="generate_image",
        description="Generate an image based on a text prompt using Azure OpenAI's DALL-E model.",
    )
    async def generate_image(
        self, prompt: Annotated[str, "The text description of the image to generate"]
    ) -> str:
        try:
            await self._ensure_service()
            response = await self.image_generator.generate_image(prompt)
            return response
        except Exception as e:
            return f"Error generating image: {str(e)}"
