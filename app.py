"""Main application entry point with Chainlit event handlers"""

import os
import logging
import chainlit as cl
from chainlit.cli import run_chainlit
from data_layer import CosmosDBDataLayer
from config import AppConfig
from shared.session_manager import SessionManager
from app_context import app_context  # Now from root
from audio.audio_session import AudioSessionManager
from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient
# Initialize configuration
config = AppConfig()

# Set up logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


@cl.on_app_startup
async def on_app_startup():
    """Initialize global kernel and services at app startup."""
    try:
        await app_context.initialize()
        cl.SemanticKernelFilter(kernel=app_context.kernel)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session"""
    SessionManager.init_chat_session()
    logger.info("New chat session started")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages from user"""
    await app_context.message_handler.handle_message(message)


@cl.on_audio_start
async def on_audio_start():
    """Handle audio recording start"""
    await AudioSessionManager.initialize_session()
    logger.info("Audio session initialized successfully")
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Handle incoming audio chunks"""
    await AudioSessionManager.process_chunk(chunk)


@cl.on_audio_end
async def on_audio_end():
    """Handle audio recording end"""
    await AudioSessionManager.finalize_session()

    # Log chat history for debugging
    chat_history = SessionManager.get("chat_history")
    logger.info(f"Current chat history: {chat_history}")

@cl.data_layer
def get_data_layer():
    """Initialize data layer for persistent storage"""
    try:
        # Check if all required environment variables are present
        required_vars = [
            "BLOB_STORAGE_ACCOUNT",
            "BLOB_STORAGE_KEY",
            "BLOB_STORAGE_CONTAINER",
            "COSMOS_DB_ENDPOINT",
            "COSMOS_DB_DATABASE",
            "COSMOS_DB_CONTAINER"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Data layer disabled - Missing environment variables: {', '.join(missing_vars)}")
            return None
        
        # Initialize storage client
        storage_client = AzureBlobStorageClient(
            storage_account=os.getenv("BLOB_STORAGE_ACCOUNT"),
            storage_key=os.getenv("BLOB_STORAGE_KEY"),
            container_name=os.getenv("BLOB_STORAGE_CONTAINER"),
        )
        
        # Initialize Cosmos DB data layer
        datalayer = CosmosDBDataLayer(
            account_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
            database_name=os.getenv("COSMOS_DB_DATABASE"),
            container_name=os.getenv("COSMOS_DB_CONTAINER"),
            use_msi=True,
            storage_provider=storage_client,
        )
        
        # Initialize database and container
        datalayer._init_database_and_container()
        
        logger.info("Data layer initialized successfully")
        return datalayer
        
    except Exception as e:
        logger.error(f"Failed to initialize data layer: {e}", exc_info=True)
        # Return None to allow the app to run without persistence
        return None
    
if __name__ == "__main__":
    run_chainlit(__file__)
    