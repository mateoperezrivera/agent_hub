# Agent Hub Realtime

A powerful AI assistant application built with Chainlit that supports real-time voice interactions, plugin extensions, and integration with Azure OpenAI services. The application features audio processing, chat completion, image generation, and extensible plugin architecture including support for Model Context Protocol (MCP) plugins.

## Features

- **Real-time Voice Interaction**: Seamless voice-to-voice conversations using Azure OpenAI's realtime API
- **Multi-modal Capabilities**: Support for text chat, voice input/output, and image generation
- **Plugin Architecture**: Extensible plugin system with built-in plugins and MCP support
- **Persistent Storage**: Optional data persistence using Azure Cosmos DB and Blob Storage
- **Session Management**: Robust session handling for chat and audio interactions
- **Azure Integration**: Full integration with Azure OpenAI services

## Prerequisites

- Python 3.12+
- Azure OpenAI account with access to:
  - GPT-4 or GPT-4o models
  - Realtime API (for voice features)
  - DALL-E 3 (for image generation)
  - Whisper (for transcription)
- (Optional) Azure Cosmos DB and Blob Storage accounts for persistence

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent_hub_realtime
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the environment template and configure:
```bash
cp .env.template .env
```

4. Configure your `.env` file with your Azure credentials and settings (see Configuration section)

## Configuration

The application uses environment variables for configuration. Key settings include:

### Required Azure OpenAI Settings
```env
AZURE_ENDPOINT=https://your-aoai-instance.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_REALTIME_DEPLOYMENT=gpt-4o-realtime-preview
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_TRANSCRIBE_DEPLOYMENT=whisper
AZURE_DALLE_DEPLOYMENT=dall-e-3
```

### Optional Persistence Settings
```env
# Cosmos DB
COSMOS_DB_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
COSMOS_DB_DATABASE=your-database-name
COSMOS_DB_CONTAINER=your-container-name

# Blob Storage
BLOB_STORAGE_ACCOUNT=your-storage-account
BLOB_STORAGE_KEY=your-storage-key
BLOB_STORAGE_CONTAINER=your-container-name
```

### API Management (Optional)
```env
APIM_ENABLED=false
AZURE_OPENAI_APIM_ENDPOINT=https://your-apim-endpoint.azure-api.net
APIM_KEY_SECRET_NAME=apimKey
```

## Running the Application

Start the Chainlit application:
```bash
chainlit run app.py -w
```

The application will be available at `http://localhost:8000`

## Using the Application

### Text Chat
Simply type your message in the chat interface. The AI will respond using the configured chat model.

### Voice Interaction
1. Click the microphone button to start recording
2. Speak your message
3. Click stop to end recording
4. The AI will transcribe your message and respond both in text and voice

### Available Commands
- **Weather queries**: "What's the weather in Seattle?"
- **Image generation**: "Generate an image of a sunset over mountains"
- **Database queries**: Access SQL Server data (if configured)
- **Fabric resources**: Access Microsoft Fabric resources (if configured)

## Plugin System

The application supports two types of plugins:

### 1. Native Python Plugins

Create a new plugin by extending the base plugin pattern:

```python
# plugins/my_custom_plugin.py
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from typing import Annotated

class MyCustomPlugin(KernelBaseModel):
    """Description of your plugin"""
    
    @kernel_function(
        name="my_function",
        description="What this function does"
    )
    async def my_function(
        self,
        parameter: Annotated[str, "Description of parameter"]
    ) -> str:
        """Function implementation"""
        # Your logic here
        return f"Result: {parameter}"
```

Register your plugin in [`app_context.py`](app_context.py):

```python
# In AppContext.initialize()
from plugins.my_custom_plugin import MyCustomPlugin

my_plugin = MyCustomPlugin()
self.kernel.add_plugin(my_plugin, plugin_name="my_custom")
```

### 2. MCP (Model Context Protocol) Plugins

MCP plugins allow integration with external tools and services. The application includes examples for SQL and Fabric access.

#### Creating an MCP Plugin

1. Create your MCP server (see [MCP documentation](https://github.com/modelcontextprotocol))

2. Add it to your application:

```python
# In app_context.py
from semantic_kernel.connectors.mcp import MCPStdioPlugin

my_mcp_plugin = MCPStdioPlugin(
    name="my_mcp_plugin",
    description="What this plugin does",
    command="python",  # or "uv", "node", etc.
    args=["path/to/your/mcp_server.py"],
    env={
        "ENV_VAR": "value"  # Any environment variables needed
    },
)
await my_mcp_plugin.connect()
self.kernel.add_plugin(my_mcp_plugin, plugin_name="my_mcp")
```

## Architecture

```
├── app.py                 # Main Chainlit application entry point
├── app_context.py         # Application context and initialization
├── config.py             # Configuration management
├── data_layer.py         # Cosmos DB persistence layer
├── audio/                # Audio processing modules
│   ├── audio_processor.py
│   └── audio_session.py
├── plugins/              # Plugin modules
│   ├── weather_plugin.py
│   └── image_generation_plugin.py
├── realtime/             # Realtime API handlers
│   └── realtime_handlers.py
└── shared/               # Shared utilities
    ├── ai_service_factory.py
    ├── message_handler.py
    └── session_manager.py
```



## Acknowledgments

- Built with [Chainlit](https://chainlit.io/)
- Powered by [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- Uses [Semantic Kernel](https://github.com/microsoft/semantic-kernel) for AI orchestration