"""Utility functions for the realtime application"""

import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


def convert_to_openai_function_format(
    function_metadata_list: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Convert Semantic Kernel function metadata to OpenAI function format.

    Args:
        function_metadata_list: List of function metadata from Semantic Kernel

    Returns:
        Tuple of (openai_functions, function_plugin_map)
        - openai_functions: List of functions in OpenAI format
        - function_plugin_map: Mapping of function_name to plugin_name
    """
    openai_functions = []
    function_plugin_map = {}

    for func in function_metadata_list:
        function_plugin_map[func["name"]] = func["plugin_name"]

        openai_func = {
            "type": "function",
            "name": func["name"],
            "description": func["description"] or "",  # Convert None to empty string
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }

        for param in func["parameters"]:
            param_name = param["name"]
            param_info = {
                "type": param["schema_data"]["type"],
                "description": param["description"]
                or "",  # Convert None to empty string
            }

            openai_func["parameters"]["properties"][param_name] = param_info

            if param["is_required"]:
                openai_func["parameters"]["required"].append(param_name)

        openai_functions.append(openai_func)

    return openai_functions, function_plugin_map


def format_error_message(error: Exception, context: str = "") -> str:
    """Format error message for display.

    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred

    Returns:
        Formatted error message
    """
    if context:
        return f"❌ {context}: {str(error)}"
    return f"❌ Error: {str(error)}"


def format_transcription_message(transcript: str) -> str:
    """Format transcription message for display.

    Args:
        transcript: The transcribed text

    Returns:
        Formatted transcription message
    """
    return f"🎤 Transcription: {transcript}"
