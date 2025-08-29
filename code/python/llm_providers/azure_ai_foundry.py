# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Azure AI Foundry LLM provider implementation.

This module provides integration with Azure AI Foundry endpoints for LLM functionality.
Azure AI Foundry is Microsoft's unified AI platform for deploying and managing AI models.
"""

import json
import aiohttp
import asyncio
import threading
from typing import Dict, Any, Optional

from llm_providers.llm_provider import LLMProvider
from core.config import CONFIG
from misc.logger.logging_config_helper import get_configured_logger

logger = get_configured_logger("azure_ai_foundry")


class AzureAIFoundryProvider(LLMProvider):
    """Implementation of LLMProvider for Azure AI Foundry."""
    
    # Global client session with thread-safe initialization
    _session_lock = threading.Lock()
    _session = None

    @classmethod
    def get_azure_endpoint(cls) -> str:
        """Get the Azure AI Foundry endpoint from configuration."""
        provider_config = CONFIG.llm_endpoints.get("azure_ai_foundry")
        if provider_config and provider_config.endpoint:
            endpoint = provider_config.endpoint
            if endpoint:
                endpoint = endpoint.strip('"')  # Remove quotes if present
                # The endpoint should already be complete with path and API version
                # Just return it as-is
                return endpoint
        return None

    @classmethod
    def get_api_key(cls) -> str:
        """Get the Azure AI Foundry API key from configuration."""
        provider_config = CONFIG.llm_endpoints.get("azure_ai_foundry")
        if provider_config and provider_config.api_key:
            api_key = provider_config.api_key
            if api_key:
                api_key = api_key.strip('"')  # Remove quotes if present
                return api_key
        return None

    @classmethod
    def get_model_from_config(cls, high_tier=False) -> str:
        """Get the appropriate model from configuration based on tier."""
        provider_config = CONFIG.llm_endpoints.get("azure_ai_foundry")
        if provider_config and provider_config.models:
            model_name = provider_config.models.high if high_tier else provider_config.models.low
            if model_name:
                return model_name
        # Default values if not found
        default_model = "gpt-oss-120b"
        return default_model

    @classmethod
    def get_client(cls) -> aiohttp.ClientSession:
        """Get or initialize the aiohttp client session."""
        with cls._session_lock:  # Thread-safe session initialization
            if cls._session is None:
                endpoint = cls.get_azure_endpoint()
                api_key = cls.get_api_key()
                
                if not all([endpoint, api_key]):
                    error_msg = "Missing required Azure AI Foundry configuration"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Create a client session with default headers
                timeout = aiohttp.ClientTimeout(total=30)
                cls._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                )
                logger.debug("Azure AI Foundry client session initialized successfully")
                
        return cls._session

    @classmethod
    def clean_response(cls, content: str) -> Dict[str, Any]:
        """
        Clean and extract JSON content from Azure AI Foundry response.
        
        Args:
            content: The content to clean. May be None.
            
        Returns:
            Parsed JSON object or empty dict if content is None or invalid
            
        Raises:
            ValueError: If the content doesn't contain a valid JSON object
        """
        # Handle None content case
        if content is None:
            logger.warning("Received None content from Azure AI Foundry")
            return {}
            
        # Handle empty string case
        response_text = content.strip()
        if not response_text:
            logger.warning("Received empty content from Azure AI Foundry")
            return {}
            
        # Remove markdown code block indicators if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Try to parse the entire response as JSON first
        try:
            result = json.loads(response_text)
            logger.debug(f"Successfully parsed entire response as JSON: {result}")
            
            # Handle the 'final' wrapper issue - unwrap if needed
            if isinstance(result, dict) and 'final' in result and 'score' not in result:
                final_content = result['final']
                logger.debug(f"Unwrapping 'final' key, content: {final_content}")
                
                # Handle array wrapper
                if isinstance(final_content, list) and len(final_content) > 0:
                    final_content = final_content[0]
                
                # Parse JSON string if it's a string
                if isinstance(final_content, str):
                    try:
                        unwrapped = json.loads(final_content)
                        logger.debug(f"Successfully unwrapped final content: {unwrapped}")
                        return unwrapped
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse 'final' content as JSON: {final_content}")
                        return {"score": 0, "description": "Malformed response from LLM"}
                
                # If final_content is already a dict, return it
                if isinstance(final_content, dict):
                    return final_content
            
            return result
            
        except json.JSONDecodeError:
            # If that fails, try to find and extract a JSON object
            logger.error("Failed to parse entire response as JSON, trying to extract JSON object")
            
        # Find the JSON object within the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            error_msg = "No valid JSON object found in response"
            logger.error(f"{error_msg}, content: {response_text}")
            return {}
            
        json_str = response_text[start_idx:end_idx]
                
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response as JSON: {e}"
            logger.error(f"{error_msg}, content: {json_str}")
            return {}

    async def get_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        high_tier: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get completion from Azure AI Foundry.
        
        Args:
            prompt: The prompt to send to the model
            schema: JSON schema for the expected response
            model: Specific model to use (overrides configuration)
            temperature: Model temperature (0-1)
            max_tokens: Maximum tokens in the generated response
            timeout: Request timeout in seconds
            high_tier: Whether to use the high-tier model from config
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Parsed JSON response
            
        Raises:
            ValueError: If the response cannot be parsed as valid JSON
            TimeoutError: If the request times out
        """
        # Use specified model or get from config based on tier
        model_to_use = model if model else self.get_model_from_config(high_tier)
        
        session = self.get_client()
        endpoint = self.get_azure_endpoint()
        
        system_prompt = f"""You must respond with valid JSON that matches this schema: {json.dumps(schema)}
Your response must be a JSON object with the exact structure specified in the schema."""
        
        request_body = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "model": model_to_use,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low"
        }
        
        logger.debug(f"Sending completion request to Azure AI Foundry with model: {model_to_use}")
        
        try:
            async with session.post(endpoint, json=request_body, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Azure AI Foundry API error (status {response.status}): {error_text}")
                    return {}
                
                response_json = await response.json()
                
                # Extract content from the response
                if not response_json or "choices" not in response_json or not response_json["choices"]:
                    logger.error("Invalid or empty response from Azure AI Foundry")
                    return {}
                
                choice = response_json["choices"][0]
                if "message" not in choice or "content" not in choice["message"]:
                    logger.error("Response does not contain expected 'message.content' structure")
                    return {}
                
                content = choice["message"]["content"]
                result = self.clean_response(content)
                
                logger.debug(f"Successfully received and parsed response from Azure AI Foundry")
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"Azure AI Foundry request timed out after {timeout} seconds")
            return {}
        except aiohttp.ClientError as e:
            logger.error(f"Azure AI Foundry HTTP request failed: {type(e).__name__}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Azure AI Foundry completion failed: {type(e).__name__}: {str(e)}")
            raise

    @classmethod
    async def close(cls):
        """Close the aiohttp session when shutting down."""
        with cls._session_lock:
            if cls._session:
                await cls._session.close()
                cls._session = None
                logger.debug("Azure AI Foundry client session closed")


# Create a singleton instance
provider = AzureAIFoundryProvider()

# For backwards compatibility
get_azure_ai_foundry_completion = provider.get_completion