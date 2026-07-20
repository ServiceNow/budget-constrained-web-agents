"""
Unified LLM client that routes to vLLM, OpenRouter, Azure, or direct Anthropic based on model_name.
"""
import os
import csv
import time
import random
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Callable, TypeVar, Union
from openai import AzureOpenAI, OpenAI, ChatCompletion, RateLimitError as OpenAIRateLimitError
from anthropic import Anthropic, AnthropicFoundry, RateLimitError as AnthropicRateLimitError

T = TypeVar("T")

# Retries for 429 / token-based rate limits (each attempt sleeps first after failure)
_RATE_LIMIT_MAX_RETRIES = 3

# A built-in retry mechanism already exists, we use this as an extra layer of retries to wait for longer.
def _call_with_rate_limit_retry(fn: Callable[[], T]) -> T:
    """Run fn(), on OpenAI/Anthropic rate limit sleep and retry up to _RATE_LIMIT_MAX_RETRIES times."""
    last: Optional[BaseException] = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        try:
            return fn()
        except (OpenAIRateLimitError, AnthropicRateLimitError) as e:
            last = e
            if attempt == _RATE_LIMIT_MAX_RETRIES - 1:
                raise last
            delay = 60 + random.uniform(0, 10.)
            print(
                f"[llm_client] Rate limited ({type(e).__name__}); sleeping {delay:.1f}s "
                f"then retry ({attempt + 1}/{_RATE_LIMIT_MAX_RETRIES - 1})..."
            )
            time.sleep(delay)
    assert last is not None
    raise last

_client_cache: dict[str, OpenAI] = {}

# Path to API usage CSV file at repo root
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_API_DIR)
API_USAGE_CSV = os.path.join(_REPO_ROOT, "api_usage.csv")


def log_api_usage(provider: str, model_name: str, usage: Any):
    """
    Log API usage to CSV file for all providers (vllm, openrouter, azure, anthropic).
    
    Args:
        provider: Provider name (e.g., 'vllm', 'openrouter', 'azure', 'anthropic')
        model_name: Name of the model used
        usage: Usage object from OpenAI response (has prompt_tokens, completion_tokens, total_tokens)
    """
    timestamp = datetime.now().isoformat()
    
    # Extract token usage values
    prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
    completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
    total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(API_USAGE_CSV)
    
    # Append row to CSV
    with open(API_USAGE_CSV, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'provider', 'model_name', 'prompt_tokens', 'completion_tokens', 'total_tokens']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write usage data
        writer.writerow({
            'timestamp': timestamp,
            'provider': provider,
            'model_name': model_name,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        })


def get_client(model_name: str) -> Tuple[OpenAI, str]:
    """
    Get OpenAI client configured for vLLM or OpenRouter based on model prefix.
    Clients are cached by port number for vLLM, by name for OpenRouter.
    
    Args:
        model_name: Model name to parse:
            - "vllm" or "vllm/model_name" -> port 8000
            - "vllm:8000" or "vllm:8000/model_name" -> port 8000
            - "vllm:8001" or "vllm:8001/model_name" -> port 8001
            - "openrouter" or any other -> OpenRouter client
            - "anthropic/MODEL" -> Anthropic API (ANTHROPIC_API_KEY; optional ANTHROPIC_BASE_URL)
        
    Returns:
        Tuple of (OpenAI client, actual_model_name)
        For vLLM: actual_model_name is the model name without the "vllm:PORT/" prefix
        For OpenRouter: actual_model_name is the original model_prefix
        For anthropic/: actual_model_name is the Anthropic model id after the prefix
    """
    kwargs = {"api_key": None, "base_url": None}
    # Parse vLLM model prefix to extract port
    if model_name.startswith("vllm"):
        prefix, actual_model_name = model_name.split("/", 1)
        if ":" in prefix:
            try:
                port = int(prefix.split(":")[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid vLLM model prefix format: {model_name}. Expected 'vllm:PORT/model_name' or 'vllm/model_name'")
        else:
            port = 8000
        
        cache_key = prefix
        kwargs["base_url"] = f"http://localhost:{port}/v1"
        kwargs["api_key"] = os.environ.get("VLLM_API_KEY", "EMPTY")
        kwargs["max_retries"] = 4
        client_class = OpenAI
    elif model_name.startswith("anthropic"):
        kwargs["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
        if not kwargs["api_key"]:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set.")
        actual_model_name = model_name.split("/", 1)[-1]
        cache_key = "anthropic"
        client_class = Anthropic
        base = os.environ.get("ANTHROPIC_BASE_URL")
        if base:
            kwargs["base_url"] = base
    elif model_name.startswith("azure"):
        kwargs["api_key"] = os.environ.get("AZURE_API_KEY")
        if not kwargs["api_key"]:
            raise ValueError("AZURE_API_KEY environment variable must be set.")
        actual_model_name = model_name.split("/", 1)[-1]
        if "claude" in actual_model_name:
            cache_key = "azure_anthropic"
            client_class = AnthropicFoundry
            kwargs["base_url"] = os.environ.get("AZURE_ANTHROPIC_ENDPOINT")
        else:
            cache_key = "azure_openai"
            client_class = AzureOpenAI
            kwargs["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            kwargs["api_version"] = os.environ.get("AZURE_API_VERSION")
    else:
        actual_model_name = model_name.replace("openrouter/", "")
        cache_key = "openrouter"
        kwargs["base_url"] = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY")
        client_class = OpenAI
        if not kwargs["api_key"]:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set.")
        
    if cache_key not in _client_cache:
        _client_cache[cache_key] = client_class(**kwargs)
    return _client_cache[cache_key], actual_model_name


def llm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    n: int = 1,
    usage_name: Optional[str] = None,
    **kwargs
):
    """
    Unified LLM completion function that routes to vLLM, OpenRouter, or Azure.
    
    Args:
        model: Name of the model to use. For vLLM, supports:
            - "vllm/model_name" (defaults to port 8000)
            - "vllm:PORT/model_name" (explicit port PORT)
            For OpenRouter: "openrouter/openai/gpt-4o", "openrouter/anthropic/claude-3-5-sonnet", etc.
            For Azure: "azure/model_name" (supports both OpenAI and Anthropic models)
            For direct Anthropic: "anthropic/claude-sonnet-4-5-20250514"
        messages: List of messages in OpenAI format
        temperature: Sampling temperature
        n: Number of completions to generate
        usage_name: Optional name/identifier for this LLM call. If provided, prints token usage.
        **kwargs: Additional arguments to pass to the completion function
    
    Returns:
        Response object compatible with OpenAI format
    """
    client, actual_model_name = get_client(model)
    
    # Anthropic API (direct or Azure Foundry): OpenAI-shaped messages -> Anthropic messages.create
    if isinstance(client, (Anthropic, AnthropicFoundry)):
        response = _anthropic_completion(
            client, actual_model_name, messages, temperature, n, **kwargs
        )
    else:
        # OpenAI-compatible client (OpenAI, AzureOpenAI, vLLM)
        create_kwargs = {
            "model": actual_model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if n > 1:
            create_kwargs["n"] = n
        if "gpt-5-mini" in actual_model_name:
            create_kwargs.pop("temperature", None)
        if "gpt-5" in actual_model_name and "max_tokens" in kwargs:
            create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens")
        response = _call_with_rate_limit_retry(
            lambda: client.chat.completions.create(**create_kwargs)
        )
    
    # Determine provider and log usage for all providers
    if hasattr(response, 'usage'):
        if model.startswith("vllm"):
            pass
        elif model.startswith("anthropic"):
            log_api_usage("anthropic", model, response.usage)
        elif model.startswith("azure"):
            log_api_usage("azure", model, response.usage)
        else:
            log_api_usage("openrouter", model, response.usage)
        
    
    # Print token usage if usage_name is provided
    if usage_name is not None and getattr(response, 'usage', None):
        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
        completion_tokens = getattr(response.usage, 'completion_tokens', 0)
        total_tokens = getattr(response.usage, 'total_tokens', 0)
        print(f"[{usage_name}] LLM: {model} | Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
    
    return response


def _anthropic_completion(
    client: Union[Anthropic, AnthropicFoundry],
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    n: int,
    **kwargs
):
    """
    Handle completion for AnthropicFoundry client.
    Converts OpenAI-style messages to Anthropic format and normalizes the response.
    """
    # Extract system message if present
    system_message = None
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Build create kwargs for Anthropic
    create_kwargs = {
        "model": model,
        "messages": anthropic_messages,
        "temperature": temperature,
        "max_tokens": kwargs.pop("max_tokens", 8192),
    }
    if system_message:
        create_kwargs["system"] = system_message
    
    # Add any remaining kwargs (excluding OpenAI-specific ones)
    openai_only_kwargs = {"n", "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs"}
    for key, value in kwargs.items():
        if key not in openai_only_kwargs:
            create_kwargs[key] = value
    
    # Make the API call(s) - Anthropic doesn't support n, so we make multiple calls
    if n > 1:
        responses = [
            _call_with_rate_limit_retry(lambda: client.messages.create(**create_kwargs))
            for _ in range(n)
        ]
        return _convert_anthropic_to_openai_response(responses[0], additional_responses=responses[1:])
    else:
        anthropic_response = _call_with_rate_limit_retry(
            lambda: client.messages.create(**create_kwargs)
        )
        return _convert_anthropic_to_openai_response(anthropic_response)


def _convert_anthropic_to_openai_response(anthropic_response, additional_responses=None):
    """
    Convert an Anthropic response to OpenAI-compatible format.
    If additional_responses is provided, aggregate all responses into multiple choices.
    """
    # Create a simple namespace object to mimic OpenAI response structure
    class OpenAICompatibleResponse:
        def __init__(self):
            self.id = anthropic_response.id
            self.model = anthropic_response.model
            self.choices = []
            self.usage = None
    
    class Choice:
        def __init__(self, index, message, finish_reason):
            self.index = index
            self.message = message
            self.finish_reason = finish_reason
    
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    class Usage:
        def __init__(self, prompt_tokens, completion_tokens, total_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens
    
    # Map Anthropic stop_reason to OpenAI finish_reason
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    
    def extract_choice(resp, index):
        content_text = ""
        for block in resp.content:
            if hasattr(block, 'text'):
                content_text += block.text
        finish_reason = finish_reason_map.get(resp.stop_reason, resp.stop_reason)
        message = Message(role="assistant", content=content_text)
        return Choice(index=index, message=message, finish_reason=finish_reason)
    
    response = OpenAICompatibleResponse()
    
    # Collect all responses to process
    all_responses = [anthropic_response] + (additional_responses or [])
    
    # Create choices from all responses
    for idx, resp in enumerate(all_responses):
        response.choices.append(extract_choice(resp, idx))
    
    # Aggregate usage from all responses
    total_input_tokens = 0
    total_output_tokens = 0
    for resp in all_responses:
        if hasattr(resp, 'usage'):
            total_input_tokens += getattr(resp.usage, 'input_tokens', 0)
            total_output_tokens += getattr(resp.usage, 'output_tokens', 0)
    
    response.usage = Usage(
        prompt_tokens=total_input_tokens,
        completion_tokens=total_output_tokens,
        total_tokens=total_input_tokens + total_output_tokens
    )
    
    return response

def get_contents(response: ChatCompletion, placeholder_if_failed: str = None, remove_thinking: bool = False) -> list[str]:
    if not hasattr(response, 'choices') or len(response.choices) == 0:
        return [placeholder_if_failed] if placeholder_if_failed is not None else []
    contents = []
    for choice in getattr(response, 'choices', []):
        if remove_thinking:
            contents.append(choice.message.content.split("</think>")[-1].strip())
        else:
            contents.append(choice.message.content.strip())
    return contents
