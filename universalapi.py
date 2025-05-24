"""
Universal chat completion API client supporting multiple providers.

Supported providers:
- openai (OpenAI API)
- anthropic (Claude via OpenAI-compatible endpoint)
- google_gemini (Google Gemini via Vertex AI OpenAI compatibility)
- openrouter (OpenRouter API)
- groq (Groq Cloud API)
- pollinations (Pollinations.ai free AI API)
- azure (Azure OpenAI Service)
- gitazure (GitHub Marketplace Azure via OpenAI-compatible endpoint)
- ollama (Local Ollama server)

Authentication:
- openai: Bearer token in Authorization header.
- anthropic: x-api-key header.
- google_gemini: Bearer token in Authorization header.
- openrouter: Bearer token in Authorization header.
- groq: Bearer token in Authorization header.
- pollinations: No API key required.
- azure: api-key header.
- gitazure: Bearer token in Authorization header.
- ollama: No auth by default.

Endpoints:
- All use OpenAI-compatible paths where available.
- Chat completions: /chat/completions for most; Azure uses /openai/deployments/<deployment>/chat/completions.
- Pollinations uses POST / with JSON {prompt, model}.
- Ollama uses POST /chat with JSON {model, messages}.
- List models: GET /models for compatible providers; Ollama uses GET /tags.

Usage example (run as script with env vars):
    PROVIDER=<provider> <PROVIDER>_API_KEY=<api_key> AZURE_ENDPOINT=<endpoint> python universalapi.py
"""
import os
import requests
import random
import sys

class ProviderConfig:
    def __init__(self, base_url, auth_header=None, api_key_prefix="", extra_params=None):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.auth_header = auth_header
        self.api_key_prefix = api_key_prefix
        self.extra_params = extra_params or {}

PROVIDERS = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        auth_header="Authorization", api_key_prefix="Bearer "
    ),
    "anthropic": ProviderConfig(
        base_url="https://api.anthropic.com/v1",
        auth_header="x-api-key", api_key_prefix=""
    ),
    "google_gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        auth_header="Authorization", api_key_prefix="Bearer "
    ),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        auth_header="Authorization", api_key_prefix="Bearer "
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        auth_header="Authorization", api_key_prefix="Bearer "
    ),
    "pollinations": ProviderConfig(
        base_url="https://text.pollinations.ai/openai",
        auth_header=None, api_key_prefix=""
    ),
    "azure": ProviderConfig(
        base_url=None,  # endpoint from env
        auth_header="api-key", api_key_prefix="",
        extra_params={"api_version": "2024-10-21"}
    ),
    "gitazure": ProviderConfig(
        base_url="https://models.github.ai/inference",
        auth_header="Authorization", api_key_prefix="Bearer "
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/api",
        auth_header=None, api_key_prefix=""
    ),
}

ALIASES = {
    "gemini": "google_gemini",
    "claude": "anthropic",
    "azure_openai": "azure",
    "github": "gitazure",
}

# Default fallbacks for models if selection fails
FALLBACK_MODELS = {
    "google_gemini": "models/gemini-2.0-flash",
    "openai": "gpt-3.5-turbo",
    "groq": "qwen-qwq-32b",
    "gitazure": "openai/gpt-4.1",
}


def get_available_models(provider, api_key=None):
    """
    Fetch the list of available model IDs for the given provider.
    """
    provider = provider.lower()
    provider = ALIASES.get(provider, provider)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    config = PROVIDERS[provider]
    headers = {}
    if config.auth_header and api_key:
        headers[config.auth_header] = f"{config.api_key_prefix}{api_key}"

    # Construct URL
    if provider == "azure":
        endpoint = os.getenv("AZURE_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_ENDPOINT environment variable required for Azure")
        api_version = config.extra_params.get("api_version")
        url = f"{endpoint}/openai/models?api-version={api_version}"
    elif provider == "pollinations":
        url = "https://text.pollinations.ai/models"
    elif provider == "ollama":
        url = f"{config.base_url}/tags"
    else:
        url = f"{config.base_url}/models"

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # Normalize model list
    if provider == "pollinations":
        return [m.get("name") for m in data if m.get("name")]
    elif provider == "ollama":
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    elif isinstance(data, dict) and "data" in data:
        return [m.get("id") for m in data["data"]]
    elif isinstance(data, list):
        return [m.get("id") for m in data]
    if isinstance(data, dict):
        if "id" in data:
            return [data["id"]]
    return []


def chat_completion(provider, api_key, model, messages, **kwargs):
    """
    Send a chat completion request and return a standardized response dict.
    """
    provider = provider.lower()
    provider = ALIASES.get(provider, provider)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    config = PROVIDERS[provider]

    headers = {"Content-Type": "application/json"}
    if config.auth_header and api_key:
        headers[config.auth_header] = f"{config.api_key_prefix}{api_key}"

    # Build URL and payload by provider
    if provider == "azure":
        endpoint = os.getenv("AZURE_ENDPOINT")
        api_version = config.extra_params.get("api_version")
        url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        payload = {"messages": messages}
    elif provider == "anthropic":
        url = f"{config.base_url}/chat/completions"
        payload = {"model": model, "messages": messages}
    elif provider == "pollinations":
        url = f"{config.base_url}"
        payload = {"model": model, "messages": messages}
    elif provider == "gitazure":
        url = f"{config.base_url}/chat/completions"
        payload = {"model": model, "messages": messages}
    elif provider == "ollama":
        url = f"{config.base_url}/chat"
        payload = {"model": model, "messages": messages}
    else:
        # openai, google_gemini, openrouter, groq
        url = f"{config.base_url}/chat/completions"
        payload = {"model": model, "messages": messages}

    # Append optional parameters
    for param in ["temperature", "max_tokens", "n", "stop", "top_p"]:
        if param in kwargs:
            payload[param] = kwargs[param]

    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        print(f'[DEBUG] {data}')

    # Standardize response
    result = {
        "id": data.get("id", "chatcmpl-generic"),
        "object": data.get("object", "chat.completion"),
        "created": data.get("created", 0),
        "model": data.get("model", model),
        "choices": [],
    }

    # Pollinations returns plain text


    # Ollama returns single message
    if provider == "ollama" and data.get("message"):
        msg = data.get("message")
        result["choices"].append({
            "index": 0,
            "message": msg,
            "finish_reason": "stop" if data.get("done", True) else None
        })
        return result

    # Default OpenAI-compatible normalization
    for i, choice in enumerate(data.get("choices", [])):
        msg = choice.get("message") or {"role": "assistant", "content": choice.get("text", "")} 
        result["choices"].append({
            "index": choice.get("index", i),
            "message": msg,
            "finish_reason": choice.get("finish_reason"),
        })
    if data.get("usage"):
        result["usage"] = data["usage"]
    return result


def main_model(provider, api_key):
    global model
    if os.getenv('MODEL'):
        model = os.getenv('MODEL')
    elif 'model' not in globals():
        print("Fetching available models...")
        models = get_available_models(provider, api_key)
        if not models:
            print("No models found or authentication error.")
            exit(1)
        for idx, m in enumerate(models):
            print(f"{idx+1}: {m}")
        try:
            choice = input("Select model number: ")
            model = models[int(choice)-1]
        except (Exception, KeyboardInterrupt):
            # fallback defaults
            if provider == 'openrouter':
                filtered = [m for m in models if ':free' in m]
                model = filtered[0] if filtered else models[0]
            elif provider == 'ollama':
                model = random.choice(models)
            elif provider == 'pollinations':
                model = None
            else:
                model = FALLBACK_MODELS.get(provider, models[0])
            print(f"Invalid selection or interrupt. Falling back to default model: {model}")
    else:
        pass
    return model


def main():
    provider = os.getenv("PROVIDER") or input("Set PROVIDER: ")
    provider = provider.lower()

    api_key = os.getenv(f"{provider.upper()}_API_KEY")

    provider = ALIASES.get(provider, provider)
    print(f"Provider: {provider}")

    model = main_model(provider, api_key)
    print(f"Model: {model}")

    sys_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_content = input("User: ")
    user_msg = {"role": "user", "content": user_content}
    response = chat_completion(provider, api_key, model, [sys_msg, user_msg])
    if response.get("choices"):
        print("Assistant:", response["choices"][0]["message"]["content"])
    else:
        print("No response.")

if __name__ == "__main__":
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print("\nExiting...")
