#!/usr/bin/env python3
from datetime import datetime
import argparse
import json
import os
import re
import sys
import time
import glob
import shutil
import platform
import requests
import random

# Enhanced .env support with fallback
DOTENV_AVAILABLE = True
try:
    from dotenv import load_dotenv, set_key, dotenv_values
except ImportError:
    DOTENV_AVAILABLE = False
    print("Info: python-dotenv not available. Using fallback environment variable handling.")

# Rich imports with fallback
RICH_AVAILABLE = True
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.align import Align
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Using fallback mode.")

# Intializing the Package Manager Detection functions

# Mapping of package manager commands to descriptions
dicts = {
    # System-level (Linux)
    "apt": "Debian/Ubuntu based",
    "apt-get": "Debian/Ubuntu based (lower-level)",
    "dpkg": "Debian/Ubuntu based (core package tool)",
    "yum": "Older RHEL/Fedora/CentOS",
    "dnf": "Modern RHEL/Fedora/CentOS",
    "rpm": "RHEL/Fedora/CentOS (core package tool)",
    "pacman": "Arch Linux based",
    "zypper": "SUSE/openSUSE",
    "emerge": "Gentoo",
    "apk": "Alpine Linux",
    # Cross-distro & macOS
    "snap": "Snapcraft (Canonical)",
    "flatpak": "Flatpak",
    "brew": "Homebrew (macOS/Linux)",
    "nix-env": "Nix package manager",
    # Windows
    "choco": "Chocolatey (Windows)",
    "winget": "WinGet (Windows)",
    "scoop": "Scoop (Windows)",
    # BSD
    "pkg": "Termux apt wrapper / FreeBSD pkg",
    "pkg_add": "OpenBSD pkg_add",
    "pkgin": "NetBSD pkgin",
    # Language-specific / Environment
    "pip": "Python",
    "pip3": "Python 3",
    "conda": "Anaconda/Miniconda (Python, R, etc.)",
    "npm": "Node.js (JavaScript)",
    "yarn": "Node.js (alternative to npm)",
    "gem": "Ruby",
    "composer": "PHP",
    "cargo": "Rust",
    "go": "Go (go get/install)",
    "cpan": "Perl (CPAN)",
}

# Translation map to clean up descriptions for summary
_summary_trans = str.maketrans({
    "(": "-",
    ")": "",
    "/": "",
    " ": "",
    "\n": "",
})

def detect_pkg_managers():
    """
    Detects available package managers and returns a list of tuples (cmd, desc).
    """
    found = []
    for cmd, desc in dicts.items():
        if shutil.which(cmd):
            found.append((cmd, desc))
    return found


def summarize_pkg_managers():
    """
    Returns a comma-separated string of detected managers in the form:
    cmd-sanitizedDesc, ...
    where sanitizedDesc has no spaces or special chars.
    """
    entries = []
    for cmd, desc in detect_pkg_managers():
        clean_desc = desc.translate(_summary_trans)
        entries.append(f"{cmd}-{clean_desc}")
    return ", ".join(entries), platform.system(), platform.machine()


# Intializing the Chat Completion functions

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


# Initialize console (Rich or fallback)
if RICH_AVAILABLE:
    console = Console(force_terminal=True, color_system='truecolor')
else:
    class FallbackConsole:
        def print(self, *args, **kwargs):
            # Remove Rich markup for fallback
            text = str(args[0]) if args else ""
            # Simple markup removal
            text = re.sub(r'\[.*?\]', '', text)
            print(text)
        
        def rule(self, title=""):
            print("=" * 50)
            if title:
                print(f" {title} ")
                print("=" * 50)
    
    console = FallbackConsole()

# Enhanced default configuration with .env support
DEFAULT_CONFIG = {
    "api_key": "YOUR_API_KEY",
    "provider": "google_gemini",
    "model": "gemini-2.0-flash",
    "api_endpoint": "AZURE_API_ENDPOINT",
    "system_prompt_template": "You are ShellSense, a helpful command-line assistant. Your goal is to provide a SINGLE-Line, executable command-line command based on the user's request.\n\nOperating System: {os_name}\nArchitecture and Package Managers (if any): {os_arch}\n\nIMPORTANT INSTRUCTIONS:\n1. Respond ONLY with the command-line command.\n2. Enclose the command within single backticks (`command`).\n3. Do NOT include any explanations, comments, or language identifiers (like 'bash', 'sh', 'powershell') outside or inside the backticks.\n4. Aim to Provide only one to two command suggestions per response.\n5. Aim for single-line commands where possible.\n6.You can use EOF to create multi-line executable files.",
    "chat_history_enabled": True,
    "max_history_length": 10,
    "auto_env_vars": True,
    "use_dotenv": True,
    "create_env_file": True,
    "env_file_name": ".env"
}

CMD_DELIMITER = "`"

def rich_print(*args, **kwargs):
    """Unified print function with Rich fallback"""
    if RICH_AVAILABLE:
        console.print(*args, **kwargs)
    else:
        text = " ".join(str(arg) for arg in args)
        text = re.sub(r'\[.*?\]', '', text)  # Remove Rich markup
        print(text)

def rich_prompt(message, **kwargs):
    """Unified prompt function with Rich fallback"""
    if RICH_AVAILABLE:
        return Prompt.ask(message, **kwargs)
    else:
        clean_message = re.sub(r'\[.*?\]', '', message)
        default = kwargs.get('default')
        if default:
            user_input = input(f"{clean_message} ({default}): ").strip()
            return user_input if user_input else default
        return input(f"{clean_message}: ").strip()

def confirm_immediate(question: str, default: bool, show_choices: bool) -> bool | str:
    """Auto-parse input to truth values."""
    default_opt = 'y' if default else 'n'
    choice = " [bright_blue][y/n][/bright_blue]" if show_choices else ''
    rich_print(f"{question}{choice} [bold cyan]({default_opt})[/bold cyan]: ", end="")
    try:
        import readchar
        while True:
            try:
                key = readchar.readchar()
                if key == '\x03':
                    raise KeyboardInterrupt
                if key in ('\r', '\n'):
                    print(default_opt)
                    return default
                if key.lower() in 'yn':
                    print(key.lower())
                    return key.lower() == 'y'
            except Exception as e:
                console.print(f"\n[red]Error reading input: {e}[/red]")
                sys.exit(1)
    except ImportError:
        return "FallBack"

def rich_confirm(message, default=True, show_choices=False):
    """Unified confirm function with Rich fallback"""
    if RICH_AVAILABLE:
        truthval = confirm_immediate(message, default=default, show_choices=show_choices)
        if truthval == "Fallback": 
            return Confirm.ask(message, default=default, show_choices=show_choices)
        else:
            return truthval
    else:
        clean_message = re.sub(r'\[.*?\]', '', message)
        default_text = "Y/n" if default else "y/N"
        response = input(f"{clean_message} ({default_text}): ").strip().lower()
        if not response:
            return default
        return response in ['y', 'yes']

def rich_int_prompt(message, **kwargs):
    """Unified integer prompt function with Rich fallback"""
    if RICH_AVAILABLE:
        return IntPrompt.ask(message, **kwargs)
    else:
        clean_message = re.sub(r'\[.*?\]', '', message)
        choices = kwargs.get('choices', [])
        default = kwargs.get('default')
        
        while True:
            try:
                if default:
                    response = input(f"{clean_message} ({default}): ").strip()
                    if not response:
                        return int(default)
                else:
                    response = input(f"{clean_message}: ").strip()
                
                value = int(response)
                if choices and str(value) not in choices:
                    print(f"Please enter one of: {', '.join(choices)}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number.")

def show_progress(task_name, duration=2):
    """Show progress with Rich or fallback"""
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(task_name, total=100)
            for i in range(100):
                progress.update(task, advance=1)
                time.sleep(duration / 100)
    else:
        print(f"{task_name}...", end="", flush=True)
        for i in range(10):
            print(".", end="", flush=True)
            time.sleep(duration / 10)
        print(" Done!")

def get_env_file_path(config_dir, env_file_name=".env"):
    """Get the path to the .env file in the configuration directory"""
    if not config_dir:
        return env_file_name
    return os.path.join(config_dir, env_file_name)

def load_env_file(config_dir, env_file_name=".env"):
    """Load environment variables from .env file if available"""
    if not DOTENV_AVAILABLE:
        return {}
    
    env_file_path = get_env_file_path(config_dir, env_file_name)
    
    if os.path.exists(env_file_path):
        try:
            # Load the .env file
            load_dotenv(env_file_path, override=True)
            # Also return the values as a dict for reference
            env_values = dotenv_values(env_file_path)
            if "--debug" in sys.argv: rich_print(f"[green]Loaded .env file: [bright_blue]{env_file_path}[/bright_blue][/green]")
            return env_values
        except Exception as e:
            rich_print(f"[yellow]Warning: Could not load .env file {env_file_path}: {e}[/yellow]")
            return {}
    
    return {}

def create_env_file(config_dir, env_values, env_file_name=".env"):
    """Create or update .env file with provided values"""
    if not DOTENV_AVAILABLE:
        rich_print("[yellow]Warning: python-dotenv not available. Cannot create .env file.[/yellow]")
        return False
    
    env_file_path = get_env_file_path(config_dir, env_file_name)
    
    try:
        # Ensure config directory exists
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        
        # Create or update .env file
        for key, value in env_values.items():
            if value:  # Only set non-empty values
                set_key(env_file_path, key, value)
        
        rich_print(f"[bold green]Environment variables saved to:[/bold green] [bright_blue]{env_file_path}[/bright_blue]")
        return True
    except Exception as e:
        rich_print(f"[bold red]Error creating .env file:[/bold red] [bright_blue]{e}[/bright_blue]")
        return False

def update_env_file(config_dir, key, value, env_file_name=".env"):
    """Update a specific key in the .env file"""
    if not DOTENV_AVAILABLE:
        return False
    
    env_file_path = get_env_file_path(config_dir, env_file_name)
    
    try:
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        
        set_key(env_file_path, key, value)
        return True
    except Exception as e:
        rich_print(f"[yellow]Warning: Could not update .env file: {e}[/yellow]")
        return False

def check_env_vars(config_dir=None, use_dotenv=True, env_file_name=".env"):
    """Enhanced environment variable checking with .env support"""
    env_updates = {}
    
    # Load .env file first if enabled
    if use_dotenv and DOTENV_AVAILABLE and config_dir:
        env_values = load_env_file(config_dir, env_file_name)
    
    
    # Check for model
    model = os.getenv('MODEL')
    if model:
        env_updates['model'] = model
        source = ".env file" if use_dotenv and DOTENV_AVAILABLE and config_dir and os.path.exists(get_env_file_path(config_dir, env_file_name)) else "environment"
        if "--debug" in sys.argv: rich_print(f"[green]Using model from [cyan]{source}[/cyan]: [bright_blue]{model}[/bright_blue][/green]")
    
    # Check for provider
    provider = os.getenv('PROVIDER')
    if provider:
        env_updates['provider'] = provider
        source = ".env file" if use_dotenv and DOTENV_AVAILABLE and config_dir and os.path.exists(get_env_file_path(config_dir, env_file_name)) else "environment"
        if "--debug" in sys.argv: rich_print(f"[green]Using provider from [cyan]{source}[/cyan]: [bright_blue]{provider}[/bright_blue][/green]")

    # API key detection logic
    api_key_to_use = None
    key_variable_name_used = None

    # Define ALIASCT locally, similar to apiconf, for provider name to API service name mapping
    LOCAL_ALIASCT = {
        "google_gemini": "gemini",
        "anthropic": "claude", 
        "gitazure": "github",
    }

    current_provider_from_env = os.getenv('PROVIDER') # This is the provider from env/.env

    # 1. Try provider-specific API key
    if current_provider_from_env:
        provider_api_name = LOCAL_ALIASCT.get(current_provider_from_env, current_provider_from_env)
        specific_api_key_var = f"{provider_api_name.upper()}_API_KEY"
        
        api_key_value = os.getenv(specific_api_key_var)
        if api_key_value:
            api_key_to_use = api_key_value
            key_variable_name_used = specific_api_key_var

    # 2. If not found, try generic API_KEY
    if not api_key_to_use:
        api_key_value = os.getenv('API_KEY')
        if api_key_value:
            api_key_to_use = api_key_value
            key_variable_name_used = 'API_KEY'

    # 3. Fallback: If no specific or generic key found, iterate through other common keys
    if not api_key_to_use:
        fallback_candidates = [
            'OPENAI_API_KEY', 'GEMINI_API_KEY', 'GROQ_API_KEY', 
            'OPENROUTER_API_KEY', 'AZURE_API_KEY', 'GITHUB_API_KEY', 
            'ANTHROPIC_API_KEY', 'CLAUDE_API_KEY' 
        ]
        
        attempted_specific_key_for_fallback_check = None
        if current_provider_from_env:
             # Determine what the specific key *would have been named* to avoid re-checking if it was already tried
             provider_api_name_for_fallback = LOCAL_ALIASCT.get(current_provider_from_env, current_provider_from_env)
             attempted_specific_key_for_fallback_check = f"{provider_api_name_for_fallback.upper()}_API_KEY"

        for var_name in fallback_candidates:
            if attempted_specific_key_for_fallback_check and var_name == attempted_specific_key_for_fallback_check:
                continue # Already tried this as the provider-specific key

            api_key_value = os.getenv(var_name)
            if api_key_value:
                api_key_to_use = api_key_value
                key_variable_name_used = var_name
                break # Found one

    if api_key_to_use:
        env_updates['api_key'] = api_key_to_use
        # Determine source - this logic relies on .env file existing and being configured for use
        source = ".env file" if use_dotenv and DOTENV_AVAILABLE and config_dir and os.path.exists(get_env_file_path(config_dir, env_file_name)) else "environment"
        if "--debug" in sys.argv: rich_print(f"[green]Using API key from [cyan]{source}[/cyan]: [bright_blue]{key_variable_name_used}[/bright_blue][/green]")

    # Check for API endpoint
    endpoint = os.getenv('API_ENDPOINT')
    if endpoint:
        env_updates['api_endpoint'] = endpoint
        source = ".env file" if use_dotenv and DOTENV_AVAILABLE and config_dir and os.path.exists(get_env_file_path(config_dir, env_file_name)) else "environment"
        if "--debug" in sys.argv: rich_print(f"[green]Using API endpoint from [cyan]{source}[/cyan][/green]")
    
    return env_updates

def get_os_info():
    """Get system type, architecture and installed package managers."""
    osinfo = summarize_pkg_managers()
    if 'BSD' in osinfo[1]: 
        osinfo[1] = 'BSD'
    return osinfo[1], osinfo[2] + ' ' + osinfo[0]

def load_chat_history(config_dir):
    """Load chat history from file"""
    history_file = os.path.join(config_dir, "chat_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return []
    return []

def save_chat_history(config_dir, history):
    """Save chat history to file"""
    history_file = os.path.join(config_dir, "chat_history.json")
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        rich_print(f"[yellow]Warning: Could not save chat history: {e}[/yellow]")

def add_to_history(history, user_input, assistant_response, max_length):
    """Add interaction to chat history"""
    history.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "assistant": assistant_response
    })
    
    # Keep only the last max_length entries
    if len(history) > max_length:
        history = history[-max_length:]
    
    return history

def clear_history():
    """Clear the chat history file"""
    args = parse_arguments() # Need args to get config path
    config = load_config(args.config) # Need config to get config_dir
    config_dir = config.get('_config_dir')

    if not config_dir:
        rich_print("[red]Error: Could not determine configuration directory to clear history.[/red]")
        return

    history_file = os.path.join(config_dir, "chat_history.json")
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
            rich_print(f"[green]Chat history file deleted: [bright_blue]{history_file}[/bright_blue][/green]")
        except Exception as e:
            rich_print(f"[red]Error deleting chat history file {history_file}: {e}[/red]")
    else:
        rich_print("[yellow]No chat history file found to delete.[/yellow]")

def create_default_installation(config_dir=None, install_dir=None):
    """Creates default installation with enhanced Rich styling and .env support"""
    flag_config = config_dir
    
    if RICH_AVAILABLE:
        rich_print(Panel.fit(
            "[bold blue]ShellSense Enhanced Installation[/bold blue]",
            border_style="blue"
        ))
    
    if flag_config == 'config' or not install_dir:
        os_system = get_os_info()[0]
        if os_system == "Windows":
            config_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "shellsense")
            install_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Microsoft", "WindowsApps")
        elif os_system == "Darwin":
            config_dir = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "shellsense")
            install_dir = "/usr/local/bin"
        elif os_system in ("FreeBSD", "Linux"):
            config_dir = os.path.join(os.path.expanduser("~"), ".config", "shellsense")
            install_dir = "/usr/local/bin"
        else:
            rich_print("[red]Unsupported OS for installation.[/red]")
            return

    try:
        os.makedirs(config_dir, exist_ok=True)
        if not os.path.exists(config_dir):
            show_progress("Creating configuration directory")
    except Exception as e:
        rich_print(f"[red]Error creating config directory: {e}[/red]")
        config_dir = input("Enter custom directory path: ")
        return

    if flag_config == 'config':
        config_path = os.path.join(config_dir, "config.json")
        
        try:
            confirmwrite = None
            if os.path.exists(config_path): 
                confirmwrite = rich_confirm("Configuration file already exists, overwrite?")
            if confirmwrite or not os.path.exists(config_path):
                with open(config_path, "w") as config_file:
                    json.dump(DEFAULT_CONFIG, config_file, indent=4)
                rich_print(f"[green]Configuration written to: {config_path}[/green]")
                
                # Offer to create .env file template if dotenv is available
                if DOTENV_AVAILABLE and rich_confirm("Create .env file template for environment variables?", default=False):
                    env_template = {
                        "# ShellSense Environment Variables": "",
                        "# API Keys": "",
                        "API_KEY": "",
                        "OPENAI_API_KEY": "",
                        "GEMINI_API_KEY": "",
                        "GROQ_API_KEY": "",
                        "OPENROUTER_API_KEY": "",
                        "AZURE_API_KEY": "",
                        "GITHUB_API_KEY": "",
                        "ANTHROPIC_API_KEY": "",
                        "CLAUDE_API_KEY": "",
                        "# Configuration": "",
                        "MODEL": "",
                        "PROVIDER": "",
                        "API_ENDPOINT": ""
                    }
                    
                    env_file_path = get_env_file_path(config_dir, ".env")
                    with open(env_file_path, 'w') as env_file:
                        for key, value in env_template.items():
                            if key.startswith('#'):
                                env_file.write(f"{key}\n")
                            else:
                                env_file.write(f"{key}={value}\n")
                    
                    rich_print(f"[green].env template created: {env_file_path}[/green]")
                    rich_print("[cyan]Edit the .env file to add your API keys and preferences.[/cyan]")
            else:
                rich_print("[yellow]Using existing configuration.[/yellow]")
        except Exception as e:
            rich_print(f"[red]Error writing config file: {e}[/red]")

    if flag_config != 'config':
        try:
            if not os.path.isdir(install_dir):
                os.makedirs(install_dir, exist_ok=True)
    
            current_script = os.path.abspath(__file__)
            target_file = os.path.join(install_dir, os.path.basename(current_script))
            
            if os.access(install_dir, os.W_OK):
                if not os.path.exists(target_file if os.path.exists(target_file) else target_file.strip('.py')):
                    show_progress("Installing ShellSense Enhanced")
                else:
                    show_progress("Updating ShellSense Enhanced")
                shutil.copy2(current_script, target_file)
                if get_os_info()[0] == "Windows":
                    os.system('cmd /c "setx /M PATHEXT \\"%PATHEXT%;.PY\\""')
                    os.chmod(target_file, 0o755)
                else:
                    os.chmod(target_file, 0o755)
                    os.system(f"mv {target_file} {target_file.strip('.py')}")
                rich_print(f"[green]ShellSense Enhanced installed to: [bright_blue]{target_file.strip('.py') if os.path.exists(target_file.strip('.py')) else target_file}[/bright_blue][/green]")
                rich_print("[cyan]Make sure it is in your PATH.[/cyan]")
            else:
                rich_print(f"[red]Write permission denied for: {install_dir}[/red]")
                rich_print("[yellow]Run as administrator or with sudo.[/yellow]")
                sys.exit(1)
        except Exception as e:
            rich_print(f"[red]Error during installation: {e}[/red]")

def load_config(config_path="config.json"):
    """Enhanced configuration loading with .env support"""
    config = DEFAULT_CONFIG.copy()
    config_dir = None
    
    if os.path.exists(config_path) and config_path != "default":
        try:
            with open(config_path, 'r') as f:
                config_from_file = json.load(f)
            config.update(config_from_file)
            config_dir = os.path.dirname(config_path)
            rich_print(f"[cyan]Loaded configuration from: [bright_blue]{config_path}[/bright_blue][/cyan]")
        except json.JSONDecodeError:
            rich_print(f"[red]Invalid JSON in config file: {config_path}. Using defaults.[/red]")
        except Exception as e:
            rich_print(f"[red]Could not read config file {config_path}: {e}[/red]")
    elif config_path == "default":
        rich_print("[cyan]Using default configuration.[/cyan]")
    else:
        rich_print(f"[yellow]Config file '{config_path}' not found. Using defaults.[/yellow]")
        # Try to get config directory from path even if file doesn't exist
        if config_path != "config.json":
            config_dir = os.path.dirname(config_path)

    # Override with environment variables if auto_env_vars is enabled
    if config.get('auto_env_vars', True):
        use_dotenv = config.get('use_dotenv', True)
        env_file_name = config.get('env_file_name', '.env')
        env_updates = check_env_vars(config_dir, use_dotenv, env_file_name)
        config.update(env_updates)

    # Store config_dir in config for later use
    config['_config_dir'] = config_dir

    # Validate essential keys
    essential_keys = ["api_key", "model", "api_endpoint", "system_prompt_template"]
    if config["provider"] not in ("pollinations", "ollama"):
        if not all(key in config and config[key] for key in essential_keys):
            rich_print("[red]Configuration missing essential keys. Cannot proceed.[/red]")
            sys.exit(1)

    return config

def parse_arguments():
    """Enhanced argument parsing with .env options"""
    def get_default_config_path():
        os_system = get_os_info()[0]
        if os_system == "Windows":
            return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "shellsense", "config.json")
        elif os_system == "Darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "shellsense", "config.json")
        elif os_system in ("FreeBSD", "Linux"):
            return os.path.join(os.path.expanduser("~"), ".config", "shellsense", "config.json")
        else:
            return "config.json"

    tokens = sys.argv[1:]
    if any(token.startswith("-") for token in tokens):
        parser = argparse.ArgumentParser(
            description="ShellSense Enhanced: AI-powered command-line assistant with .env support"
        )
        parser.add_argument("question", nargs="?", default=None,
                            help="The question or task description to send to AI")
        parser.add_argument("--config", default=get_default_config_path(),
                            help="Path to configuration JSON file")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--install", nargs="?",
                            help="Install/Update system-wide with --install flag (requires admin privileges) or use `--install config` to create a config file or `--install . <PATH>` to use a specific install directory")
        parser.add_argument("--apiconfig", action="store_true", help="Configure API settings")
        parser.add_argument("--no-history", action="store_true",
                            help="Disable chat history for this session")
        parser.add_argument("--no-dotenv", action="store_true",
                            help="Disable .env file loading for this session")
        parser.add_argument("--clear-history", action="store_true",
                            help="Clear chat history and exit")
        parser.add_argument("--create-env", action="store_true",
                            help="Create .env file template and exit")
        return parser.parse_args()
    else:
        class SimpleArgs:
            pass
        simple = SimpleArgs()
        simple.question = " ".join(tokens) if tokens else None
        simple.config = get_default_config_path()
        simple.debug = False
        simple.install = None
        simple.apiconfig = False
        simple.no_history = False
        simple.no_dotenv = False
        simple.clear_history = False
        simple.create_env = False
        return simple

def apiconf():
    """Enhanced API configuration with .env file support"""
    args = parse_arguments()
    config = load_config(args.config)
    config_dir = config.get('_config_dir')

    if RICH_AVAILABLE:
        rich_print(Panel.fit(
            "[bold green]ShellSense Enhanced API Configuration[/bold green]",
            border_style="green"
        ))

    provider_options = [
        ("OpenAI", "red"),
        ("Gemini", "yellow"), 
        ("Groq", "bright_green"),
        ("OpenRouter", "bright_blue"),
        ("Azure", "magenta"),
        ("GitHub", "cyan"),
        ("Pollinations", "bright_black"),
        ("Ollama", "bright_green"),
        ("Claude", "bright_red")
    ]

    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Provider", style="white")
        
        for i, (name, color) in enumerate(provider_options, 1):
            table.add_row(str(i), f"[{color}]{name}[/{color}]")
        
        console.print(table)
    else:
        rich_print("\n[bold green]Choose a provider:[/bold green]")
        for i, (name, _) in enumerate(provider_options, 1):
            print(f"{i}. {name}")

    set_input = rich_int_prompt(
        "[bold green]Select provider number [bold cyan]1 - 9[/bold cyan][/bold green]",
        default=2,
        choices=[str(i) for i in range(1, len(provider_options) + 1)],
        show_choices=False
    )
    
    providers = ['openai', 'google_gemini', 'groq', 'openrouter', 'azure', 
                'gitazure', 'pollinations', 'ollama', 'anthropic']

    ALIASCT = {
        "google_gemini": "gemini",
        "anthropic": "claude", 
        "gitazure": "github",
    }

    provider_api = ALIASCT.get(providers[set_input-1], providers[set_input-1])

    # API key handling
    if set_input-1 in (6, 7):  # pollinations, ollama
        api_key = None
    else:
        default_api_prefixes = {
            "openai": "sk-",
            "google_gemini": "AIza",
            "groq": "gsk_",
            "openrouter": "sk-or-",
            "azure": "^.",
            "gitazure": "ghp",
            "anthropic": "sk-ant"
        }
        
        provider_key = providers[set_input-1]
        default_prefix = default_api_prefixes.get(provider_key, "")
        
        if config.get("api_key", "").startswith(default_prefix):
            short_config_key = config["api_key"][:10] + "..." if config["api_key"] else "Not found"
            use_default = rich_confirm(f"[bold green]Use API key from config? [dim]({short_config_key})[/dim][/bold green]", default=True)
        else:
            use_default = False
            
        if use_default:
            api_key = config["api_key"]
        else:
            default_key = os.getenv(f"{provider_api.upper()}_API_KEY")
            short_default = default_key[:10] + "..." if default_key else "Not found"
            api_key = rich_prompt(
                f"[bold green]API key for[/bold green] [cyan]{providers[set_input-1]}[/cyan] (Default: [dim]{short_default}[/dim])",
                default=default_key,
                show_default=False
            )

    if not (set_input-1 in (6, 7)) and (not api_key or len(api_key) < 5):
        api_key = os.getenv(f"{provider_api.upper()}_API_KEY")
        if api_key:
            rich_print(f"[yellow]Using {provider_api.upper()}_API_KEY from environment[/yellow]")
        else:
            rich_print("[red]Failed to retrieve API key. Please provide a valid key.[/red]")
            sys.exit(1)

    # Model selection
    model = os.getenv("MODEL")
    if model:
        rich_print("[green]Using MODEL from environment.[/green]")
    else:
        show_progress("Fetching available models")
        model = main_model(providers[set_input-1], api_key)

    # API endpoint for specific providers
    api_endpoint = None
    if set_input-1 in (4, 5, 7):  # azure, gitazure, ollama
        api_endpoint = rich_prompt(f"API Endpoint for {providers[set_input-1]}")
        if not api_endpoint:
            api_endpoint = os.getenv("API_ENDPOINT")

    # Update configuration
    config["api_key"] = api_key
    config["model"] = model
    config["provider"] = providers[set_input-1]
    if api_endpoint:
        config["api_endpoint"] = api_endpoint

    # Ask about .env file preference
    if DOTENV_AVAILABLE and config_dir:
        save_to_env = True # Local variable loading has some issues / rich_confirm("[bold green]Save configuration to .env file?[/bold green]", default=True)
        if save_to_env:
            env_values = {}
            if api_key:
                env_key = f"{provider_api.upper()}_API_KEY"
                env_values[env_key] = api_key
            if model:
                env_values["MODEL"] = model
            env_values["PROVIDER"] = providers[set_input-1]
            if api_endpoint:
                env_values["API_ENDPOINT"] = api_endpoint
            
            env_file_name = config.get('env_file_name', '.env')
            if create_env_file(config_dir, env_values, env_file_name):
                rich_print("[green]Configuration saved to .env file.[/green]")

    # Save configuration to JSON file
    if args.config in ("default", '', ' ', None) or not os.path.exists(args.config):
        DEFAULT_CONFIG.update(config)
        rich_print("[yellow]DEFAULT_CONFIG updated (no existing config file found).[/yellow]")
    else:
        try:
            # Remove internal keys before saving
            config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
            with open(args.config, "w") as config_file:
                json.dump(config_to_save, config_file, indent=4)
            rich_print(f"[green]Configuration saved to: {args.config}[/green]")
        except Exception as e:
            rich_print(f"[red]Failed to update configuration: {e}[/red]")

def call_ai_api(prompt_text, config, os_name, os_arch, history=None):
    """Enhanced API call with environment variable checking"""
    # Check environment variables before each call
    if config.get('auto_env_vars', True):
        config_dir = config.get('_config_dir')
        use_dotenv = config.get('use_dotenv', True) and not args.no_dotenv
        env_file_name = config.get('env_file_name', '.env')
        env_updates = check_env_vars(config_dir, use_dotenv, env_file_name)
        config.update(env_updates)
    
    api_key = config["api_key"]
    model = config["model"] 
    provider = config["provider"]
    system_prompt_template = config["system_prompt_template"]

    try:
        system_prompt = system_prompt_template.format(os_name=os_name, os_arch=os_arch)
    except KeyError as e:
        rich_print(f"[red]Invalid placeholder {e} in system_prompt_template[/red]")
        return None

    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if enabled
        if history and config.get('chat_history_enabled', True):
            for entry in history[-3:]:  # Last 3 interactions for context
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["assistant"]})
        
        messages.append({"role": "user", "content": prompt_text})
        
        if RICH_AVAILABLE:
            with console.status(f"[bold green]Calling {provider.title()} API...", spinner="dots"):
                response = chat_completion(provider, api_key, model, messages)
        else:
            print(f"Calling {provider.title()} API...")
            response = chat_completion(provider, api_key, model, messages)
            
        if response.get("choices") and len(response["choices"]) > 0:
            candidate = response["choices"][0]
            finish_reason = candidate.get("finish_reason")
            if finish_reason and finish_reason != "stop":
                rich_print(f"[yellow]API response finished with reason:[/yellow] [red]{finish_reason}[/red]")
            return candidate["message"].get("content", "")
        else:
            rich_print(f"[red]Unexpected API response format:[/red] [bright_blue]{json.dumps(response)}[/bright_blue]")
            return None

    except TimeoutError:
        rich_print("[red]API request timed out.[/red]")
        return None
    except Exception as e:
        rich_print(f"[red]API request failed:[/red] [bright_blue]{e}[/bright_blue]")
        return None

def extract_commands(api_response):
    """Enhanced command extraction with better error handling"""
    if not api_response:
        return []

    raw_commands = re.findall(r'`(.*?)`', api_response, re.DOTALL)
    cleaned_commands = []
    language_prefixes = ('bash\n', 'sh\n', 'shell\n', 'powershell\n', 'zsh\n',
                         'bash ', 'sh ', 'shell ', 'powershell ', 'zsh ')

    for cmd in raw_commands:
        cmd = cmd.strip()
        lower_cmd_start = cmd.lower()
        for prefix in language_prefixes:
            if lower_cmd_start.startswith(prefix):
                cmd = cmd[len(prefix):].strip()
                break
        if cmd:
            cleaned_commands.append(cmd)

    return cleaned_commands

def execute_command(command):
    """Enhanced command execution with Rich styling"""
    if RICH_AVAILABLE:
        panel = Panel(
            f"[bold white]{command}[/bold white]",
            title="[cyan]Executing Command[/cyan]",
            border_style="cyan"
        )
        console.print(panel)
    else:
        print(f"\nExecuting: {command}\n")
        print("=" * 50)

    try:
        global prefix
        
        if get_os_info()[0] != "Windows":
            log_file = "/tmp/shellsense.log"
            if not os.path.exists(os.path.dirname(log_file)) or not os.access(os.path.dirname(log_file), os.W_OK):
                rich_print("[yellow]Warning: /tmp not available, using local directory[/yellow]")
                os.makedirs('tmp', exist_ok=True)
                log_file = "tmp/shellsense.log"
                prefix = 'tmp'
            else:
                prefix = '/tmp'

            if os.path.exists(log_file):
                i = 1
                while os.path.exists(f"{prefix}/shellsense_{i}.log"):
                    i += 1
                log_file = f"{prefix}/shellsense_{i}.log"

            os.system("{ " + command + "; echo $?; } 2>&1 | tee -a " + log_file)
        else:
            tmpdir = os.getenv('TEMP', 'tmp').replace('\\', '/')
            log_file = os.path.join(tmpdir, "shellsense.log")
            
            if not os.path.exists(tmpdir):
                os.makedirs('tmp', exist_ok=True)
                prefix = 'tmp'
            else:
                prefix = tmpdir.strip('/\\')

            if os.path.exists(log_file):
                i = 1
                while os.path.exists(f"{prefix}/shellsense_{i}.log"):
                    i += 1
                log_file = f"{prefix}/shellsense_{i}.log"

            if 'PSModulePath' in os.environ:
                os.system('& { '+command+'; Write-Output $LASTEXITCODE } *>&1 | Tee-Object -FilePath "'+log_file+'" -Append')
            else:
                os.system('powershell -NoProfile -Command "& { '+command+'; Write-Output $LASTEXITCODE } *>&1 | Tee-Object -FilePath "'+log_file+'" -Append"')

        with open(log_file, 'r') as f:
            logfile = f.read().strip()
        
        exit_code = int(logfile.split('\n')[-1])
        output_command = logfile.strip(f"\n{exit_code}")

        if exit_code == 0:
            rich_print("\n[bold green]✓ Execution Successful[/bold green]")
            return 0, output_command
        else:
            rich_print(f"\n[bold red]✗ Execution Failed (Exit Code: {exit_code})[/bold red]")
            return exit_code, output_command

    except KeyboardInterrupt:
        rich_print("\n[bold yellow]⚠ Execution Interrupted (Ctrl+C)[/bold yellow]")
        return 130, None
    except Exception as e:
        rich_print(f"\n[bold red]Error: Failed to execute command: {e}[/bold red]")
        return -1, None

def display_welcome():
    """Display welcome message with Rich styling and .env info"""
    if RICH_AVAILABLE:
        welcome_text = Text()
        welcome_text.append("ShellSense Enhanced", style="bold blue")
        welcome_text.append(" - AI-Powered Terminal Assistant", style="white")
        
        if DOTENV_AVAILABLE:
            welcome_text.append("\n", style="white")
            welcome_text.append("✨ .env File Support Enabled", style="green")
        
        panel = Panel(
            Align.center(welcome_text),
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
        console.rule("[bold blue]Session Start[/bold blue]")
    else:
        print("="*60)
        print("     ShellSense Enhanced - AI-Powered Terminal Assistant")
        if DOTENV_AVAILABLE:
            print("                ✨ .env File Support Enabled")
        print("="*60)

def create_env_template(config_dir):
    """Create .env template file"""
    if not DOTENV_AVAILABLE:
        rich_print("[red]Error: python-dotenv not available. Cannot create .env file.[/red]")
        return False
    
    env_template_content = """# ShellSense Enhanced Environment Variables
# Edit this file to configure your API keys and preferences

# API Keys (choose one based on your provider)
API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
AZURE_API_KEY=
GITHUB_API_KEY=
ANTHROPIC_API_KEY=
CLAUDE_API_KEY=

# Model Configuration
MODEL=
PROVIDER=
API_ENDPOINT=

# Usage Examples:
# GEMINI_API_KEY=AIzaSyDhZliXXalXnzQT6amWoIz9W4wXkhZ4raE
# MODEL=gemini-2.0-flash
# PROVIDER=google_gemini
"""
    
    try:
        os.makedirs(config_dir, exist_ok=True)
        env_file_path = get_env_file_path(config_dir, ".env")
        
        if os.path.exists(env_file_path):
            if not rich_confirm(f".env file already exists at {env_file_path}. Overwrite?", default=False):
                return False
        
        with open(env_file_path, 'w') as f:
            f.write(env_template_content)
        
        rich_print(f"[green].env template created: {env_file_path}[/green]")
        rich_print("[cyan]Edit the .env file to add your API keys and preferences.[/cyan]")
        return True
    except Exception as e:
        rich_print(f"[red]Error creating .env template: {e}[/red]")
        return False

def main():
    """Enhanced main function with .env support and chat history"""
    global args
    args = parse_arguments()
    
    # Handle --create-env flag
    if args.create_env:
        config_dir = os.path.dirname(args.config) if args.config != "default" else None
        if not config_dir:
            config_dir = os.path.join(os.path.expanduser("~"), ".config", "shellsense")
        create_env_template(config_dir)
        return
    
    config = load_config(args.config)
    
    # Display welcome
    display_welcome()

    # Get OS info
    os_name, os_arch = get_os_info()
    if os_name == "Unknown":
        rich_print("[yellow]Warning: Could not determine OS[/yellow]")
    else:
        if args.debug:
            rich_print(f"[green]OS: {os_name}[/green]")
            rich_print(f"[green]Architecture: {os_arch}[/green]")

    rich_print(f"[bold green]Using model:[/bold green] [bright_blue]{config.get('model')}[/bright_blue]")
    
    # Show .env status
    config_dir = config.get('_config_dir')
    if DOTENV_AVAILABLE and config.get('use_dotenv', True) and not args.no_dotenv and config_dir:
        env_file_path = get_env_file_path(config_dir, config.get('env_file_name', '.env'))
        if os.path.exists(env_file_path):
            rich_print(f"[dim cyan].env file loaded from: {env_file_path}[/dim cyan]")
        else:
            rich_print(f"[yellow].env file not found at: {env_file_path}[/yellow]")
    
    # Initialize chat history
    history = []
    
    if config.get('chat_history_enabled', True) and not args.no_history and config_dir:
        history = load_chat_history(config_dir)
        if history:
            rich_print(f"[cyan]Loaded {len(history)} previous interactions[/cyan]")

    current_question = args.question
    if not current_question:
        rich_print("[red]Error: No question provided[/red]")
        sys.exit(1)
    
    last_prompt = None
    conversation_active = True

    while conversation_active:
        try:
            if last_prompt:
                prompt_to_send = last_prompt
                if args.debug:
                    rich_print(f"[dim]Using retry prompt: {prompt_to_send[:100]}...[/dim]")
            else:
                prompt_to_send = current_question
                rich_print(f"\n[bold]Question:[/bold] [italic]{current_question}[/italic]")

            # API call with history
            api_response_text = call_ai_api(prompt_to_send, config, os_name, os_arch, history)
            last_prompt = prompt_to_send

            if not api_response_text:
                rich_print("[red]Failed to get valid API response[/red]")
                if rich_confirm("[bold white]Try again with the same prompt?[/bold white]", default=True):
                    continue
                else:
                    break

            if args.debug:
                rich_print(f"[dim]Raw response:\n{api_response_text}[/dim]")

            # Add to history
            if config.get('chat_history_enabled', True) and not args.no_history:
                history = add_to_history(
                    history, 
                    prompt_to_send, 
                    api_response_text,
                    config.get('max_history_length', 10)
                )

            suggested_commands = extract_commands(api_response_text)

            if not suggested_commands:
                rich_print(f"[red]No commands found in backticks[/red]")
                rich_print(f"Full response:\n{api_response_text}")
                if rich_confirm("Try again with clarification?", default=True):
                    current_question = f"{current_question}\n\n(Please ensure the command is wrapped in backticks: `command`)"
                    last_prompt = None
                    continue
                else:
                    if rich_confirm("Modify the question?", default=False):
                        current_question = rich_prompt("Enter modified question", default=current_question)
                        last_prompt = None
                        continue
                    else:
                        break

            # Command selection
            chosen_command = None
            if len(suggested_commands) == 1:
                chosen_command = suggested_commands[0]
                if RICH_AVAILABLE:
                    panel = Panel(
                        f"[bold magenta]{chosen_command}[/bold magenta]",
                        title="[green]Suggested Command[/green]",
                        border_style="green"
                    )
                    console.print(panel)
                else:
                    print(f"\nSuggested command: {chosen_command}")
            else:
                rich_print("\n[yellow]Multiple commands found:[/yellow]")
                if RICH_AVAILABLE:
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Index", style="cyan", width=8)
                    table.add_column("Command", style="white")
                    
                    for i, cmd in enumerate(suggested_commands):
                        table.add_row(str(i), f"[magenta]{cmd}[/magenta]")
                    
                    console.print(table)
                else:
                    for i, cmd in enumerate(suggested_commands):
                        print(f"  [{i}]: {cmd}")

                while True:
                    try:
                        choice = rich_int_prompt(
                            f"Choose command index [0-{len(suggested_commands)-1}]",
                            choices=[str(i) for i in range(len(suggested_commands))],
                            show_choices=False
                        )
                        chosen_command = suggested_commands[choice]
                        break
                    except (ValueError, IndexError):
                        rich_print("[red]Invalid choice[/red]")
                    except KeyboardInterrupt:
                        rich_print("\n[yellow]Selection cancelled[/yellow]")
                        break
                
                if not chosen_command:
                    last_prompt = None
                    continue

                rich_print(f"Selected: [bold magenta]{chosen_command}[/bold magenta]")

            # Action choice
            action_input = rich_prompt(
                "\n[bold green]Choose action:[/bold green] [purple]E[/purple]xecute, [purple]M[/purple]odify, [purple]R[/purple]evise, [purple]Q[/purple]uit",
                default="e",
                show_choices=False
            ).lower()

            if action_input == "e":
                if rich_confirm(f"[bold green]Execute:[/bold green] [cyan]'{chosen_command}'[/cyan]?", default=True):
                    exit_code, output_command = execute_command(chosen_command)
                    if exit_code == 0:
                        rich_print("[bold green]✓ Command completed successfully[/bold green]")
                        conversation_active = False
                    else:
                        rich_print(f"[yellow]Command failed (Exit Code: {exit_code})[/yellow]")
                        if rich_confirm("[bold green]Ask AI for a fix?[/bold green]", default=True):
                            error_prompt = (
                                f"The following command failed with exit code {exit_code}:\\n```\\n{chosen_command}\\n```\\n"
                                f"Original request: {current_question}\\n"
                                f"Error output: {output_command}\\n\\n"
                                f"Please provide a corrected command in backticks."
                            )
                            user_feedback_comment = rich_prompt(
                                "[bold green]Provide additional feedback to help the AI improve the command[/bold green] [bold cyan](Default: skip)[/bold cyan]",
                                default="",
                                show_default=False
                            )
                            if user_feedback_comment:
                                error_prompt += f"\\n\\nUser's feedback for improvement: {user_feedback_comment}"
                            
                            current_question = f"Fixing failed command for: {current_question}"[:200]
                            last_prompt = error_prompt
                            continue
                        else:
                            break
                else:
                    rich_print("[yellow]Execution cancelled[/yellow]")
                    continue

            elif action_input == "m":
                modified_command = rich_prompt("Modify command", default=chosen_command)
                if rich_confirm(f"[bold green]Execute modified command:[/bold green] [cyan]'{modified_command}'[/cyan]?", default=True):
                    exit_code, output_command = execute_command(modified_command)
                    if exit_code == 0:
                        rich_print("[bold green]✓ Command completed successfully[/bold green]")
                        conversation_active = False
                    else:
                        rich_print(f"[yellow]Command failed (Exit Code: {exit_code})[/yellow]")
                        if rich_confirm("[bold green]Ask AI for a fix?[/bold green]", default=True):
                            error_prompt = (
                                f"The following command failed with exit code {exit_code}:\\n```\\n{modified_command}\\n```\\n"
                                f"Original request: {current_question}\\n"
                                f"Error output: {output_command}\\n\\n"
                                f"Please provide a corrected command in backticks."
                            )
                            user_feedback_comment = rich_prompt(
                                "[bold green]Provide additional feedback to help the AI improve the command[/bold green] [bold cyan](Default: skip)[/bold cyan]",
                                default="",
                                show_default=False
                            )
                            if user_feedback_comment:
                                error_prompt += f"\\n\\nUser's feedback for improvement: {user_feedback_comment}"

                            current_question = f"Fixing failed command for: {current_question}"[:200] # Use original question context for fix
                            last_prompt = error_prompt
                            continue
                        else:
                            break
                else:
                    rich_print("[yellow]Execution cancelled[/yellow]")
                    continue

            elif action_input == "r":
                rich_print("[yellow]Generating another suggestion...[/yellow]")
                last_prompt = f"{last_prompt}\n\n(Please provide a different command suggestion for the original request.)"
                continue

            elif action_input == "q":
                rich_print("[bold]Exiting ShellSense Enhanced[/bold]")
                break

        except KeyboardInterrupt:
            rich_print("\n[yellow]Session interrupted[/yellow]")
            break

    # Save chat history
    if config.get('chat_history_enabled', True) and not args.no_history and config_dir and history and rich_confirm("[bold green]⌛️ Add to history?[/bold green]", default=True):
        save_chat_history(config_dir, history)
        rich_print(f"[cyan]Saved {len(history)} interactions to history[/cyan]")

    # Cleanup log files
    if 'prefix' in globals():
        log_files = glob.glob(f"{prefix}/shellsense*.log")
        for log_file in log_files:
            try:
                os.remove(log_file)
                if args.debug:
                    rich_print(f"[dim]Cleaned up: {log_file}[/dim]")
            except Exception:
                pass

    if RICH_AVAILABLE:
        console.rule("[bold blue]Session End[/bold blue]")
        rich_print("[bold blue]Thank you for using ShellSense Enhanced![/bold blue]")
    else:
        print("\n" + "="*60)
        print("     Thank you for using ShellSense Enhanced!")
        print("="*60)

if __name__ == "__main__":
    try:
        if '--install' in sys.argv:
            idx = sys.argv.index('--install')
            try:
                config_dir = sys.argv[idx + 1]
            except IndexError:
                config_dir = None
            try:
                install_dir = sys.argv[idx + 2]
            except IndexError:
                install_dir = None
            create_default_installation(config_dir, install_dir)
            sys.exit(0)
        
        if '--apiconfig' in sys.argv:
            apiconf()
        
        if '--clear' in sys.argv:
            clear_history()
            sys.exit(0)
        
        main()
        
    except ImportError as e:
        if "rich" in str(e).lower():
            print("Warning: Rich library not available. Using fallback mode.")
            RICH_AVAILABLE = False
            main()
        elif "dotenv" in str(e).lower():
            print("Info: python-dotenv not available. .env file support disabled.")
            DOTENV_AVAILABLE = False
            main()
        else:
            print(f"Error: Required library not found: {e}")
            print("Please install required dependencies.")
            sys.exit(1)
    except KeyboardInterrupt:
        rich_print("\n[bold yellow]Operation cancelled by user. Exiting.[/bold yellow]")
        sys.exit(0)
