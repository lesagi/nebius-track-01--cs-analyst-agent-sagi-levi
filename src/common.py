"""Configuration: config.yaml for non-secrets, .env for secrets, one model factory."""

import json
import os
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

with open(Path(__file__).parent.parent / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = os.getenv("BITEXT_DB_PATH") or CONFIG["dataset"]["db_path"]
PROVIDER = os.getenv("MODEL_PROVIDER") or CONFIG["provider"]
PROVIDER_CONFIG = CONFIG["models"][PROVIDER]


def get_model(role: str = "default"):
    """The LLM for a role from the active provider block (see config.yaml models)."""
    name = PROVIDER_CONFIG.get(role) or PROVIDER_CONFIG["default"]
    if PROVIDER == "ollama":
        return init_chat_model(
            model=name,
            model_provider="ollama",
            base_url=PROVIDER_CONFIG["base_url"],
            # ponytail: temperature 0 - routing/planning must not flip-flop between runs
            temperature=0,
        )
    # nebius token factory is OpenAI-compatible
    return init_chat_model(
        model=name,
        model_provider="openai",
        base_url=PROVIDER_CONFIG["base_url"],
        api_key=os.environ["NEBIUS_API_KEY"],
        temperature=0,
        # typical calls run seconds; a stalled connection or a runaway
        # generation should fail fast, not hang for the client's
        # 10-minute default
        timeout=120,
        max_retries=2,
    )


def get_context_tokens() -> int:
    """Context window of the default model: env override, then asking Ollama, then a safe default."""
    if os.getenv("MODEL_CONTEXT_TOKENS"):
        return int(os.getenv("MODEL_CONTEXT_TOKENS"))
    if PROVIDER != "ollama":
        # ponytail: no metadata endpoint on the hosted API - safe default,
        # override with MODEL_CONTEXT_TOKENS
        return 32_000
    try:
        request = urllib.request.Request(
            f"{PROVIDER_CONFIG['base_url']}/api/show",
            data=json.dumps({"model": PROVIDER_CONFIG["default"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            model_info = json.load(response)["model_info"]
        return next(
            int(value)
            for key, value in model_info.items()
            if key.endswith(".context_length")
        )
    except Exception:
        return 32_000


# ponytail: ~4 chars/token - a rough but sufficient ratio for budgeting
CONTEXT_CHARS = get_context_tokens() * 4
