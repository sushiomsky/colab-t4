# Model catalog for the hand-built Colab T4 (Tesla T4, 16 GiB VRAM).
#
# Every entry must fit alongside the others on disk (Colab disk is roomy;
# VRAM is the constraint — only ONE model sits in VRAM at a time, Ollama
# swaps on demand, so disk is the only shared budget).
#
# Serving names (= what clients send as `model` in /v1/chat/completions):
# - `qwen`: the 16K-context serving variant built on-VM from PULL_MODEL
#   via Modelfile (num_ctx 16384). Tool-calling capable (OMP agent loop).
# - every other entry is served under its exact hub name (no variant).
#
# Alias names (= what users type in Telegram `/model <alias>`):
# short, memorable, no colons. Defined as model_aliases in the
# willnotrefuse profile config; they resolve to serving name + router URL.

# pull: exact `ollama pull` ref on the VM.
# serve: model id clients request through the router.
# vram_gb: approx weights size (KV cache needs headroom on top).
# ctx: default context (serving variant only; hub models use their built-in).
# uncensored: true = abliterated / dolphin / wizard (no refusal tuning).
# tools: true = emits native OpenAI tool_calls (safe for agent loops).
# needs_variant: Modelfile rebuild for num_ctx (hub pins 8192, ignored env).
MODELS = [
    {
        "alias": "qwen",
        "pull": "huihui_ai/qwen3-abliterated:14b-v2",
        "serve": "qwen-coder-16k",
        "vram_gb": 9.0,
        "ctx": 16384,
        "uncensored": True,
        "tools": True,
        "needs_variant": True,
        "blurb": "Qwen3 14B abliterated, 16K ctx — default, tool-calling, agent-safe",
    },
    {
        "alias": "dolphin",
        "pull": "dolphin3:8b",
        "serve": "dolphin3:8b",
        "vram_gb": 4.9,
        "ctx": 8192,
        "uncensored": True,
        "tools": False,
        "needs_variant": False,
        "blurb": "Dolphin 8B uncensored — fast, small, chat-first",
    },
    {
        "alias": "mistral",
        "pull": "dolphin-mistral:7b",
        "serve": "dolphin-mistral:7b",
        "vram_gb": 4.4,
        "ctx": 8192,
        "uncensored": True,
        "tools": False,
        "needs_variant": False,
        "blurb": "Dolphin Mistral 7B uncensored — fastest, smallest",
    },
    {
        "alias": "wizard",
        "pull": "wizard-vicuna-uncensored:13b-q4_K_M",
        "serve": "wizard-vicuna-uncensored:13b-q4_K_M",
        "vram_gb": 8.0,
        "ctx": 4096,
        "uncensored": True,
        "tools": False,
        "needs_variant": False,
        "blurb": "Wizard Vicuna 13B uncensored Q4 — classic uncensored, larger",
    },
]

# Backwards-compat: default serving model (keeper smoke, router fallback).
DEFAULT_SERVE = "qwen-coder-16k"
DEFAULT_PULL = "huihui_ai/qwen3-abliterated:14b-v2"


def serving_names() -> list[str]:
    return [m["serve"] for m in MODELS]


def pull_refs() -> list[str]:
    """Unique hub refs to `ollama pull` (variant builds need their base too)."""
    seen: list[str] = []
    for m in MODELS:
        if m["pull"] not in seen:
            seen.append(m["pull"])
    return seen


def by_serve(name: str) -> dict | None:
    for m in MODELS:
        if m["serve"] == name:
            return m
    return None
