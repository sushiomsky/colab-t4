"""colab-t4: GPU Colab runtime on your Tailnet with an OpenAI-compatible API."""

__version__ = "0.3.0"

from .notebook import (
    DEFAULT_CTX,
    DEFAULT_HOSTNAME,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_QUANT,
    build_notebook,
)
from .provider import ModelSpec, account_rows, available_models, resolve_model

__all__ = [
    "__version__",
    "build_notebook",
    "DEFAULT_MODEL",
    "DEFAULT_HOSTNAME",
    "DEFAULT_PORT",
    "DEFAULT_CTX",
    "DEFAULT_QUANT",
    "ModelSpec",
    "account_rows",
    "available_models",
    "resolve_model",
]
