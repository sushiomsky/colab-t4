"""colab-t4: GPU Colab runtime on your Tailnet with an OpenAI-compatible API."""

__version__ = "0.2.0"

from .notebook import (
    DEFAULT_CTX,
    DEFAULT_HOSTNAME,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_QUANT,
    build_notebook,
)

__all__ = [
    "__version__",
    "build_notebook",
    "DEFAULT_MODEL",
    "DEFAULT_HOSTNAME",
    "DEFAULT_PORT",
    "DEFAULT_CTX",
    "DEFAULT_QUANT",
]
