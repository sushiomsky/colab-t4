"""Stable, sanitized provider metadata for integrations such as AblitBot."""
from __future__ import annotations

from dataclasses import dataclass

from .accounts import load_accounts


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repository: str
    quant: str
    description: str


_MODELS = (
    ModelSpec(
        alias="qwen3.5-9b-uncensored",
        repository="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
        quant="Q6_K",
        description="Qwen3.5 9B uncensored aggressive instruct model",
    ),
    ModelSpec(
        alias="llama3.1-8b-ablit",
        repository="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF",
        quant="Q4_K_M",
        description="Llama 3.1 8B abliterated instruct model",
    ),
    ModelSpec(
        alias="mistral-nemo-12b-ablit",
        repository="QuantFactory/Mistral-Nemo-Instruct-2407-abliterated-GGUF",
        quant="Q4_K_M",
        description="Mistral Nemo 12B abliterated instruct model",
    ),
)


def available_models() -> tuple[ModelSpec, ...]:
    return _MODELS


def resolve_model(alias: str) -> ModelSpec:
    for model in _MODELS:
        if model.alias == alias:
            return model
    raise ValueError(f"unknown model alias: {alias}")


def account_rows() -> list[dict[str, object]]:
    """Return account metadata safe for a local integration to consume."""
    rows: list[dict[str, object]] = []
    for account in load_accounts():
        if account.last_error:
            status = "error"
        elif account.last_ok:
            status = "ok"
        else:
            status = "unused"
        rows.append({
            "id": account.id,
            "email": account.email or "unknown",
            "status": status,
            "last_used": account.last_used_at or "-",
        })
    return rows
