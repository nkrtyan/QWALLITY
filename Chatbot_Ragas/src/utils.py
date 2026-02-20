# src/utils.py

import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv


def load_env() -> None:
    """
    Loads .env first, then .env.local (secrets override).
    Call this once at the beginning of your scripts.
    """
    load_dotenv(".env")
    load_dotenv(".env.local", override=True)


def get_env(name: str, default: str = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        return default
    return value


def parse_float_env(name: str) -> Optional[float]:
    """
    Reads an env var and parses it as float.
    Returns None if missing/empty.
    """
    raw = get_env(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except Exception:
        raise ValueError(f"Env var '{name}' must be a float, got: {raw!r}")


def load_model_pricing_table(path: str = "Data/model_pricing.json") -> Dict[str, Any]:
    """
    Loads a JSON pricing table keyed by model name.

    Expected structure:
      {
        "gemini-2.0-flash": {"input_usd_per_million_tokens": 0.0, "output_usd_per_million_tokens": 0.0},
        ...
      }

    Missing file -> returns {} (so code can fall back to defaults/env).
    """
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_model_token_costs_usd_per_million(model_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Pricing lookup keyed by the model name string you use in env (e.g. `.env` `model=`,
    `.env` `evaluation_model=`, etc.).

    Source of truth: Data/model_pricing.json

    If the model is missing (or values are null), returns (None, None).
    """
    key = (model_name or "").strip()
    if not key:
        return None, None

    pricing = load_model_pricing_table()
    if key not in pricing or not isinstance(pricing.get(key), dict):
        return None, None

    row = pricing[key]
    in_cost = row.get("input_usd_per_million_tokens", None)
    out_cost = row.get("output_usd_per_million_tokens", None)
    try:
        in_val = None if in_cost is None else float(in_cost)
        out_val = None if out_cost is None else float(out_cost)
        return in_val, out_val
    except Exception:
        return None, None


def resolve_token_costs_usd_per_million(
    *,
    model_name: str,
    stage_prefix: Optional[str] = None,
    provider_prefix: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Resolve (input_cost_per_million, output_cost_per_million) in USD.

    Priority:
      1) stage-specific env vars, e.g. "{stage_prefix}_input_cost_per_million"
      2) exact model lookup in Data/model_pricing.json
      3) provider-specific env vars, e.g. "{provider_prefix}_input_cost_per_million"

    Returns: (in_cost, out_cost, source_string)
    """
    # 1) stage-specific env vars
    if stage_prefix:
        in_stage = parse_float_env(f"{stage_prefix}_input_cost_per_million")
        out_stage = parse_float_env(f"{stage_prefix}_output_cost_per_million")
        if in_stage is not None and out_stage is not None:
            return in_stage, out_stage, f"env:{stage_prefix}_*_cost_per_million"

    # 2) model pricing table
    in_model, out_model = get_model_token_costs_usd_per_million(model_name)
    if in_model is not None and out_model is not None:
        return in_model, out_model, "Data/model_pricing.json"

    # 3) provider-specific env vars
    if provider_prefix:
        in_p = parse_float_env(f"{provider_prefix}_input_cost_per_million")
        out_p = parse_float_env(f"{provider_prefix}_output_cost_per_million")
        if in_p is not None and out_p is not None:
            return in_p, out_p, f"env:{provider_prefix}_*_cost_per_million"

    return None, None, "missing"


def load_article_chunks(folder_path: str = "Article_Chunk") -> List[Dict[str, str]]:
    """
    Reads all .md files from Article_Chunk folder.
    Returns list of { "slug": "...", "text": "..." }.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    items = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".md"):
            continue

        full_path = os.path.join(folder_path, filename)
        slug = os.path.splitext(filename)[0]

        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # light cleanup (optional)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # remove markdown images
        text = re.sub(r"\[(?:link|հղում).*?\]\(.*?\)", "", text, flags=re.IGNORECASE)  # remove markdown links
        text = text.strip()

        if len(text) < 50:
            continue

        items.append({"slug": slug, "text": text})

    if not items:
        raise ValueError("No valid .md files found in Article_Chunk.")

    return items
