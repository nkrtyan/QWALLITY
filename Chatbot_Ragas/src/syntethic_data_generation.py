import os
import sys

# ✅ allow running this file directly: python src/synthetic_data_generation.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import Dict, List, Optional
import time

import google.generativeai as genai

from src.utils import load_env, get_env, load_article_chunks, resolve_token_costs_usd_per_million
from prompts.synthetic_data_generation_prompt import synthetic_data_prompt


FORBIDDEN_UI_WORDS = {
    "click", "select", "choose", "enter", "then", "after", "re-enter", "save", "button"
}


def require_env(name: str, where: str) -> str:
    value = get_env(name)
    if value is None or value == "":
        raise ValueError(f"Missing '{name}' in {where}")
    return value


def extract_json_first_object(raw: str) -> Dict:
    """
    Robust JSON extraction:
    - strips markdown fences
    - finds first '{'
    - parses ONLY the first JSON object (ignores extra text after it)
    """
    raw = (raw or "").replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("Model response does not contain a valid JSON object.")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])  # parse first JSON only
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def build_gemini_model() -> genai.GenerativeModel:
    api_key = require_env("gemini_api_key", ".env.local")
    model_name = get_env("gemini_model") or get_env("model")
    if not model_name:
        raise ValueError("Missing Gemini model name. Set gemini_model (preferred) or model in .env/.env.local.")
    temperature = float(require_env("temperature", ".env"))

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 1024,
        },
    )


def normalize_question(q: str) -> str:
    return " ".join((q or "").lower().split())


def looks_like_ui_steps(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in FORBIDDEN_UI_WORDS)


def generate_one_item_json(model: genai.GenerativeModel, prompt: str, repair_tries: int = 2) -> Dict:
    """
    Calls the model and guarantees returning a parsed JSON object.
    If model returns non-JSON, silently retries and uses a "repair" prompt.
    """
    start_time = time.perf_counter()
    resp = model.generate_content(prompt)
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    raw = (resp.text or "").strip()
    usage = getattr(resp, "usage_metadata", None) or {}

    def _usage_get(u, *names):
        if isinstance(u, dict):
            for n in names:
                v = u.get(n)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        pass
        else:
            for n in names:
                v = getattr(u, n, None)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        pass
        return 0

    # Gemini usage fields may appear as prompt_token_count/candidates_token_count/total_token_count
    input_tokens = _usage_get(usage, "input_tokens", "prompt_token_count")
    output_tokens = _usage_get(usage, "output_tokens", "candidates_token_count")
    total_tokens = _usage_get(usage, "total_tokens", "total_token_count")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    try:
        obj = extract_json_first_object(raw)
        return {
            "obj": obj,
            "synthetic_input_tokens": input_tokens,
            "synthetic_output_tokens": output_tokens,
            "synthetic_total_tokens": total_tokens,
            "synthetic_duration_ms": duration_ms,
        }
    except Exception:
        # Repair attempts: ask model to convert its previous output to valid JSON ONLY.
        last = raw[:2000]  # keep it short
        repair_prompt = (
            "Convert the following content into EXACTLY ONE valid JSON object with keys "
            "\"question\" and \"expected_context\". Return ONLY JSON. No markdown. No extra text.\n\n"
            "CONTENT:\n"
            f"{last}"
        )
        for _ in range(repair_tries):
            start2 = time.perf_counter()
            resp2 = model.generate_content(repair_prompt)
            dur2_ms = int((time.perf_counter() - start2) * 1000)
            raw2 = (resp2.text or "").strip()
            usage2 = getattr(resp2, "usage_metadata", None) or {}
            in2 = _usage_get(usage2, "input_tokens", "prompt_token_count")
            out2 = _usage_get(usage2, "output_tokens", "candidates_token_count")
            tot2 = _usage_get(usage2, "total_tokens", "total_token_count") or (in2 + out2)
            try:
                obj2 = extract_json_first_object(raw2)
                return {
                    "obj": obj2,
                    "synthetic_input_tokens": in2,
                    "synthetic_output_tokens": out2,
                    "synthetic_total_tokens": tot2,
                    "synthetic_duration_ms": dur2_ms,
                }
            except Exception:
                continue

    # If still not possible, raise a clean error (but we won't print spam in the loop).
    raise ValueError("Could not obtain valid JSON from model after repair attempts.")


def generate_synthetic_dataset(questions_per_article: int = 3) -> List[Dict]:
    """
    Sequential generation:
    - article → generate N questions (unique per article + globally)
    - next article → generate N questions
    """

    chunks_path = require_env("article_chunks_path", ".env")
    chunks = load_article_chunks(chunks_path)

    model = build_gemini_model()

    # Cost config (USD per 1M tokens), resolved by model name from env
    model_name = get_env("gemini_model") or get_env("model")
    if not model_name:
        raise ValueError("Missing gemini_model (or legacy model) in env.")
    gen_in_cost_per_m, gen_out_cost_per_m, cost_source = resolve_token_costs_usd_per_million(
        model_name=model_name,
        stage_prefix="generation",
        provider_prefix="gemini",
    )
    if gen_in_cost_per_m is None or gen_out_cost_per_m is None:
        print(
            f"ℹ️ Synthetic cost is unknown for model={model_name!r} (no pricing found). "
            "Fill Data/model_pricing.json or set generation_*_cost_per_million (or gemini_*_cost_per_million)."
        )
    results: List[Dict] = []
    seen_questions_global = set()

    total_articles = len(chunks)
    print(f"Articles found: {total_articles}")
    print(f"Questions per article: {questions_per_article}\n")

    for article_index, item in enumerate(chunks, start=1):
        slug = item["slug"]
        context = item["text"]

        print(f"--- Article {article_index}/{total_articles}: {slug} ---")

        created = 0
        attempts_guard = 0
        max_attempts = questions_per_article * 20  # more tolerant, still bounded

        seen_questions_article = set()
        # Charge input tokens only once per article (do not duplicate per question)
        article_input_tokens_once: Optional[int] = None
        article_input_charged: bool = False

        while created < questions_per_article and attempts_guard < max_attempts:
            attempts_guard += 1

            prompt = synthetic_data_prompt.format(slug=slug, context=context)

            try:
                result = generate_one_item_json(model, prompt)
                obj = result["obj"]

                question = (obj.get("question") or "").strip()
                ref_ctx = (obj.get("expected_context") or "").strip()

                if not question or not ref_ctx:
                    continue

                # Reject UI-step reference contexts
                if looks_like_ui_steps(ref_ctx):
                    continue

                # Enforce uniqueness per article + global
                q_norm = normalize_question(question)
                if q_norm in seen_questions_article or q_norm in seen_questions_global:
                    continue

                seen_questions_article.add(q_norm)
                seen_questions_global.add(q_norm)

                # Raw tokens from the generation call
                in_tok_raw = int(result.get("synthetic_input_tokens", 0) or 0)
                out_tok = int(result.get("synthetic_output_tokens", 0) or 0)
                # Allocate input tokens only once per article
                if article_input_tokens_once is None:
                    article_input_tokens_once = in_tok_raw
                if not article_input_charged:
                    in_tok = int(article_input_tokens_once or in_tok_raw or 0)
                    article_input_charged = True
                else:
                    in_tok = 0
                tot_tok = in_tok + out_tok
                dur_ms = int(result.get("synthetic_duration_ms", 0) or 0)
                if gen_in_cost_per_m is None or gen_out_cost_per_m is None:
                    input_cost_usd = None
                    output_cost_usd = None
                    total_cost_usd = None
                else:
                    input_cost_usd = (in_tok / 1_000_000.0) * gen_in_cost_per_m
                    output_cost_usd = (out_tok / 1_000_000.0) * gen_out_cost_per_m
                    total_cost_usd = input_cost_usd + output_cost_usd

                results.append({
                    "question": question,
                    "expected_context": ref_ctx,
                    "slug": slug,
                    "synthetic_input_tokens": in_tok,
                    "synthetic_output_tokens": out_tok,
                    "synthetic_total_tokens": tot_tok,
                    # Costs (string with higher precision so small costs don't round to 0.00)
                    "synthetic_input_cost_usd": (f"{input_cost_usd:.9f}" if input_cost_usd is not None else None),
                    "synthetic_output_cost_usd": (f"{output_cost_usd:.9f}" if output_cost_usd is not None else None),
                    "synthetic_total_cost_usd": (f"{total_cost_usd:.9f}" if total_cost_usd is not None else None),
                    "synthetic_duration_ms": dur_ms,
                })
                created += 1
                print(f"  Q{created}: {question}")

            except Exception:
                # Silent retry (no "failed" spam)
                continue

        if created < questions_per_article:
            # One clean message only if model truly couldn't produce enough after many tries
            print(f"  ⚠️ Only generated {created}/{questions_per_article} for this article (after {attempts_guard} attempts)\n")
        else:
            print()

    return results


def save_json(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    load_env()

    output_path = get_env("synthetic_data_output") or get_env("synthetic_dataset_path")
    if not output_path:
        raise ValueError("Missing synthetic dataset output path. Set synthetic_data_output (preferred) or synthetic_dataset_path in .env/.env.local.")

    dataset = generate_synthetic_dataset(questions_per_article=1)

    save_json(output_path, dataset)
    print(f"\nSaved synthetic dataset → {output_path}")
