import os
import sys
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
import time
import math

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from src.utils import load_env, get_env, parse_float_env, get_model_token_costs_usd_per_million, resolve_token_costs_usd_per_million  # must load .env.local


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_chunk_text(article_dir: str, slug: str) -> str:
    slug = slug.removesuffix(".md").removesuffix(".txt")
    fp = os.path.join(article_dir, f"{slug}.md")
    if not os.path.exists(fp):
        return ""
    with open(fp, "r", encoding="utf-8") as f:
        return f.read().strip()


def links_to_contexts(article_dir: str, retrieved_links: List[str]) -> List[str]:
    contexts: List[str] = []
    for s in (retrieved_links or []):
        txt = read_chunk_text(article_dir, s)
        if txt:
            contexts.append(txt)
    return contexts


def init_metric(m):
    return m() if callable(m) else m


def normalize_openai_model_name(name: str) -> str:
    """
    Some setups use shorthand model names like '4o-mini'. OpenAI expects 'gpt-4o-mini'.
    Normalize common shorthands; otherwise return unchanged.
    """
    n = (name or "").strip()
    mapping = {
        "4o-mini": "gpt-4o-mini",
        "4o": "gpt-4o",
        "4.1-mini": "gpt-4.1-mini",
        "4.1": "gpt-4.1",
    }
    return mapping.get(n, n)


def build_judge_llm() -> LangchainLLMWrapper:
    """
    Builds the judge LLM used by RAGAS metrics.

    Your setup:
      - synthetic generation + chatbot_run: Gemini (separate scripts)
      - evaluation (this file): OpenAI (default)

    Configure via env:
      - evaluation_provider=openai|gemini  (default: openai)
      - evaluation_model=... (OpenAI model name or Gemini model name depending on provider)
      - OPENAI_API_KEY (for openai)
      - GEMINI_API_KEY/GOOGLE_API_KEY (for gemini)
    """
    provider = (get_env("evaluation_provider") or "openai").strip().lower()
    temperature = float(os.getenv("RAGAS_TEMPERATURE", "0") or 0)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY. Ensure .env.local is loaded.")

        model_name = (
            get_env("evaluation_model")
            or get_env("OPENAI_MODEL")
            or get_env("openai_model")
        )
        if not model_name:
            raise ValueError("Missing OpenAI evaluation model. Set evaluation_model (or OPENAI_MODEL/openai_model) in .env/.env.local.")
        model_name = normalize_openai_model_name(model_name)
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise ImportError(
                "Missing dependency for OpenAI evaluation. Install `langchain-openai`."
            ) from e

        print(f"Using OpenAI judge model: {model_name}")
        judge = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        return LangchainLLMWrapper(judge)

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Ensure .env.local is loaded.")

        model_name = get_env("evaluation_model") or get_env("gemini_model") or get_env("model")
        if not model_name:
            raise ValueError("Missing Gemini evaluation model. Set evaluation_model or model in .env/.env.local.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            raise ImportError(
                "Missing dependency for Gemini evaluation. Install `langchain-google-genai`."
            ) from e

        judge = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
        return LangchainLLMWrapper(judge)

    raise ValueError(f"Unsupported evaluation_provider={provider!r}. Use 'openai' or 'gemini'.")


def build_rows(items: List[Dict], article_dir: str) -> List[Dict]:
    """
    Output row order you requested (base fields for ragas_input.json / ragas_results.csv):
    question
    expected_context
    chatbot_answer
    chatbot_retrieved_contexts
    slug
    retrieved_links
    synthetic_input_tokens
    synthetic_output_tokens
    synthetic_total_tokens
    synthetic_input_cost_usd
    synthetic_output_cost_usd
    synthetic_total_cost_usd
    synthetic_duration_ms
    chatbot_input_tokens
    chatbot_output_tokens
    chatbot_total_tokens
    chatbot_input_cost_usd
    chatbot_output_cost_usd
    chatbot_cost_usd
    chatbot_duration_ms
    """
    rows: List[Dict] = []

    for it in items:
        question = (it.get("question") or "").strip()
        expected_context = (it.get("expected_context") or "").strip()
        slug = (it.get("slug") or "").strip()

        chatbot_answer = (it.get("chatbot_answer") or "").strip()
        retrieved_links = it.get("retrived_docs") or []

        if not question or not chatbot_answer:
            continue

        # Prefer contexts already persisted by chatbot_run (if present), otherwise build from slugs
        persisted_ctx = it.get("chatbot_retrieved_contexts")
        if isinstance(persisted_ctx, list) and all(isinstance(x, str) for x in persisted_ctx):
            chatbot_retrieved_contexts = persisted_ctx
        else:
            chatbot_retrieved_contexts = links_to_contexts(article_dir, retrieved_links)

        # Base fields (exact keys you requested)
        row = {
            "question": question,
            "expected_context": expected_context,
            "chatbot_answer": chatbot_answer,
            "chatbot_retrieved_contexts": chatbot_retrieved_contexts,
            "slug": slug,
            "retrieved_links": retrieved_links,
            "synthetic_input_tokens": it.get("synthetic_input_tokens"),
            "synthetic_output_tokens": it.get("synthetic_output_tokens"),
            "synthetic_total_tokens": it.get("synthetic_total_tokens"),
            "synthetic_input_cost_usd": it.get("synthetic_input_cost_usd"),
            "synthetic_output_cost_usd": it.get("synthetic_output_cost_usd"),
            "synthetic_total_cost_usd": it.get("synthetic_total_cost_usd"),
            "synthetic_duration_ms": it.get("synthetic_duration_ms"),
            "chatbot_input_tokens": it.get("chatbot_input_tokens"),
            "chatbot_output_tokens": it.get("chatbot_output_tokens"),
            "chatbot_total_tokens": it.get("chatbot_total_tokens"),
            "chatbot_input_cost_usd": it.get("chatbot_input_cost_usd"),
            "chatbot_output_cost_usd": it.get("chatbot_output_cost_usd"),
            "chatbot_cost_usd": it.get("chatbot_cost_usd"),
            "chatbot_duration_ms": it.get("chatbot_duration_ms"),
        }

        rows.append(row)

    return rows


def estimate_tokens(text: str) -> int:
    t = (text or "")
    return max(1, math.ceil(len(t) / 4))


def _format_costs_for_debug(row: Dict[str, Any], decimals: int = 9) -> Dict[str, Any]:
    """
    Create a shallow copy of row and format monetary fields as fixed-decimal strings
    so they are clearly visible in ragas_input.json instead of 0.
    """
    out = dict(row)
    for key in [
        "synthetic_input_cost_usd",
        "synthetic_output_cost_usd",
        "synthetic_total_cost_usd",
        "chatbot_input_cost_usd",
        "chatbot_output_cost_usd",
        "chatbot_cost_usd",
        "evaluation_input_cost_usd",
        "evaluation_output_cost_usd",
        "evaluation_cost_usd",
    ]:
        val = out.get(key, None)
        try:
            if val is not None:
                out[key] = f"{float(val):.{decimals}f}"
        except Exception:
            # leave as-is if not convertible
            pass
    return out


def run_ragas(
    inference_path: Optional[str] = None,
    article_dir: str = "Article_Chunk",
    ragas_input_debug_path: str = "Data/ragas_input.json",
    output_csv_path: str = "Data/ragas_results.csv",
) -> pd.DataFrame:

    # Default to the chatbot output file if present in env
    if not inference_path:
        inference_path = (
            get_env("inference_output_path")
            or get_env("chatbot_output")
            or get_env("chatbot_output_path")
            or "Data/chatbot_answers.json"
        )

    if not os.path.exists(inference_path):
        raise FileNotFoundError(
            f"Missing file: {inference_path}. "
            "Set inference_output_path or chatbot_output in .env/.env.local, or pass inference_path explicitly."
        )
    if not os.path.isdir(article_dir):
        raise FileNotFoundError(f"Missing folder: {article_dir}")

    items = load_json(inference_path)
    if not isinstance(items, list):
        raise ValueError("inference_output.json must be a list of objects")

    rows = build_rows(items, article_dir)
    if not rows:
        print("No valid rows to evaluate.")
        return pd.DataFrame()

    llm = build_judge_llm()
    metrics = [init_metric(m) for m in [context_precision, context_recall, faithfulness, answer_relevancy]]

    # Evaluate per row to capture per-question timing
    eval_in_cost_per_m = parse_float_env("evaluation_input_cost_per_million")
    eval_out_cost_per_m = parse_float_env("evaluation_output_cost_per_million")
    eval_out_tokens_per_metric = int(get_env("evaluation_output_tokens_per_metric") or 64)

    # Optional infer from model pricing table
    provider = (get_env("evaluation_provider") or "openai").strip().lower()
    model_name = (
        get_env("evaluation_model")
        or get_env("OPENAI_MODEL")
        or get_env("openai_model")
        or get_env("openai_model")  # alias
        or get_env("gemini_model")
        or get_env("model")
    )
    if provider == "openai":
        model_name = normalize_openai_model_name(model_name or "")

    # Resolve eval cost rates primarily by model name (OpenAI in your setup)
    resolved_in, resolved_out, eval_cost_source = resolve_token_costs_usd_per_million(
        model_name=model_name or "",
        stage_prefix="evaluation",
        provider_prefix=("openai" if provider == "openai" else "gemini"),
    )
    if eval_in_cost_per_m is None:
        eval_in_cost_per_m = resolved_in
    if eval_out_cost_per_m is None:
        eval_out_cost_per_m = resolved_out

    metric_rows: List[Dict[str, Any]] = []
    for r in rows:
        single_ds = Dataset.from_dict(
            {
                "question": [r["question"]],
                "answer": [r["chatbot_answer"]],
                "contexts": [r["chatbot_retrieved_contexts"]],
                "reference": [r["expected_context"]],
            }
        )

        # Evaluation input tokens: treat the whole ragas_input row as input
        eval_input_payload = dict(r)
        eval_in_tokens = estimate_tokens(json.dumps(eval_input_payload, ensure_ascii=False))
        # We'll compute output tokens after we get the metric outputs, then compute costs.

        t0 = time.perf_counter()
        try:
            # Fail fast if the judge model is invalid; otherwise metrics will be empty/NaN.
            res = evaluate(
                single_ds,
                metrics=metrics,
                llm=llm,
                raise_exceptions=True,
            )
        except Exception as e:
            raise RuntimeError(
                "RAGAS evaluation failed while calling the judge LLM. "
                "Most common cause: invalid OpenAI model name or missing access. "
                "Fix evaluation_model/openai_model in .env/.env.local and try again."
            ) from e
        eval_ms = int((time.perf_counter() - t0) * 1000)
        df_one = res.to_pandas()
        # Keep only metric numbers (no text)
        row_metrics = df_one.iloc[0].to_dict() if len(df_one) > 0 else {}
        metric_rows.append(row_metrics)

        # Evaluation output tokens: treat ragas_result (metrics) as output
        out_json = json.dumps(row_metrics, ensure_ascii=False)
        eval_out_tokens_est = estimate_tokens(out_json)
        eval_out_tokens = max(eval_out_tokens_est, eval_out_tokens_per_metric * len(metrics))
        eval_tot_tokens = eval_in_tokens + eval_out_tokens

        if eval_in_cost_per_m is None or eval_out_cost_per_m is None:
            eval_input_cost = None
            eval_output_cost = None
            eval_cost_total = None
        else:
            eval_input_cost = (eval_in_tokens / 1_000_000.0) * eval_in_cost_per_m
            eval_output_cost = (eval_out_tokens / 1_000_000.0) * eval_out_cost_per_m
            eval_cost_total = eval_input_cost + eval_output_cost

        # Persist evaluation tracing on the row (for ragas_input.json and ragas_results.csv)
        r["evaluation_input_tokens"] = eval_in_tokens
        r["evaluation_output_tokens"] = eval_out_tokens
        r["evaluation_total_tokens"] = eval_tot_tokens
        r["evaluation_input_cost_usd"] = (f"{eval_input_cost:.9f}" if eval_input_cost is not None else None)
        r["evaluation_output_cost_usd"] = (f"{eval_output_cost:.9f}" if eval_output_cost is not None else None)
        r["evaluation_cost_usd"] = (f"{eval_cost_total:.9f}" if eval_cost_total is not None else None)
        r["evaluation_duration_ms"] = eval_ms

    df_metrics = pd.DataFrame(metric_rows)

    # Save ragas_input.json with requested fields (including evaluation tokens/cost/time)
    ragas_input_keys = [
        "question",
        "expected_context",
        "chatbot_answer",
        "chatbot_retrieved_contexts",
        "slug",
        "retrieved_links",
        "synthetic_input_tokens",
        "synthetic_output_tokens",
        "synthetic_total_tokens",
        "synthetic_input_cost_usd",
        "synthetic_output_cost_usd",
        "synthetic_total_cost_usd",
        "synthetic_duration_ms",
        "chatbot_input_tokens",
        "chatbot_output_tokens",
        "chatbot_total_tokens",
        "chatbot_input_cost_usd",
        "chatbot_output_cost_usd",
        "chatbot_cost_usd",
        "chatbot_duration_ms",
        "evaluation_input_tokens",
        "evaluation_output_tokens",
        "evaluation_total_tokens",
        "evaluation_input_cost_usd",
        "evaluation_output_cost_usd",
        "evaluation_cost_usd",
        "evaluation_duration_ms",
    ]
    rows_out = [{k: r.get(k) for k in ragas_input_keys} for r in rows]
    rows_debug = [_format_costs_for_debug(r, decimals=9) for r in rows_out]
    save_json(ragas_input_debug_path, rows_debug)
    print(f"Saved RAGAS input -> {ragas_input_debug_path}")

    # Build output table with EXACT columns requested, then append metric columns as numbers
    def _json_list(x):
        try:
            return json.dumps(x or [], ensure_ascii=False)
        except Exception:
            return "[]"

    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    metric_data: Dict[str, Any] = {}
    for col in metric_cols:
        if col in df_metrics.columns:
            metric_data[col] = pd.to_numeric(df_metrics[col], errors="coerce")
        else:
            metric_data[col] = [None for _ in rows]

    # Build output table with EXACT column order requested:
    # base fields -> metrics -> tokens/costs/durations
    df = pd.DataFrame(
        {
            "question": [r["question"] for r in rows],
            "expected_context": [r["expected_context"] for r in rows],
            "chatbot_answer": [r["chatbot_answer"] for r in rows],
            "chatbot_retrieved_contexts": [_json_list(r["chatbot_retrieved_contexts"]) for r in rows],
            "slug": [r["slug"] for r in rows],
            "retrieved_links": [_json_list(r["retrieved_links"]) for r in rows],
            **metric_data,
            "synthetic_input_tokens": [r["synthetic_input_tokens"] for r in rows],
            "synthetic_output_tokens": [r["synthetic_output_tokens"] for r in rows],
            "synthetic_total_tokens": [r["synthetic_total_tokens"] for r in rows],
            "synthetic_input_cost_usd": [r["synthetic_input_cost_usd"] for r in rows],
            "synthetic_output_cost_usd": [r["synthetic_output_cost_usd"] for r in rows],
            "synthetic_total_cost_usd": [r["synthetic_total_cost_usd"] for r in rows],
            "synthetic_duration_ms": [r["synthetic_duration_ms"] for r in rows],
            "chatbot_input_tokens": [r["chatbot_input_tokens"] for r in rows],
            "chatbot_output_tokens": [r["chatbot_output_tokens"] for r in rows],
            "chatbot_total_tokens": [r["chatbot_total_tokens"] for r in rows],
            "chatbot_input_cost_usd": [r["chatbot_input_cost_usd"] for r in rows],
            "chatbot_output_cost_usd": [r["chatbot_output_cost_usd"] for r in rows],
            "chatbot_cost_usd": [r["chatbot_cost_usd"] for r in rows],
            "chatbot_duration_ms": [r["chatbot_duration_ms"] for r in rows],
            "evaluation_input_tokens": [r.get("evaluation_input_tokens") for r in rows],
            "evaluation_output_tokens": [r.get("evaluation_output_tokens") for r in rows],
            "evaluation_total_tokens": [r.get("evaluation_total_tokens") for r in rows],
            "evaluation_input_cost_usd": [r.get("evaluation_input_cost_usd") for r in rows],
            "evaluation_output_cost_usd": [r.get("evaluation_output_cost_usd") for r in rows],
            "evaluation_cost_usd": [r.get("evaluation_cost_usd") for r in rows],
            "evaluation_duration_ms": [r.get("evaluation_duration_ms") for r in rows],
        }
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Saved RAGAS results -> {output_csv_path}")

    return df


if __name__ == "__main__":
    load_env()
    run_ragas()
