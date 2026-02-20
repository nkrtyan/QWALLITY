import json
import random
import string
import requests
import os
import sys
import argparse
import time
import math

# allow running directly: python -m src.chatbot_run
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    load_env,
    get_env,
    resolve_token_costs_usd_per_million,
)

# optional performance decorator
try:
    from src.utils import track_performance
except ImportError:
    def track_performance(func):
        return func


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_cfg():
    load_env()
    return {
        "api": {
            "url": get_env("chatbot_api_url") or get_env("api_url") or get_env("url"),
            "timeout": int(get_env("chatbot_api_timeout") or get_env("timeout") or 60),
        },
        "article_dir": get_env("article_chunks_path") or get_env("article_dir") or "Article_Chunk",
        "paths": {
            "synthetic_dataset_path": (
                get_env("synthetic_dataset_path")
                or get_env("synthetic_data_output")
                or "Data/synthetic_dataset.json"
            ),
            "inference_output_path": (
                get_env("inference_output_path")
                or get_env("chatbot_output")
                or get_env("chatbot_output_path")
                or "Data/inference_output.json"
            ),
        },
    }


def read_chunk_text(article_dir: str, slug: str) -> str:
    fp = os.path.join(article_dir, f"{slug}.txt")
    if not os.path.exists(fp):
        return ""
    with open(fp, "r", encoding="utf-8") as f:
        return f.read().strip()


def links_to_contexts(article_dir: str, retrieved_docs) -> list:
    contexts = []
    for slug in (retrieved_docs or []):
        txt = read_chunk_text(article_dir, slug)
        if txt:
            contexts.append(txt)
    return contexts


def compute_tokens_and_costs(
    *,
    question: str,
    chatbot_answer: str,
    retrived_docs,
    chatbot_retrieved_contexts,
    chat_in_cost_per_m,
    chat_out_cost_per_m,
    estimate_tokens_fn,
):
    """
    Token/cost accounting:
      - input: question
      - output: chatbot_answer + retrived_docs + chatbot_retrieved_contexts
    """
    in_tok = estimate_tokens_fn(question)
    out_tok_answer = estimate_tokens_fn(chatbot_answer)
    out_tok_docs = estimate_tokens_fn(json.dumps(retrived_docs or [], ensure_ascii=False))
    out_tok_ctx = estimate_tokens_fn(
        "\n".join(chatbot_retrieved_contexts or [])
    ) if chatbot_retrieved_contexts else 0

    out_tok = out_tok_answer + out_tok_docs + out_tok_ctx
    tot_tok = in_tok + out_tok

    if chat_in_cost_per_m is None or chat_out_cost_per_m is None:
        return {
            "chatbot_input_tokens": in_tok,
            "chatbot_output_tokens": out_tok,
            "chatbot_total_tokens": tot_tok,
            "chatbot_input_cost_usd": None,
            "chatbot_output_cost_usd": None,
            "chatbot_cost_usd": None,
        }

    input_cost = (in_tok / 1_000_000.0) * chat_in_cost_per_m
    output_cost = (out_tok / 1_000_000.0) * chat_out_cost_per_m

    return {
        "chatbot_input_tokens": in_tok,
        "chatbot_output_tokens": out_tok,
        "chatbot_total_tokens": tot_tok,
        "chatbot_input_cost_usd": f"{input_cost:.9f}",
        "chatbot_output_cost_usd": f"{output_cost:.9f}",
        "chatbot_cost_usd": f"{(input_cost + output_cost):.9f}",
    }


@track_performance
def get_bot_answers(dataset, cfg):
    print(f"[2/4] Querying API ({len(dataset)} items)...")

    url = cfg["api"]["url"]
    article_dir = cfg["article_dir"]

    model_name = get_env("chatbot_model") or get_env("gemini_model") or get_env("model")
    chat_in_cost_per_m, chat_out_cost_per_m, _ = resolve_token_costs_usd_per_million(
        model_name=model_name or "",
        stage_prefix="chatbot",
        provider_prefix="gemini",
    )

    def estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text or "") / 4))

    headers = {"Content-Type": "application/json"}

    for item in dataset:
        question = (item.get("question") or "").strip()

        if not question:
            item.update({
                "chatbot_answer": "EMPTY_QUESTION",
                "retrived_docs": [],
                "chatbot_retrieved_contexts": [],
                "chatbot_duration_ms": 0,
                "chatbot_input_tokens": None,
                "chatbot_output_tokens": None,
                "chatbot_total_tokens": None,
                "chatbot_input_cost_usd": None,
                "chatbot_output_cost_usd": None,
                "chatbot_cost_usd": None,
            })
            continue

        t0 = time.perf_counter()

        res = requests.post(
            url,
            headers=headers,
            json={"message": question, "show_docs": True},
            timeout=cfg["api"]["timeout"],
        )

        dur_ms = int((time.perf_counter() - t0) * 1000)

        if res.status_code == 200:
            body = res.json()

            answer = body.get("answer", "")
            retrived_docs = body.get("retrived_docs", [])

            item["chatbot_answer"] = answer
            item["retrived_docs"] = retrived_docs
            item["chatbot_retrieved_contexts"] = links_to_contexts(
                article_dir, retrived_docs
            )
            item["chatbot_duration_ms"] = dur_ms

            item.update(
                compute_tokens_and_costs(
                    question=question,
                    chatbot_answer=answer,
                    retrived_docs=retrived_docs,
                    chatbot_retrieved_contexts=item["chatbot_retrieved_contexts"],
                    chat_in_cost_per_m=chat_in_cost_per_m,
                    chat_out_cost_per_m=chat_out_cost_per_m,
                    estimate_tokens_fn=estimate_tokens,
                )
            )
        else:
            item.update({
                "chatbot_answer": "API_ERROR",
                "retrived_docs": [],
                "chatbot_retrieved_contexts": [],
                "chatbot_duration_ms": dur_ms,
                "chatbot_input_tokens": None,
                "chatbot_output_tokens": None,
                "chatbot_total_tokens": None,
                "chatbot_input_cost_usd": None,
                "chatbot_output_cost_usd": None,
                "chatbot_cost_usd": None,
            })

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recalc-only",
        action="store_true",
        help="Recalculate tokens/costs without calling the API",
    )
    args = parser.parse_args()

    cfg = build_cfg()

    if not cfg["api"]["url"] and not args.recalc_only:
        raise RuntimeError("Missing chatbot API URL. Check .env")

    if args.recalc_only:
        dataset = load_json(cfg["paths"]["inference_output_path"])

        model_name = get_env("chatbot_model") or get_env("gemini_model") or get_env("model")
        chat_in_cost_per_m, chat_out_cost_per_m, _ = resolve_token_costs_usd_per_million(
            model_name=model_name or "",
            stage_prefix="chatbot",
            provider_prefix="gemini",
        )

        def estimate_tokens(text: str) -> int:
            return max(1, math.ceil(len(text or "") / 4))

        for item in dataset:
            q = (item.get("question") or "").strip()
            ans = (item.get("chatbot_answer") or "").strip()
            retrived_docs = item.get("retrived_docs") or []

            item["chatbot_retrieved_contexts"] = links_to_contexts(
                cfg["article_dir"], retrived_docs
            )

            if not q or not ans:
                continue

            item.update(
                compute_tokens_and_costs(
                    question=q,
                    chatbot_answer=ans,
                    retrived_docs=retrived_docs,
                    chatbot_retrieved_contexts=item["chatbot_retrieved_contexts"],
                    chat_in_cost_per_m=chat_in_cost_per_m,
                    chat_out_cost_per_m=chat_out_cost_per_m,
                    estimate_tokens_fn=estimate_tokens,
                )
            )
    else:
        dataset = load_json(cfg["paths"]["synthetic_dataset_path"])
        dataset = get_bot_answers(dataset, cfg)

    save_json(cfg["paths"]["inference_output_path"], dataset)
    print(f"\nSaved → {cfg['paths']['inference_output_path']}")