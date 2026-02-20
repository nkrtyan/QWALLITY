synthetic_data_prompt = """
You are an expert at creating high-quality synthetic evaluation data for a RAG chatbot.

TASK:
Generate exactly ONE user question that can be answered using the SOURCE TEXT.
Also generate a clear and self-contained reference context (ground truth) that supports the answer.


QUESTION RULES:
- Write like a real user (short, simple, everyday wording).
- The question MUST be fully answerable using the source text.
- The question MUST be in the SAME language as the source text.
- Do NOT generate duplicate or very similar questions.
- The expected_context MUST contain ONLY information found in the source text.
- Avoid repeating the same question pattern across attempts (rephrase).

REFERENCE CONTEXT RULES:
- The expected_context MUST be a grammatically complete sentence and MUST end with a full stop (.). Do NOT stop early.
- It MUST contain ONLY information found in the source text (no new facts).
- Do NOT copy partial sentences.
- Do NOT add new facts or assumptions.
- You MAY mention UI labels (e.g., "My Settings", "Change Password") if they appear in the text.
- It MUST directly support the answer and be understandable on its own.
- Avoid step-by-step phrasing and filler words like "then/after that", but do NOT truncate meaning.
- The expected_context must NOT be partial, cut off, or unfinished and must include the main information from the article.


OUTPUT RULES:
- Return ONLY valid JSON.
- No markdown.
- No explanations.
- Output MUST start with {{ and end with }}.

OUTPUT JSON FORMAT:
{{
  "question": "...",
  "expected_context": "..."
}}

SOURCE FILE (slug):
{slug}

SOURCE TEXT:
{context}
"""
