# engine/questions.py
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from config import CCNA_BLUEPRINT, GEMINI_API_KEY, GEMINI_MODEL
from db.database import bulk_upsert_questions, get_questions, adjust_question_difficulty


# ---------------------------
# Parsing helpers
# ---------------------------
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Remove common Markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_list(raw: str) -> List[Dict[str, Any]]:
    """
    Parses Gemini output expecting a JSON array of objects.
    Tries:
      1) direct JSON
      2) extracting the first [...] array
    """
    cleaned = _strip_code_fences(raw)

    # 1) direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass

    # 2) find the first JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        data = json.loads(snippet)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

    raise ValueError("Model did not return a valid JSON array (list) of questions.")


def _ensure_four_options(opts: Any) -> List[str]:
    """
    Normalize options into a list of exactly 4 strings if possible.
    If the model returns fewer/more, we trim/pad safely.
    """
    if isinstance(opts, str):
        # Sometimes a JSON string or one long string
        try:
            parsed = json.loads(opts)
            opts = parsed
        except Exception:
            opts = [opts]

    if not isinstance(opts, list):
        return []

    # Keep only strings
    out = [str(x).strip() for x in opts if str(x).strip()]
    if len(out) >= 4:
        return out[:4]

    # Pad with placeholders if too short (rare)
    while len(out) < 4:
        out.append(f"(Option {len(out)+1})")
    return out


def _normalize_question(q: Dict[str, Any], domain: str, chapter_id: Optional[str], chapter_title: Optional[str], source_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate + normalize a single question dict so Practice Mode won't crash.
    Returns normalized dict or None if unusable.
    """
    qid = (q.get("id") or "").strip()
    question_text = (q.get("question") or "").strip()
    blueprint = (q.get("blueprint") or "").strip()

    # Must-have fields
    if not qid or not question_text or not blueprint:
        return None

    # Normalize difficulty
    diff = (q.get("difficulty") or "Medium").strip().title()
    if diff not in ("Easy", "Medium", "Hard"):
        diff = "Medium"

    # Normalize options
    options = _ensure_four_options(q.get("options"))
    if not options:
        return None

    # Normalize answer: must exactly match an option
    answer = (q.get("answer") or "").strip()
    if answer not in options:
        # Try to fix by index-like answers ("A", "B", "C", "D")
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        up = answer.upper()
        if up in letter_map and letter_map[up] < len(options):
            answer = options[letter_map[up]]
        else:
            # fallback: pick first option to avoid crashing UI
            answer = options[0]

    explanation = (q.get("explanation") or "").strip()

    out = {
        "id": qid,
        "domain": domain,
        "blueprint": blueprint,
        "difficulty": diff,
        "question": question_text,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }

    # Backfill optional metadata
    if chapter_id and not q.get("chapter_id"):
        out["chapter_id"] = chapter_id
    else:
        out["chapter_id"] = q.get("chapter_id")

    if chapter_title and not q.get("chapter_title"):
        out["chapter_title"] = chapter_title
    else:
        out["chapter_title"] = q.get("chapter_title")

    if source_name and not q.get("source_name"):
        out["source_name"] = source_name
    else:
        out["source_name"] = q.get("source_name")

    return out


# ---------------------------
# Main API
# ---------------------------
def gen_questions(
    text: str,
    n: int,
    domain: str,
    *,
    chapter_id: Optional[str] = None,
    chapter_title: Optional[str] = None,
    source_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in your environment variables or config.py.")

    if domain not in CCNA_BLUEPRINT:
        raise ValueError(f"Unknown domain '{domain}'. Check config.CCNA_BLUEPRINT keys.")

    blueprint_targets = CCNA_BLUEPRINT[domain]

    # Keep prompt size reasonable
    MAX_CHARS = 12000
    context = (text or "").strip()
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS]

    meta_parts = []
    if source_name:
        meta_parts.append(f"source_name={source_name}")
    if chapter_id:
        meta_parts.append(f"chapter_id={chapter_id}")
    if chapter_title:
        meta_parts.append(f"chapter_title={chapter_title}")
    meta_str = ", ".join(meta_parts) if meta_parts else "none"

    prompt = f"""
Act as a Cisco CCNA 200-301 v1.1 exam proctor.

You MUST generate questions ONLY from the provided study material context below.

Metadata: {meta_str}

STUDY MATERIAL (use ONLY this):
\"\"\"{context}\"\"\"

For each question:
- Map it to EXACTLY ONE blueprint objective from this list:
{blueprint_targets}

Return JSON ONLY.
Return a JSON array (list) of exactly {n} questions.

Each question object MUST include:
- id (unique string)
- domain (must be "{domain}")
- blueprint (must exactly match one objective above)
- difficulty (Easy/Medium/Hard)
- question
- options (array of 4 strings)
- answer (must exactly match one of the options)
- explanation

Also include these OPTIONAL fields if possible:
- chapter_id
- chapter_title
- source_name
""".strip()

    client = genai.Client(api_key=GEMINI_API_KEY)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    raw = resp.text or ""
    parsed = _parse_json_list(raw)

    normalized: List[Dict[str, Any]] = []
    for q in parsed:
        nq = _normalize_question(q, domain, chapter_id, chapter_title, source_name)
        if nq:
            normalized.append(nq)

    # If Gemini returns fewer usable questions than requested, still store what we got
    if not normalized:
        raise ValueError("Gemini returned no usable questions (missing fields/options). Try again or reduce chapter text.")

    # ✅ Fix A: ONE fast transaction instead of n commits
    bulk_upsert_questions(normalized)

    return normalized


def select_adaptive_questions(
    domain: str,
    n: int,
    *,
    chapter_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    all_qs = [q for q in (get_questions() or []) if q.get("domain") == domain]

    if chapter_ids:
        ch_set = set(chapter_ids)
        all_qs = [q for q in all_qs if q.get("chapter_id") in ch_set]

    new_qs = [q for q in all_qs if int(q.get("seen", 0) or 0) == 0]
    seen_qs = [q for q in all_qs if int(q.get("seen", 0) or 0) > 0]

    for q in seen_qs:
        qid = q.get("id")
        if qid:
            adjust_question_difficulty(qid)

    def _acc(q: Dict[str, Any]) -> float:
        seen = int(q.get("seen", 0) or 0)
        correct = int(q.get("correct", 0) or 0)
        return (correct / seen) if seen > 0 else 0.0

    weak_qs = [q for q in seen_qs if _acc(q) < 0.8]
    strong_qs = [q for q in seen_qs if _acc(q) >= 0.8]

    pool = weak_qs + new_qs + strong_qs
    random.shuffle(pool)
    return pool[:n]
