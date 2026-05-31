import os
import re
import json
import tempfile
import shutil
import time
from typing import List, Dict, Optional, Callable

import fitz
import docx
import requests
from pymongo import MongoClient

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
DOC_EXTENSIONS = {".pdf", ".docx"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".json", ".yaml", ".yml", ".sh", ".bat", ".css", ".html"}

HF_API_URL = "https://api-inference.huggingface.co/models/valhalla/t5-small-qg-hl"

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n\n".join(
        page.get_text("text") for page in doc if page.get_text("text")
    )


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS:
        return extract_text_from_txt(path)
    return ""


# ---------------------------------------------------------------------------
# Text cleaning & sentence splitting
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# IOB-inspired answer span extraction (from AQG repo approach)
# ---------------------------------------------------------------------------

def extract_answer_spans(sentence: str) -> List[str]:
    """
    Extracts candidate answer spans using the same logic as the AQG repo:
    named-entity-like capitalized phrases, numbers, and quoted strings.
    """
    candidates = []

    # Capitalized multi-word phrases (named entities)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", sentence):
        candidates.append(m.group(0))

    # Numbers with optional units
    for m in re.finditer(
        r"\b\d[\d,\.]*(?:\s+(?:million|billion|thousand|percent|km|kg|mph|years?|months?|days?))?\b",
        sentence,
    ):
        candidates.append(m.group(0).strip())

    # Quoted strings
    for m in re.finditer(r'"([^"]{3,60})"', sentence):
        candidates.append(m.group(1))

    # Deduplicate
    seen, unique = set(), []
    for c in candidates:
        if c not in seen and len(c) > 2:
            seen.add(c)
            unique.append(c)

    # Fallback: first two words
    if not unique:
        words = sentence.split()
        if len(words) >= 2:
            unique.append(" ".join(words[:2]))

    return unique[:3]


# ---------------------------------------------------------------------------
# HuggingFace Inference API — no model download, Streamlit Cloud safe
# ---------------------------------------------------------------------------

def generate_question_hf_api(context: str, answer: str, hf_token: str) -> Optional[str]:
    """
    Calls HuggingFace Inference API for valhalla/t5-small-qg-hl.
    Highlights the answer span with <hl> tokens exactly as the model expects.
    Free tier: ~30k requests/month, no GPU cost, no local memory.
    """
    highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
    prompt = f"generate question: {highlighted}"

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 64, "num_beams": 4},
    }

    for attempt in range(3):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                # Model is loading on HF side — wait and retry
                time.sleep(10)
                continue
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    question = data[0].get("generated_text", "").strip()
                    if question:
                        return question
            break
        except requests.RequestException:
            break

    return None


# ---------------------------------------------------------------------------
# Rule-based fallback (zero dependencies, always works)
# ---------------------------------------------------------------------------

def _rule_based_question(sentence: str, answer: str) -> str:
    s = sentence.strip()
    m = re.match(r"^([A-Za-z][^,]{2,40}?)\s+is\s+(.+)", s)
    if m and answer in s:
        return f"What is {m.group(1).strip()}?"
    if re.match(r"^\d", answer):
        return f"How many {answer.split()[-1] if answer.split() else 'units'} are mentioned?"
    return f"What is meant by \"{answer}\" in this context?"


# ---------------------------------------------------------------------------
# Main QA builder
# ---------------------------------------------------------------------------

def build_qa_pairs(
    text: str,
    source: str = "document",
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict[str, str]]:
    """
    Builds QA pairs from text.
    - If hf_token is provided: uses HuggingFace Inference API (smart questions).
    - Otherwise: falls back to rule-based generation (instant, offline).
    """
    text = clean_text(text)
    sentences = split_into_sentences(text)
    qas = []

    # Pre-compute all (sentence, answer) pairs to track progress accurately
    pairs = []
    for sentence in sentences:
        for answer in extract_answer_spans(sentence):
            pairs.append((sentence, answer))

    total = len(pairs)

    for idx, (sentence, answer) in enumerate(pairs):
        question = None
        if hf_token:
            question = generate_question_hf_api(sentence, answer, hf_token)
        if not question:
            question = _rule_based_question(sentence, answer)

        qas.append({
            "source": source,
            "context": sentence,
            "question": question,
            "answer": answer,
        })

        if progress_callback and total > 0:
            progress_callback((idx + 1) / total)

    return qas


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def save_jsonl(items: List[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

def save_to_mongodb(uri: str, database: str, collection: str, items: List[Dict]) -> Dict[str, int]:
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[database]
    coll = db[collection]
    result = coll.insert_many(items)
    client.close()
    return {"inserted_count": len(result.inserted_ids)}


def test_mongodb_connection(uri: str) -> bool:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False
