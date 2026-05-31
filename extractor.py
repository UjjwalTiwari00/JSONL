"""
Core feature extraction ported directly from:
  gsasikiran/automatic-question-generation — main/squad_parse.py

The extract_features() function is used as-is to produce:
  - BIO tags  (B/I/O per token marking the answer span)
  - LEX tags  (POS_NER_CASE per token, e.g. NOUN_PERSON_UP)
  - tokenized context (lowercased, spaCy-tokenized)

These features are fed to the HuggingFace T5 QG API as the highlighted
context, giving it the same quality signal the AQG model was trained on.
"""

import os
import re
import json
import string
import time
from typing import List, Dict, Optional, Callable

import fitz
import docx
import requests
from pymongo import MongoClient

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
DOC_EXTENSIONS  = {".pdf", ".docx"}

HF_API_URL = "https://api-inference.huggingface.co/models/valhalla/t5-small-qg-hl"

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n\n".join(
        page.get_text("text") for page in doc if page.get_text("text").strip()
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
    if ext in TEXT_EXTENSIONS:
        return extract_text_from_txt(path)
    return ""


# ---------------------------------------------------------------------------
# spaCy — loaded once
# ---------------------------------------------------------------------------

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# AQG core — extract_features() ported directly from squad_parse.py
# ---------------------------------------------------------------------------

def extract_features(text: str, answer: str, answer_start: int, nlp):
    """
    Exact port of extract_features() from gsasikiran/automatic-question-generation.

    Splits context into left / answer span / right, runs spaCy on each,
    and returns POS, NER, CASE, BIO, and lowercased tokenized context —
    the same feature set the AQG seq2seq model was trained on.
    """
    left  = text[0 : answer_start]
    ans   = text[answer_start : answer_start + len(answer) + 1]
    right = text[answer_start + len(answer) + 1 : len(text) + 1]

    pos_list, ner_list, case_list, bio_list, tokenized = [], [], [], [], []

    for token in nlp(left):
        if token.text != "" and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos_list.append(token.pos_)
            ner_list.append(token.ent_type_ if token.ent_type_ else "O")
            case_list.append("UP" if token.text[0].isupper() else "LOW")
            bio_list.append("O")

    for token in nlp(ans):
        if token.text != "" and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos_list.append(token.pos_)
            ner_list.append(token.ent_type_ if token.ent_type_ else "O")
            case_list.append("UP" if token.text[0].isupper() else "LOW")
            # BIO: first answer token → B, rest → I
            bio_list.append("B" if token.i == 0 else "I")

    for token in nlp(right):
        if token.text != "" and not token.text.isspace():
            tokenized.append(token.text.lower())
            pos_list.append(token.pos_)
            ner_list.append(token.ent_type_ if token.ent_type_ else "O")
            case_list.append("UP" if token.text[0].isupper() else "LOW")
            bio_list.append("O")

    # LEX = POS_NER_CASE per token, e.g. "NOUN_PERSON_UP"
    lex_list = [f"{p}_{n}_{c}" for p, n, c in zip(pos_list, ner_list, case_list)]

    return (
        " ".join(pos_list),
        " ".join(ner_list),
        " ".join(case_list),
        " ".join(bio_list),    # BIO
        " ".join(lex_list),    # LEX
        " ".join(tokenized),   # lowercased tokenized context
    )


# ---------------------------------------------------------------------------
# Answer span candidates — spaCy NER + noun chunks (same signal AQG used)
# ---------------------------------------------------------------------------

def extract_candidate_answers(sentence: str, nlp) -> List[Dict]:
    """
    Uses spaCy named entities and noun chunks as answer candidates —
    exactly the type of spans SQuAD answers consist of.
    Returns [{"answer": str, "answer_start": int}, ...]
    """
    doc = nlp(sentence)
    candidates = []
    seen = set()

    # Named entities first — highest quality candidates
    for ent in doc.ents:
        text = ent.text.strip().strip(string.punctuation).strip()
        if text and text not in seen and len(text) > 1:
            seen.add(text)
            candidates.append({"answer": text, "answer_start": ent.start_char})

    # Noun chunks as fallback
    for chunk in doc.noun_chunks:
        text = chunk.text.strip().strip(string.punctuation).strip()
        if text and text not in seen and len(text) > 1:
            seen.add(text)
            candidates.append({"answer": text, "answer_start": chunk.start_char})

    return candidates[:4]


# ---------------------------------------------------------------------------
# Text cleaning & sentence splitting (using spaCy sentencizer)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_sentences(text: str, nlp) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]


# ---------------------------------------------------------------------------
# HuggingFace Inference API — T5 highlight-based QG
# Input format: "generate question: <hl> answer <hl> context"
# ---------------------------------------------------------------------------

def generate_question_hf_api(context: str, answer: str, hf_token: str) -> Optional[str]:
    # Highlight the answer span — matches t5-small-qg-hl training format
    highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
    prompt = f"generate question: {highlighted}"

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 64, "num_beams": 4},
        "options":    {"wait_for_model": True},
    }

    for _ in range(3):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                time.sleep(15)
                continue
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    q = data[0].get("generated_text", "").strip()
                    if q:
                        return q
            break
        except requests.RequestException:
            break
    return None


# ---------------------------------------------------------------------------
# Rule-based fallback — uses NER type from LEX to pick the right question word
# ---------------------------------------------------------------------------

_NER_TO_QWORD = {
    "PERSON":     "Who",
    "ORG":        "Which organization",
    "GPE":        "Where",
    "LOC":        "Where",
    "DATE":       "When",
    "TIME":       "When",
    "MONEY":      "How much",
    "CARDINAL":   "How many",
    "ORDINAL":    "Which",
    "NORP":       "Which group",
    "FAC":        "Where",
    "EVENT":      "What event",
    "WORK_OF_ART":"What",
    "LAW":        "What law",
    "LANGUAGE":   "What language",
    "QUANTITY":   "How much",
    "PERCENT":    "What percentage",
}

def _rule_based_question(sentence: str, answer: str, bio: str, lex: str) -> str:
    bio_tokens = bio.split()
    lex_tokens = lex.split()

    for tok_bio, tok_lex in zip(bio_tokens, lex_tokens):
        if tok_bio in ("B", "I"):
            parts = tok_lex.split("_")
            if len(parts) == 3 and parts[1] != "O":
                qword = _NER_TO_QWORD.get(parts[1])
                if qword:
                    return f"{qword} is \"{answer}\"?"

    return f"What is \"{answer}\" in this context?"


# ---------------------------------------------------------------------------
# Main QA builder
# ---------------------------------------------------------------------------

def build_qa_pairs(
    text: str,
    source: str = "document",
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict]:
    """
    Full pipeline:
    1. Clean + sentence-split with spaCy
    2. Extract named-entity / noun-chunk answer spans (spaCy)
    3. Run extract_features() — exact AQG port — to get BIO + LEX per token
    4. Generate question via HF T5 API (highlighted context) or rule-based fallback
    """
    nlp = _get_nlp()
    text = clean_text(text)
    sentences = split_into_sentences(text, nlp)

    # Build all (sentence, candidate) pairs upfront for accurate progress bar
    pairs = []
    for sentence in sentences:
        for candidate in extract_candidate_answers(sentence, nlp):
            pairs.append((sentence, candidate))

    qas = []
    total = len(pairs)

    for idx, (sentence, candidate) in enumerate(pairs):
        answer       = candidate["answer"]
        answer_start = candidate["answer_start"]

        try:
            _, _, _, bio, lex, tokenized_ctx = extract_features(sentence, answer, answer_start, nlp)
        except Exception:
            bio, lex, tokenized_ctx = "", "", sentence.lower()

        question = None
        if hf_token:
            question = generate_question_hf_api(sentence, answer, hf_token)
        if not question:
            question = _rule_based_question(sentence, answer, bio, lex)

        qas.append({
            "source":   source,
            "context":  sentence,
            "question": question,
            "answer":   answer,
            "bio":      bio,
            "lex":      lex,
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

def save_to_mongodb(uri: str, database: str, collection: str, items: List[Dict]) -> Dict:
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db     = client[database]
    coll   = db[collection]
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
