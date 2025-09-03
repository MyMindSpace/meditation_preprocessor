import json
import re
from pathlib import Path
from typing import Any, Dict, List


# ----------------------
# Diagnosis Processor (DP)
# ----------------------
# Input: Raw diagnosis data from the Data Input Layer (JSON files in this folder)
# Processing:
#   - Text Cleaning
#   - Tokenization
#   - Feature Engineering
# Output: Cleaned, tokenized, and engineered diagnosis data (diagnosis_processed.json)


DEFAULT_INPUT_FILE = "diagnosis_data.json"
DEFAULT_OUTPUT_FILE = "diagnosis_processed.json"


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "to", "in", "on",
    "for", "with", "as", "by", "from", "at", "is", "was", "were", "be", "been",
    "am", "are", "it", "this", "that", "these", "those", "i", "you", "he", "she",
    "they", "we", "me", "him", "her", "them", "my", "your", "his", "their", "our",
    "not", "no", "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "should", "would", "may", "might", "will", "shall", "than", "then"
}

SEVERITY_KEYWORDS = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "extreme": 4,
    "chronic": 3,
    "acute": 2,
    "occasional": 1,
    "frequent": 3,
    "constant": 4,
    "intermittent": 2,
}

MENTAL_HEALTH_TERMS = {
    "anxiety", "depression", "stress", "panic", "worry", "fear", "overwhelmed",
    "burnout", "trauma", "ptsd", "ocd", "bipolar", "mood", "emotional", "insomnia",
    "sleep"
}

PHYSICAL_SYMPTOMS = {
    "headache", "pain", "fatigue", "tired", "nausea", "dizzy", "tension", "muscle",
    "joint", "back", "neck", "chest"
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in STOPWORDS]
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens


def calculate_severity_score(tokens: List[str]) -> float:
    if not tokens:
        return 1.0
    scores = [SEVERITY_KEYWORDS[t] for t in tokens if t in SEVERITY_KEYWORDS]
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def extract_symptom_features(tokens: List[str]) -> Dict[str, Any]:
    mental = sum(1 for t in tokens if t in MENTAL_HEALTH_TERMS)
    physical = sum(1 for t in tokens if t in PHYSICAL_SYMPTOMS)
    total = len(tokens)
    return {
        "total_symptom_count": mental + physical,
        "mental_health_ratio": (mental / total) if total else 0.0,
        "physical_symptoms_ratio": (physical / total) if total else 0.0,
        "symptom_diversity": len(set(tokens) & (MENTAL_HEALTH_TERMS | PHYSICAL_SYMPTOMS)),
        "has_mental_health_terms": mental > 0,
        "has_physical_symptoms": physical > 0,
    }


def build_text_from_record(record: Dict[str, Any]) -> str:
    user_id = record.get("userId", "")
    diag_type = record.get("diagnosisType", "")
    symptoms = record.get("symptoms", [])
    timestamp = record.get("timestamp", "")
    if isinstance(symptoms, list):
        symptoms_text = ", ".join(symptoms)
    else:
        symptoms_text = str(symptoms)
    parts = [
        f"user {user_id}" if user_id else "",
        f"diagnosed with {diag_type}" if diag_type else "",
        f"symptoms: {symptoms_text}" if symptoms_text else "",
        f"timestamp: {timestamp}" if timestamp else "",
    ]
    return " ".join([p for p in parts if p]).strip()


def process_text(text: str) -> Dict[str, Any]:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    severity = calculate_severity_score(tokens)
    symptom_feats = extract_symptom_features(tokens)
    features = {
        "text_length": len(cleaned),
        "token_count": len(tokens),
        "severity_score": severity,
        **symptom_feats,
    }
    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "features": features,
    }


def load_input(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array of diagnosis records")
    return data


def save_output(output_path: Path, payload: List[Dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    cwd = Path.cwd()
    input_path = cwd / DEFAULT_INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_input(input_path)
    processed: List[Dict[str, Any]] = []

    print(f"Processing {len(records)} diagnosis records from {input_path}...\n")
    for rec in records:
        if not isinstance(rec, dict):
            text = clean_text(str(rec))
            proc = process_text(text)
            processed.append({"userId": None, **proc})
            continue

        text = build_text_from_record(rec)
        proc = process_text(text)
        processed.append({"userId": rec.get("userId"), **proc})

        # concise console summary
        feats = proc["features"]
        print(
            f"User {rec.get('userId','unknown')}: "
            f"len={feats['text_length']}, tokens={feats['token_count']}, "
            f"severity={feats['severity_score']:.2f}, "
            f"symptoms={feats['total_symptom_count']} (M {feats['mental_health_ratio']:.2f} / P {feats['physical_symptoms_ratio']:.2f}), "
            f"diversity={feats['symptom_diversity']}"
        )

    output_path = cwd / DEFAULT_OUTPUT_FILE
    save_output(output_path, processed)
    print(f"\nWrote processed output to {output_path}")


if __name__ == "__main__":
    main()


