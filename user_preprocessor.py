import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import re


DEFAULT_INPUT_FILE = "user_feedback.json"
DEFAULT_OUTPUT_FILE = "user_feedback_processed.json"


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "to", "in", "on",
    "for", "with", "as", "by", "from", "at", "is", "was", "were", "be", "been",
    "am", "are", "it", "this", "that", "these", "those", "i", "you", "he", "she",
    "they", "we", "me", "him", "her", "them", "my", "your", "his", "their", "our",
    "not", "no", "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "should", "would", "may", "might", "will", "shall", "than", "then"
}


def parse_ts(ts: str) -> datetime:
    # Handle ISO 8601 'Z'
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


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


def load_feedback(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array of feedback records")
    return data


def aggregate(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # per-user aggregation
    per_user: Dict[str, Any] = {}
    # overall daily aggregation
    overall_daily: Dict[str, List[int]] = {}

    for rec in records:
        user_id = rec.get("userId", "unknown")
        rating = int(rec.get("rating", 0) or 0)
        text = rec.get("feedbackText", "")
        ts_str = rec.get("timestamp") or ""
        try:
            ts = parse_ts(ts_str)
            day = ts.date().isoformat()
        except Exception:
            ts = None
            day = "unknown"

        overall_daily.setdefault(day, []).append(rating)

        bucket = per_user.setdefault(user_id, {
            "sessions": 0,
            "ratings": [],
            "firstFeedback": None,
            "lastFeedback": None,
            "daily": {},  # day -> [ratings]
            "keywords": {},
            "recentFeedback": [],  # keep last 5
        })

        bucket["sessions"] += 1
        bucket["ratings"].append(rating)
        if ts:
            if not bucket["firstFeedback"] or ts < parse_ts(bucket["firstFeedback"]):
                bucket["firstFeedback"] = ts.isoformat()
            if not bucket["lastFeedback"] or ts > parse_ts(bucket["lastFeedback"]):
                bucket["lastFeedback"] = ts.isoformat()
        bucket["daily"].setdefault(day, []).append(rating)

        # keywords from feedback
        tokens = tokenize(clean_text(text))
        for t in tokens:
            bucket["keywords"][t] = bucket["keywords"].get(t, 0) + 1

        # maintain recent feedbacks
        bucket["recentFeedback"].append({
            "timestamp": ts.isoformat() if ts else ts_str,
            "rating": rating,
            "feedbackText": text,
        })
        bucket["recentFeedback"] = sorted(
            bucket["recentFeedback"], key=lambda x: x.get("timestamp") or ""
        )[-5:]

    # finalize per-user stats
    for user_id, bucket in per_user.items():
        ratings = bucket["ratings"] or [0]
        bucket["avgRating"] = sum(ratings) / len(ratings)
        bucket["minRating"] = min(ratings)
        bucket["maxRating"] = max(ratings)
        # top keywords
        kws = sorted(bucket["keywords"].items(), key=lambda x: (-x[1], x[0]))[:15]
        bucket["topKeywords"] = [{"token": k, "count": c} for k, c in kws]
        # daily time series
        series = []
        for day, vals in sorted(bucket["daily"].items()):
            if day == "unknown":
                continue
            series.append({
                "date": day,
                "count": len(vals),
                "avgRating": (sum(vals) / len(vals)) if vals else 0.0,
            })
        bucket["timeSeriesDaily"] = series
        # cleanup
        del bucket["ratings"]
        del bucket["daily"]
        del bucket["keywords"]

    # finalize overall series
    overall_series = []
    for day, vals in sorted(overall_daily.items()):
        if day == "unknown":
            continue
        overall_series.append({
            "date": day,
            "count": len(vals),
            "avgRating": (sum(vals) / len(vals)) if vals else 0.0,
        })

    summary = {
        "totalUsers": len(per_user),
        "totalSessions": len(records),
        "overallAvgRating": (
            sum(v for vals in overall_daily.values() for v in vals) / max(1, sum(len(vals) for vals in overall_daily.values()))
        ),
    }

    overall = {
        "timeSeriesDaily": overall_series
    }

    return per_user, {"summary": summary, "overall": overall}


def main() -> None:
    cwd = Path.cwd()
    input_path = cwd / DEFAULT_INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_feedback(input_path)
    per_user, meta = aggregate(records)

    output = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        **meta,
        "perUser": per_user,
    }

    out_path = cwd / DEFAULT_OUTPUT_FILE
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Processed {meta['summary']['totalSessions']} sessions across {meta['summary']['totalUsers']} users.")
    print(f"Overall average rating: {meta['summary']['overallAvgRating']:.2f}")
    print(f"Wrote output: {out_path}")


if __name__ == "__main__":
    main()


