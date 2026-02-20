"""
Groq qualitative tagging helper utilities.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

try:
    from groq import Groq
except ImportError:
    Groq = None


QUAL_FIELDS = {
    "hook_type",
    "hook_emotion",
    "topic_primary",
    "technical_depth",
    "has_payoff",
    "has_numbers",
    "has_examples",
    "insider_language",
}


def get_groq_client() -> "Groq":
    """Return a Groq client instance using GROQ_API_KEY env var."""
    if Groq is None:
        raise ImportError(
            "groq package is not installed. Install with `pip install groq`."
        )
    return Groq()


def call_qual_v1_tags(
    transcript_text: str,
    system_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
) -> Dict[str, object]:
    """Call Groq LLM to generate qualitative tags for a transcript."""
    if not transcript_text.strip():
        raise ValueError("transcript_text must not be empty.")

    client = get_groq_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Return ONLY a JSON object with the qualitative fields defined in the schema. "
                "Here is the transcript:\n\n"
                f"{transcript_text.strip()}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=2048,
    )

    payload = _extract_json_dict(response)
    missing = QUAL_FIELDS.difference(payload.keys())
    if missing:
        raise ValueError(
            f"Groq qualitative response missing fields: {sorted(missing)}"
        )
    return payload


def compute_qual_v1_from_transcript(
    transcript_json: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, object]:
    """Extract transcript text and call call_qual_v1_tags."""
    transcript_text = _transcript_to_text(transcript_json)
    return call_qual_v1_tags(transcript_text, system_prompt)


def _extract_json_dict(response: Any) -> Dict[str, Any]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Groq qualitative response contained no choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ValueError("Groq qualitative response missing textual content.")

    content = _strip_code_fence(content.strip())
    json_text = _extract_json_block(content)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse Groq JSON: {exc}\nResponse: {content}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Groq qualitative response was not a JSON object.")
    return payload


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```", 1)[-1]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _transcript_to_text(transcript_json: Dict[str, Any]) -> str:
    tokens: List[str] = []

    words = transcript_json.get("words")
    if isinstance(words, list):
        tokens.extend(_extract_words_from_list(words))

    if not tokens:
        segments = transcript_json.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                seg_words = segment.get("words")
                if isinstance(seg_words, list):
                    tokens.extend(_extract_words_from_list(seg_words))
                else:
                    text = segment.get("text")
                    if isinstance(text, str):
                        tokens.append(text.strip())

    if not tokens:
        text_field = transcript_json.get("text")
        if isinstance(text_field, str):
            tokens.append(text_field.strip())

    transcript_text = " ".join(token for token in tokens if token).strip()
    if not transcript_text:
        raise ValueError("Transcript JSON does not contain textual content.")
    return transcript_text


def _extract_words_from_list(entries: Iterable[Any]) -> List[str]:
    words: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            word = entry.get("word")
            if isinstance(word, str):
                words.append(word.strip())
    return words
