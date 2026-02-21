"""
Quantitative features for Whisper transcript JSON payloads.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Sequence, Tuple

PUNCTUATION = ",.?!:;\"'()`"
VOWELS = set("aeiouy")
SENTENCE_BOUNDARY_RE = re.compile(r"([.!?]+)")
SYLLABLE_CLEAN_RE = re.compile(r"[^a-z]")

SINGLE_FILLERS = {"like", "um", "uh"}
FILLER_BIGRAMS = {("you", "know"), ("i", "mean")}

FIRST_PERSON = {
    "i", "i'm", "im", "ive", "i've", "me", "my", "mine",
    "we", "we're", "were", "weve", "we've", "us", "our", "ours",
}

SECOND_PERSON = {"you", "you're", "youre", "your", "yours", "u"}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "not",
    "no", "so", "if", "then", "that", "this", "these", "those", "there",
    "here", "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "just", "about", "also", "as",
}


def _compute_repetition_score(tokens: List[str]) -> float:
    """Fraction of content words (non-stopword) that appear 2+ times."""
    content_words = [t for t in tokens if t and t not in _STOPWORDS]
    if not content_words:
        return 0.0
    counts: Dict[str, int] = {}
    for w in content_words:
        counts[w] = counts.get(w, 0) + 1
    repeated = sum(1 for w in content_words if counts[w] >= 2)
    return repeated / len(content_words)


def compute_quant_v1(whisper_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute deterministic quantitative features from a Whisper transcript JSON payload.
    """
    duration_s = float(whisper_json.get("duration", 0.0) or 0.0)
    word_entries = list(_iter_word_entries(whisper_json))
    word_count = len(word_entries)

    lexical_tokens: List[str] = []
    text_tokens: List[str] = []
    for entry in word_entries:
        raw = entry.get("word", "")
        lexical_tokens.append(_normalize_token(raw))
        text_token = _normalize_token(raw, strip_punct=False)
        if text_token:
            text_tokens.append(text_token)

    hook_word_count = sum(
        1
        for entry in word_entries
        if (start := _safe_float(entry.get("start"))) is not None and start < 3.0
    )
    wpm = word_count / (duration_s / 60.0) if duration_s > 0 else 0.0
    hook_wpm = hook_word_count / (3.0 / 60.0) if hook_word_count > 0 else 0.0

    filler_count = _count_fillers(lexical_tokens)
    filler_density = filler_count / word_count if word_count > 0 else 0.0

    sentences = _extract_sentences(" ".join(text_tokens))
    num_sentences = len(sentences)

    first_person_count = sum(1 for token in lexical_tokens if token in FIRST_PERSON)
    second_person_count = sum(1 for token in lexical_tokens if token in SECOND_PERSON)

    first_person_ratio = first_person_count / word_count if word_count > 0 else 0.0
    second_person_ratio = second_person_count / word_count if word_count > 0 else 0.0

    syllable_count = sum(_count_syllables(token) for token in lexical_tokens if token)
    if word_count == 0 or num_sentences == 0:
        reading_level = 0.0
    else:
        reading_level = (
            0.39 * (word_count / num_sentences)
            + 11.8 * (syllable_count / word_count)
            - 15.59
        )

    # --- New v2 computed features ---

    # Lexical diversity: type-token ratio
    non_empty = [t for t in lexical_tokens if t]
    lexical_diversity = len(set(non_empty)) / len(non_empty) if non_empty else 0.0

    # Average sentence length
    avg_sentence_length = word_count / num_sentences if num_sentences > 0 else 0.0

    # Hook density: hook_wpm / rest_wpm (front-loading measure)
    rest_duration = duration_s - 3.0
    if rest_duration < 1.0:
        hook_density = 1.0
    else:
        rest_word_count = word_count - hook_word_count
        rest_wpm = rest_word_count / (rest_duration / 60.0) if rest_duration > 0 else 0.0
        hook_density = hook_wpm / rest_wpm if rest_wpm > 0 else 1.0

    # Repetition score
    repetition_score = _compute_repetition_score(lexical_tokens)

    return {
        "duration_s": duration_s,
        "word_count": word_count,
        "wpm": wpm,
        "hook_word_count": hook_word_count,
        "hook_wpm": hook_wpm,
        "filler_count": filler_count,
        "filler_density": filler_density,
        "first_person_ratio": first_person_ratio,
        "second_person_ratio": second_person_ratio,
        "num_sentences": num_sentences,
        "reading_level": reading_level,
        "lexical_diversity": lexical_diversity,
        "avg_sentence_length": avg_sentence_length,
        "hook_density": hook_density,
        "repetition_score": repetition_score,
    }


def _iter_word_entries(whisper_json: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    words = whisper_json.get("words")
    if isinstance(words, list):
        for entry in words:
            if isinstance(entry, dict) and "word" in entry:
                yield entry

    segments = whisper_json.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            seg_words = segment.get("words")
            if isinstance(seg_words, list):
                for entry in seg_words:
                    if isinstance(entry, dict) and "word" in entry:
                        yield entry


def _normalize_token(token: Any, strip_punct: bool = True) -> str:
    if not isinstance(token, str):
        return ""
    normalized = token.strip()
    if strip_punct:
        normalized = normalized.strip(PUNCTUATION)
    return normalized.lower()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_fillers(tokens: Sequence[str]) -> int:
    single_count = sum(1 for token in tokens if token in SINGLE_FILLERS)
    bigram_count = 0
    for first, second in zip(tokens, tokens[1:]):
        if first and second and (first, second) in FILLER_BIGRAMS:
            bigram_count += 1
    return single_count + bigram_count


def _extract_sentences(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    pieces = SENTENCE_BOUNDARY_RE.split(text)
    sentences: List[Tuple[str, str]] = []
    for idx in range(0, len(pieces), 2):
        chunk = pieces[idx].strip()
        if not chunk:
            continue
        delimiter = pieces[idx + 1] if idx + 1 < len(pieces) else ""
        sentences.append((chunk, delimiter.strip()))
    return sentences


def _count_syllables(word: str) -> int:
    if not word:
        return 0
    cleaned = SYLLABLE_CLEAN_RE.sub("", word.lower())
    if not cleaned:
        return 1
    syllables = 0
    previous_is_vowel = False
    for char in cleaned:
        is_vowel = char in VOWELS
        if is_vowel and not previous_is_vowel:
            syllables += 1
        previous_is_vowel = is_vowel
    return max(syllables, 1)
