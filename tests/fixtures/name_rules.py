"""Small name-matching module used in tests. Its constants are written the way
real projects often write them: plain values next to frozenset(...),
re.compile(...) and f-strings built from earlier constants."""

import re

MATCHED = "matched"
UNMATCHED = "unmatched"
QUOTE_LIMIT = 400
_SENTENCE_ZONE = 0.6

_SUFFIXES = frozenset(
    "inc corp ltd llc "
    "gmbh plc".split()
)
_LETTERS = r"a-z"
_WORD_RE = re.compile(f"[{_LETTERS}]+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?=\s)")


def clean(name: str) -> str:
    """Lower-case a name and drop trailing legal suffixes."""
    words = _WORD_RE.findall(name.lower())
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    return " ".join(words)


def match(applicant: str, company: str) -> str:
    return MATCHED if clean(applicant) == clean(company) else UNMATCHED
