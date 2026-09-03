"""Instant answers for simple, patterned small talk - no LLM round trip.

The CLI backend adds several seconds of latency to every turn. Greetings and
other fixed patterns ("안녕", "이름이 뭐야") do not need a language model at all,
so we answer them from a small hand-curated + log-mined note file and skip the
broker entirely. That makes the most common exchanges feel instant.

Matching is deliberately conservative: only a near-exact match on a SHORT
utterance fires, so a real question is never shortcut by a canned answer. The
motion / vision / stop-word routers run first, so notes only ever see leftover
chat text.

The note file (config/quick_notes.json) is seeded by hand and grown by
broker/build_notes.py, which mines the conversation logs for frequent simple
utterances and reuses the reply the robot actually gave.
"""

import json
import logging
import os

logger = logging.getLogger('reachy.notes')

# Punctuation / fillers stripped before matching. Korean sentence-final
# particles are kept (they change meaning), but decoration is not.
_STRIP = ' \t\n.,!?~"\'“”‘’·…'
_NAME_TOKENS = ('리치야', '리치아', '리치',)


def normalize(text):
    """Collapse an utterance to a comparable key.

    Removes whitespace and decoration, drops the wake-name, lowercases Latin.
    """
    if not text:
        return ''
    s = text.strip().lower()
    for tok in _NAME_TOKENS:
        s = s.replace(tok, '')
    s = ''.join(ch for ch in s if ch not in _STRIP)
    return s


class QuickNotes(object):
    """Load note entries and look up an instant reply for an utterance.

    Each entry: {"patterns": [...], "reply": "...", "kind": "greeting"}.
    A lookup hits when the normalized utterance equals a normalized pattern,
    or (for short utterances) closely contains one - never on a long sentence.
    """

    # Only utterances this short are eligible for the looser "contains" match,
    # so a long question can only ever hit by an exact-normalized pattern.
    MAX_LOOSE = 12
    LOOSE_MARGIN = 3

    def __init__(self, entries):
        self._compiled = []
        for e in entries:
            reply = (e.get('reply') or '').strip()
            if not reply:
                continue
            pats = [normalize(p) for p in e.get('patterns', []) if normalize(p)]
            if not pats:
                continue
            self._compiled.append((pats, reply, e.get('kind', 'note')))
        logger.info('QuickNotes: %d entries loaded', len(self._compiled))

    @classmethod
    def load(cls, path):
        """Build from a JSON file; returns an empty (harmless) set on error."""
        try:
            with open(os.path.expanduser(path), encoding='utf-8') as f:
                data = json.load(f)
            entries = data.get('notes', data) if isinstance(data, dict) else data
            return cls(entries)
        except Exception:
            logger.warning('QuickNotes: could not load %s, running without it', path)
            return cls([])

    def __len__(self):
        return len(self._compiled)

    def lookup(self, text):
        """Return an instant reply for `text`, or None to fall through to the LLM."""
        norm = normalize(text)
        if not norm:
            return None

        # 1) Exact normalized match - always allowed, any length.
        for pats, reply, _ in self._compiled:
            if norm in pats:
                return reply

        # 2) Loose contains - only for short utterances, only for patterns that
        #    are nearly the whole utterance (so "여기 어디야" hits "여기어디"
        #    but "화장실이 어디 있는지" cannot hit a bare location pattern).
        if len(norm) <= self.MAX_LOOSE:
            for pats, reply, _ in self._compiled:
                for p in pats:
                    if len(p) >= 3 and p in norm and len(norm) - len(p) <= self.LOOSE_MARGIN:
                        return reply
        return None
