"""Mine conversation logs for frequent simple exchanges -> quick-answer notes.

Reachy answers greetings and other fixed patterns instantly from
config/quick_notes.json instead of paying the CLI round-trip (see
robot/quick_notes.py). This script grows that file from real usage:

  1. Scan every TurnLogger events.jsonl under a logs dir.
  2. Keep SHORT chat utterances (a greeting, a one-liner - not a real question).
  3. For each utterance seen more than once, reuse the reply the robot ACTUALLY
     gave most often (never a newly invented one, so no hallucination sneaks
     into the cache). Replies that promise an action are dropped - the persona
     forbids them and we must not cache a promise.
  4. Write proposals to config/quick_notes.proposed.json for review, and with
     --merge, add the safe/consistent ones into config/quick_notes.json
     (existing patterns are never overwritten).

Run on the DGX after refreshing logs (Pi sessions must be copied in first):

    python3 build_notes.py                 # write proposals only
    python3 build_notes.py --merge         # also promote the safe ones
"""

import argparse
import collections
import glob
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import para  # PARA 기준 경로 (위치 계산은 para.py 한 곳에만)
CONFIG = os.path.join(para.PROJECT, 'config')
DEFAULT_LOGS = para.CONVERSATION_LOGS

NOTES_PATH = os.path.join(CONFIG, 'quick_notes.json')
PROPOSED_PATH = os.path.join(CONFIG, 'quick_notes.proposed.json')

# Only short utterances are note-worthy; a long one is a real question.
MAX_LEN = 12
# How many times an utterance must appear to be worth caching.
MIN_COUNT = 2
# For --merge: stricter, and the dominant reply must be this fraction of hits.
MERGE_MIN_COUNT = 3
MERGE_DOMINANCE = 0.6

# A cached reply must never promise an action the words alone cannot perform.
_PROMISE_MARKERS = ('할게요', '해볼게요', '하겠습니다', '해드릴게요', '움직여볼게요',
                    '해볼게', '할게', '볼게요')
# Utterances that are really motion/vision requests are handled by other
# routers; if one slipped into the chat log, do not cache it as small talk.
_ACTION_MARKERS = ('들어', '올려', '내려', '흔들', '움직', '뻗', '펼쳐', '접어',
                   '돌려', '가리', '인사', '악수', '만세', '박수', '젖혀', '숙여',
                   '보여', '보고', '읽어', '뭐가 보', '무슨 색')

_STRIP = ' \t\n.,!?~"\'“”‘’·…'
_NAME_TOKENS = ('리치야', '리치아', '리치')


def normalize(text):
    """Same normalization as robot/quick_notes.py (kept in sync by hand)."""
    if not text:
        return ''
    s = text.strip().lower()
    for tok in _NAME_TOKENS:
        s = s.replace(tok, '')
    return ''.join(ch for ch in s if ch not in _STRIP)


def _iter_turns(logs_dir):
    """Yield (text, reply) for every chat/note turn in every events.jsonl."""
    pattern = os.path.join(logs_dir, '**', 'events.jsonl')
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get('kind') not in ('chat', 'note', 'vision_chat'):
                        continue
                    text = (e.get('text') or '').strip()
                    reply = (e.get('reply') or '').strip()
                    if text and reply:
                        yield text, reply
        except OSError:
            continue


def _looks_action(text):
    return any(m in text for m in _ACTION_MARKERS)


def _is_promise(reply):
    return any(m in reply for m in _PROMISE_MARKERS)


def mine(logs_dir):
    """Return candidate note entries mined from the logs, most frequent first."""
    # normalized utterance -> {'texts': Counter, 'replies': Counter}
    buckets = collections.defaultdict(
        lambda: {'texts': collections.Counter(), 'replies': collections.Counter()})

    for text, reply in _iter_turns(logs_dir):
        norm = normalize(text)
        if not norm or len(norm) > MAX_LEN:
            continue
        if _looks_action(text) or _is_promise(reply):
            continue
        b = buckets[norm]
        b['texts'][text] += 1
        b['replies'][reply] += 1

    candidates = []
    for norm, b in buckets.items():
        count = sum(b['replies'].values())
        if count < MIN_COUNT:
            continue
        top_reply, top_n = b['replies'].most_common(1)[0]
        top_text = b['texts'].most_common(1)[0][0]
        candidates.append({
            'norm': norm,
            'example': top_text,
            'count': count,
            'reply': top_reply,
            'dominance': round(top_n / count, 2),
        })

    candidates.sort(key=lambda c: c['count'], reverse=True)
    return candidates


def _load_json(path, default):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _existing_norms(notes):
    """Every normalized pattern already covered by the note file."""
    seen = set()
    for e in notes.get('notes', []):
        for p in e.get('patterns', []):
            seen.add(normalize(p))
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', default=DEFAULT_LOGS,
                        help='directory tree of events.jsonl logs to mine')
    parser.add_argument('--merge', action='store_true',
                        help='promote safe, consistent proposals into quick_notes.json')
    args = parser.parse_args()

    candidates = mine(args.logs)
    notes = _load_json(NOTES_PATH, {'notes': []})
    covered = _existing_norms(notes)

    fresh = [c for c in candidates if c['norm'] not in covered]

    with open(PROPOSED_PATH, 'w', encoding='utf-8') as f:
        json.dump({'proposals': fresh}, f, ensure_ascii=False, indent=2)
    print('{} candidate utterances, {} not yet covered -> {}'.format(
        len(candidates), len(fresh), PROPOSED_PATH))
    for c in fresh[:20]:
        print('  x{count} ({dominance}) {example!r} -> {reply!r}'.format(**c))

    if not args.merge:
        if fresh:
            print('\nReview the proposals, then rerun with --merge to promote them.')
        return

    promoted = 0
    for c in fresh:
        if c['count'] < MERGE_MIN_COUNT or c['dominance'] < MERGE_DOMINANCE:
            continue
        notes.setdefault('notes', []).append({
            'kind': 'mined',
            'patterns': [c['example']],
            'reply': c['reply'],
        })
        covered.add(c['norm'])
        promoted += 1

    if promoted:
        with open(NOTES_PATH, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        print('\nPromoted {} new notes into {}'.format(promoted, NOTES_PATH))
    else:
        print('\nNothing met the merge bar (count>={}, dominance>={}).'.format(
            MERGE_MIN_COUNT, MERGE_DOMINANCE))


if __name__ == '__main__':
    main()
