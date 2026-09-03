"""One-day activity digest from the pulled Pi logs.

Reads every TurnLogger events.jsonl under a logs dir and prints a compact,
reviewable summary: how much happened, hallway foot traffic, what people said,
which motions failed, and the most common new utterances worth turning into
quick-notes. Run by ops/daily_update.sh after pulling the Pi logs.

    python3 log_digest.py <logs_dir> [--since YYYY-MM-DD]
"""

import argparse
import collections
import glob
import json
import os


def _iter_events(logs_dir, since=None):
    for path in glob.glob(os.path.join(logs_dir, '**', 'events.jsonl'),
                          recursive=True):
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if since and (e.get('ts') or '') < since:
                        continue
                    yield e
        except OSError:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs_dir')
    ap.add_argument('--since', help='only events with ts >= this (YYYY-MM-DD)')
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()

    events = list(_iter_events(args.logs_dir, args.since))
    if not events:
        print('  (기간 내 이벤트 없음)')
        return

    kinds = collections.Counter(e.get('kind', '?') for e in events)
    span = (min(e.get('ts', '') for e in events),
            max(e.get('ts', '') for e in events))

    print('기간: {}  ~  {}'.format(span[0], span[1]))
    print('총 이벤트 {}건: {}'.format(
        len(events), ', '.join('{} {}'.format(k, n) for k, n in kinds.most_common())))

    # -- hallway foot traffic ------------------------------------------------
    hall = [e for e in events if e.get('kind') == 'hallway']
    if hall:
        ev = collections.Counter(e.get('event') for e in hall)
        stays = [e['stayed_s'] for e in hall
                 if e.get('event') == 'left' and isinstance(e.get('stayed_s'), (int, float))]
        greeted = sum(1 for e in hall if e.get('event') == 'left' and e.get('greeted'))
        left_n = ev.get('left', 0)
        print('\n[복도] 등장 {} · 음성인사 {} · 무음웨이브 {} · 말걸기 {} · 떠남 {}'.format(
            ev.get('appeared', 0), ev.get('greeted', 0),
            ev.get('greeted_silent', 0), ev.get('prompted', 0), left_n))
        if stays:
            print('       평균 체류 {:.0f}s (최대 {:.0f}s), 인사받고 떠난 비율 {}/{}'.format(
                sum(stays) / len(stays), max(stays), greeted, left_n))

    # -- what people said ----------------------------------------------------
    said = collections.Counter()
    for e in events:
        if e.get('kind') in ('chat', 'note', 'vision_chat') and e.get('text'):
            said[e['text'].strip()] += 1
    if said:
        print('\n[자주 나온 대화 발화]')
        for text, n in said.most_common(args.top):
            mark = ' *반복*' if n > 1 else ''
            print('  {:2}x  {}{}'.format(n, text, mark))

    # -- motion outcomes -----------------------------------------------------
    motions = [e for e in events if e.get('kind') == 'motion']
    if motions:
        outc = collections.Counter(e.get('outcome', '?') for e in motions)
        print('\n[동작] {}건: {}'.format(
            len(motions), ', '.join('{} {}'.format(k, n) for k, n in outc.most_common())))
        bad = [e for e in motions
               if e.get('outcome') in ('rejected', 'broker_failed', 'failed')]
        for e in bad[:8]:
            print('  ! {}: {!r} ({})'.format(
                e.get('outcome'), (e.get('text') or '')[:30],
                (e.get('reason') or '')[:40]))

    # -- utterances that fell through to the slow LLM repeatedly -------------
    repeated_chat = [(t, n) for t, n in said.most_common() if n >= 2]
    if repeated_chat:
        print('\n[노트 후보(반복된 대화 발화 - build_notes 가 답변 재사용 검토)]')
        for t, n in repeated_chat[:args.top]:
            print('  {}x  {}'.format(n, t))


if __name__ == '__main__':
    main()
