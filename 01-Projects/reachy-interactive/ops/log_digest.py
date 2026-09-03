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

    # -- STT engine + LLM latency (drift over time) --------------------------
    engines = collections.Counter(e.get('engine') for e in events
                                  if e.get('engine'))
    lats = [e['latency_s'] for e in events
            if isinstance(e.get('latency_s'), (int, float))]
    if engines or lats:
        line = '\n[STT/지연] '
        if engines:
            line += '엔진: ' + ', '.join('{} {}'.format(k, n)
                                        for k, n in engines.most_common())
        if lats:
            lats.sort()
            line += '  |  LLM 지연: 중앙 {:.1f}s / 최대 {:.1f}s (n={})'.format(
                lats[len(lats) // 2], lats[-1], len(lats))
        print(line)

    # -- mic level distribution (tune --fixed-energy) ------------------------
    _rms_report(args.logs_dir)


def _rms_report(logs_dir):
    """Captured-audio RMS distribution from voice_chat.log, for mic tuning."""
    path = os.path.join(logs_dir, 'voice_chat.log')
    if not os.path.exists(path):
        return
    import re
    rms, thr = [], []
    pat = re.compile(r'captured RMS (\d+) \(threshold (\d+)\)')
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    rms.append(int(m.group(1)))
                    thr.append(int(m.group(2)))
    except OSError:
        return
    if not rms:
        return
    rms.sort()
    n = len(rms)
    pct = lambda p: rms[min(n - 1, int(n * p))]
    cur_thr = thr[-1] if thr else 0
    print('\n[마이크 감지 레벨 (captured RMS, n={})]'.format(n))
    print('  분포: p10 {} · 중앙 {} · p90 {} · 최대 {}'.format(
        pct(0.1), pct(0.5), pct(0.9), rms[-1]))
    print('  현재 threshold ~{} 기준: 아래 {}건(소음성) / 위 {}건(발화성)'.format(
        cur_thr, sum(1 for r in rms if r < cur_thr),
        sum(1 for r in rms if r >= cur_thr)))
    print('  → 소음 많으면 --fixed-energy 를 중앙({})~p90({}) 사이로 올리기'.format(
        pct(0.5), pct(0.9)))


if __name__ == '__main__':
    main()
