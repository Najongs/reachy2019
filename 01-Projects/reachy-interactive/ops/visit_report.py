"""방문 하나하나를 사람·사진·대화로 묶어 본다.

'이 사람이 누구인지 알고 그에 맞춰 대응한다'가 목표다. 그러려면 먼저
**한 방문에서 누가 있었고 무슨 말이 오갔는지**가 한 줄로 이어져야 한다.
사진(person_db)과 대화(events.jsonl)는 `visit_id` 로 이어진다.

나중에 군집이 방문마다 사람(identity_id)을 붙이면, 이 보고서가 그대로
'이 사람은 지난번에 이런 이야기를 했다'가 된다.

사용:
    python3 ops/visit_report.py                  # 최근 방문들
    python3 ops/visit_report.py --talked         # 대화가 있었던 방문만
    python3 ops/visit_report.py --day 2026-09-04
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                    '04-Archives', 'person-dataset'))
DEFAULT_DB = os.path.join(BASE, 'persons', 'persons.db')
DEFAULT_LOGS = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                            '04-Archives',
                                            'conversation-logs', 'pi'))


def load_turns(log_root):
    """방문 id -> 그 방문에서 오간 말들."""
    turns = defaultdict(list)
    for path in glob.glob(os.path.join(log_root, '*', 'events.jsonl')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                visit = d.get('visit_id')
                if not visit or d.get('kind') not in ('chat', 'note',
                                                      'vision_chat', 'motion'):
                    continue
                turns[visit].append(d)
    return turns


def load_visits(db_path):
    """방문 id -> 사진·사람 정보."""
    conn = sqlite3.connect(db_path)
    visits = {}
    for vid, day, n, first, last, inter in conn.execute(
            '''SELECT visit_id, day, COUNT(*), MIN(ts), MAX(ts), MAX(interacted)
               FROM persons WHERE visit_id IS NOT NULL
               GROUP BY visit_id'''):
        visits[vid] = {'day': day, 'photos': n, 'first': first, 'last': last,
                       'interacted': inter, 'people': 0, 'identity': None}

    # 사람 수와 (있다면) 군집이 붙인 신원
    try:
        for vid, people, ident in conn.execute(
                '''SELECT p.visit_id, COUNT(b.id),
                          GROUP_CONCAT(DISTINCT b.identity_id)
                   FROM persons p JOIN person_boxes b ON b.photo_id = p.id
                   GROUP BY p.visit_id'''):
            if vid in visits:
                visits[vid]['people'] = people
                visits[vid]['identity'] = ident
    except sqlite3.OperationalError:
        pass                        # person_boxes 가 아직 없다
    return visits


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--logs', default=DEFAULT_LOGS)
    ap.add_argument('--day', help='이 날짜만 (YYYY-MM-DD)')
    ap.add_argument('--talked', action='store_true', help='대화가 있었던 방문만')
    ap.add_argument('--limit', type=int, default=20, help='최근 몇 개까지')
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print('DB 가 없습니다: %s' % args.db)
        print('먼저:  bash ops/daily_update.sh')
        return 1

    visits = load_visits(args.db)
    turns = load_turns(args.logs)
    if not visits:
        print('방문 기록이 없습니다.')
        return 0

    keys = sorted(visits, key=lambda v: visits[v]['first'] or '', reverse=True)
    if args.day:
        keys = [k for k in keys if visits[k]['day'] == args.day]
    if args.talked:
        keys = [k for k in keys if turns.get(k)]

    linked = sum(1 for k in visits if turns.get(k))
    print('방문 %d개 · 그중 대화가 이어진 방문 %d개' % (len(visits), linked))
    if not linked:
        print('  (대화 기록에 visit_id 가 붙기 시작한 뒤의 방문부터 이어집니다)')
    print()

    for vid in keys[:args.limit]:
        v = visits[vid]
        said = turns.get(vid, [])
        who = v['identity'] or '아직 모름'
        print('%s  사진 %d장 · 사람 %d명 · 신원 %s'
              % (v['first'] or vid, v['photos'], v['people'], who))
        for t in said[:6]:
            if t.get('kind') == 'motion':
                print('     [동작] %s' % (t.get('text') or '')[:50])
            else:
                print('     사람: %-28s → 리치: %s'
                      % ((t.get('text') or '')[:28],
                         (t.get('reply') or '')[:34]))
        if len(said) > 6:
            print('     ... 외 %d턴' % (len(said) - 6))
        if not said:
            print('     (대화 없음 - 지나가기만 했다)')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
