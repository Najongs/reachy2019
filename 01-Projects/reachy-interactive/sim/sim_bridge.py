"""시뮬 <-> 실물 다리. 두 방향 모두 '제안까지만' - 적용은 사람이 검토한다.

방향 1 (--export): 시뮬에서 학습한 행동 파라미터를 실물 후보로 승격 제안.
  검증 배터리(다양한 테이블 높이 x 태스크)를 통과해야 제안이 만들어진다:
    충돌 0 필수 + 성공률 문턱 + 품질 문턱
  -> config/behavior_real.proposed.json (파라미터 + 검증 성적 + 근거).
  quick_notes.proposed.json 과 같은 검토-승격 패턴이다. 실물 반영은
  실물 서보 런타임이 생길 때 이 파일을 사람이 검토해 옮긴다.

방향 2 (--absorb): 실물에서 쌓인 로그를 뒤져 시뮬에 반영할 것을 뽑는다.
  지금 실물이 주는 것: 사람들이 실제로 시키는 동작 명령 분포(제스처
  학습 분기의 명령 생성이 실수요를 따라가게), 동작 실패 사유.
  -> config/real_commands.json + 요약 출력.

사용:
    python3 sim/sim_bridge.py --export            # 검증 후 제안 생성
    python3 sim/sim_bridge.py --absorb            # 실물 로그 -> 시뮬 반영거리
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import para  # PARA 기준 경로 (위치 계산은 para.py 한 곳에만)
CONFIG = os.path.join(para.PROJECT, 'config')
PI_LOGS = para.PI_LOGS


def export(n_tasks=6, min_success=5, min_quality=70, natural_min=65):
    import sim_tasks as T
    import sim_improve as I
    import sim_agent as A

    print('검증 배터리: 태스크 %d개 (높이 다양), 기준 충돌0 + 성공>=%d + 품질>=%d'
          % (n_tasks, min_success, min_quality))
    tasks = T.generate_feasible(n_tasks, seed=8117,
                                kinds=('reach',) * (n_tasks - 1) + ('look',))
    results = []
    for t in tasks:
        stage, q, _ = I.episode(t, natural_min=natural_min)
        results.append({'kind': t['kind'], 'table': t.get('table_top'),
                        'stage': stage, 'quality': q.get('score'),
                        'touch_speed_cm': q.get('touch_speed_cm'),
                        'final_cm': q.get('final_cm')})
        print('  %-5s 테이블 %.2f: %-9s 품질 %s' % (
            t['kind'], t.get('table_top', 0), stage, q.get('score')))

    n_coll = sum(1 for r in results if r['stage'] == 'collision')
    n_succ = sum(1 for r in results if r['stage'] == 'success')
    quals = [r['quality'] for r in results if r['quality']]
    avg_q = sum(quals) / max(len(quals), 1)
    ok = n_coll == 0 and n_succ >= min_success and avg_q >= min_quality
    print('배터리: 충돌%d 성공%d/%d 평균품질 %.0f -> %s'
          % (n_coll, n_succ, len(results), avg_q,
             '통과' if ok else '탈락 (제안 안 만듦)'))
    if not ok:
        return False

    out = os.path.join(CONFIG, 'behavior_real.proposed.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({
            'params': dict(A.BEHAVIOR),
            'validated': {'tasks': results, 'collisions': n_coll,
                          'success': '%d/%d' % (n_succ, len(results)),
                          'avg_quality': round(avg_q, 1),
                          'natural_min': natural_min},
            'note': '시뮬 검증 통과 제안. 실물 적용 전 사람이 검토할 것 - '
                    '특히 속도 상한과 작업영역. 모터 켠 상태의 감독 시연 필수.',
            'created': time.strftime('%Y-%m-%d %H:%M'),
        }, fh, ensure_ascii=False, indent=1)
    print('제안 저장: %s' % os.path.normpath(out))
    return True


def absorb(since=None):
    rows = []
    for p in glob.glob(os.path.join(PI_LOGS, '**', 'events.jsonl'),
                       recursive=True):
        try:
            with open(p, encoding='utf-8') as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if since and (e.get('ts') or '') < since:
                        continue
                    rows.append(e)
        except OSError:
            continue

    import collections
    motions = [r for r in rows if r.get('kind') == 'motion']
    cmd = collections.Counter((r.get('text') or '').strip()
                              for r in motions if r.get('text'))
    fails = collections.Counter(
        r.get('outcome') for r in motions
        if r.get('outcome') not in (None, 'ok', 'executed'))
    print('실물 이벤트 %d건, 동작 요청 %d건' % (len(rows), len(motions)))
    print('자주 시키는 동작 (제스처 학습의 명령 시드로 쓴다):')
    for text, n in cmd.most_common(10):
        print('  %2d x %s' % (n, text[:50]))
    if fails:
        print('실패 사유 분포:', dict(fails))

    out = os.path.join(CONFIG, 'real_commands.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'commands': [{'text': t, 'count': n}
                                for t, n in cmd.most_common(50)],
                   'failures': dict(fails),
                   'events': len(rows),
                   'updated': time.strftime('%Y-%m-%d %H:%M')},
                  fh, ensure_ascii=False, indent=1)
    print('저장: %s (제스처 분기 sim_scenarios 가 명령 시드로 읽게 한다)'
          % os.path.normpath(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--absorb', action='store_true')
    ap.add_argument('--since')
    args = ap.parse_args()
    if args.export:
        export()
    if args.absorb:
        absorb(args.since)
    if not (args.export or args.absorb):
        print('--export 또는 --absorb 를 지정하라. 도움말: -h')


if __name__ == '__main__':
    main()
