"""동작을 스스로 만들고, 시뮬에서 돌려 보고, 평가받아 고치는 것을 되풀이한다.

    말 만들기(ollama, 무료)  ->  동작 설계(opus)  ->  시뮬 실행 + 채점(로컬)
                                      ^                        |
                                      +--- 지적 + 궤적 그림 ----+

강화학습은 아니다. 보상으로 정책을 갱신하는 게 아니라, 채점 결과를 말로 돌려
주고 설계자(opus)가 다시 쓰게 한다. 이 로봇에서는 그쪽이 맞다 - 관절이 15개고
동작은 키프레임 열두 개 이하라, 수백만 번 굴릴 대상이 아니라 '몇 번 고쳐 쓰면
되는' 대상이다. 그리고 채점기가 실물과 같은 검증기를 쓰므로, 여기서 통과한
동작은 실물에서도 통과한다.

GPU 는 ollama 가 이미 올라가 있는 한 장만 쓴다. 채점은 순기구학이라 CPU 로
돌고, opus 는 이 기계의 GPU 를 쓰지 않는다.

비용 주의: opus 호출은 사용자의 구독을 쓴다. 한 과제당 최대 (1 + --fix-rounds)
번이다. 기본값은 5과제 x 3번 = 15번.

사용:
    python3 sim/sim_train.py --rounds 5 --token reachy2019
    python3 sim/sim_train.py --tasks config/sim_commands.json --rounds 10
    python3 sim/sim_train.py --rounds 3 --keep-images out/
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_eval                                  # noqa: E402
import sim_opt                                   # noqa: E402

CONFIG = os.path.join(HERE, '..', 'config')
GOOD_SCORE = 90          # 이 점수를 넘고 지적이 없으면 그만 고친다
LESSONS = os.path.join(CONFIG, 'sim_lessons.json')
LESSON_MIN = 2           # 이만큼 되풀이된 지적만 교훈으로 올린다
LESSON_SHOW = 6          # 설계 프롬프트에 얹을 교훈 개수

# 지적을 종류별로 묶는다. 숫자가 달라도 같은 실수는 같은 교훈이다.
LESSON_MAP = [
    ('복도 건너에서는',
     '동작이 너무 작다 - 어깨를 크게 써서 손이 최소 15cm 는 움직이게 하라'),
    ('속도 상한',
     'duration 을 너무 짧게 잡는다 - 큰 각도 변화에는 넉넉한 시간을 처음부터 줘라'),
    ('금지 구역에',
     '몸통·머리에 너무 붙인다 - 여유를 5cm 이상 둬라'),
    ('한 번도 움직이지 않는',
     '쓰지도 않을 관절을 키프레임에 넣는다 - 움직이는 관절만 써라'),
    ('마지막 자세가 휴식에서',
     '마지막 키프레임을 휴식에 가깝게 두어라 - 복귀가 크면 뚝 끊긴 느낌이 난다'),
    ('지나가는 사람에게 깁니다',
     '동작이 길다 - 12초를 넘기지 마라'),
    ('한계를 넘은 각도',
     '관절 한계를 넘겨 적는다 - 프롬프트의 범위 안에서만 써라'),
]


def _lesson_of(finding):
    for key, label in LESSON_MAP:
        if key in finding:
            return key, label
    return None, None


def load_lessons():
    try:
        with open(LESSONS, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_lessons(counts):
    try:
        with open(LESSONS, 'w', encoding='utf-8') as fh:
            json.dump(counts, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print('  교훈을 저장하지 못했습니다: %s' % e)


def note_lessons(counts, findings):
    """이번 과제에서 나온 지적을 교훈 장부에 더한다."""
    for f in findings or []:
        key, label = _lesson_of(f)
        if key is None:
            continue
        row = counts.setdefault(key, {'n': 0, 'label': label})
        row['n'] += 1
        row['label'] = label
    return counts


def lesson_block(counts):
    """설계 프롬프트 앞에 붙일 교훈. 되풀이된 순서대로.

    이 루프에서 '학습' 에 해당하는 부분이다. 가중치를 갱신하는 대신, 되풀이된
    지적을 말로 정리해 다음 설계의 앞에 놓는다. 같은 실수를 두 번 지적받으면
    세 번째부터는 처음부터 피하게 된다.
    """
    ranked = sorted(((v['n'], v['label']) for v in counts.values()
                     if v['n'] >= LESSON_MIN), reverse=True)
    if not ranked:
        return ''
    lines = ['지난 설계들에서 시뮬레이터가 되풀이해 지적한 것들이다. '
             '이번에는 처음부터 피해라:']
    for n, label in ranked[:LESSON_SHOW]:
        lines.append('  - %s (%d번 지적됨)' % (label, n))
    return '\n'.join(lines) + '\n\n'


def ask_motion(url, token, text, session, image=None, timeout=180):
    """브로커 /motion 에 설계를 시킨다. {'say','preset','moves'} 또는 None."""
    body = {'text': text, 'session': session}
    if image:
        body['image'] = image
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url.rstrip('/') + '/motion', data=data,
                                 method='POST')
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('X-Auth-Token', token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print('    설계 실패 HTTP %s: %s' % (e.code, e.read()[:160]))
    except Exception as e:
        print('    설계 실패: %s' % e)
    return None


CRITIQUE = """방금 네가 설계한 동작을 시뮬레이터에서 실제로 돌려 봤다.
시뮬레이터는 실물과 **같은 검증기와 순기구학**을 쓴다 - 여기서 나온 지적은
실물에서도 그대로 일어난다.

{report}

첨부한 그림은 손이 그린 궤적을 위/옆/앞에서 본 것과, 관절 각도의 시간 변화다.
회색 영역이 부딪히면 안 되는 곳이다.

위 지적을 고쳐서 같은 요청("{say}")에 대한 동작을 **다시** 설계해라.
- 지적된 것만 고치고 잘 되던 부분은 유지해라
- 평소와 같은 JSON 형식으로만 답해라"""


def improve(task, url, token, session, fix_rounds, keep_images=None,
            lessons=None, opt_iters=10, opt_pop=40, workers=8):
    """한 과제를 설계하고, 탐색으로 다듬고, 남은 지적은 다시 설계시킨다.

    역할이 셋이다:
      opus   무엇을 할지 (관절 선택, 키프레임 구성, 이야기)
      CEM    각도와 시간의 미세 조정 (opus 호출 0회, 1분에 수천 번)
      교훈   되풀이된 지적을 다음 설계 앞에 놓는다
    """
    say = task['say']
    prompt = say if not task.get('scene') else \
        '%s\n(지금 로봇 앞 상황: %s)' % (say, task['scene'])
    # 학습에서는 기존 프리셋을 재사용하면 배울 것이 없다. 실제 로봇에서는
    # 프리셋으로 답하는 편이 빠르고 안전하므로, 이 지시는 여기서만 붙인다.
    prompt = ('이번에는 기존 프리셋을 쓰지 말고 moves 키프레임을 직접 '
              '설계해라.\n\n') + prompt
    if lessons:
        block = lesson_block(lessons)
        if block:
            prompt = block + prompt

    history = []
    best = None

    motion = ask_motion(url, token, prompt, session)
    if motion is None:
        return {'say': say, 'error': '설계를 받지 못했습니다', 'history': []}

    for attempt in range(fix_rounds + 1):
        moves = motion.get('moves')
        if not moves:
            # 그래도 프리셋으로 답했다면 배울 것이 없다. 조용히 넘기지 말고
            # 남긴다 - 예전에는 아무것도 출력되지 않아 실패처럼 보였다.
            print('    프리셋 %r 로 답했습니다 - 새 설계가 아니라 건너뜁니다'
                  % motion.get('preset'))
            return {'say': say, 'preset': motion.get('preset'),
                    'note': '프리셋으로 답했습니다 (고칠 대상 아님)',
                    'history': history}

        raw = sim_eval.evaluate(moves)
        polished, info = (moves, {})
        if raw['ok']:
            # 구조는 그대로 두고 숫자만 다듬는다. opus 호출은 늘지 않는다.
            polished, info = sim_opt.optimize(
                moves, iters=opt_iters, pop=opt_pop, workers=workers,
                seed=attempt)
        result = sim_eval.evaluate(polished)
        moves = polished

        history.append({'attempt': attempt,
                        'score_raw': raw['score'],
                        'score': result['score'],
                        'opt_curve': info.get('curve'),
                        'findings': result['findings'],
                        'error': result.get('error')})
        gain = ('  (설계 %d -> 탐색 %d)' % (raw['score'], result['score'])
                if result['score'] != raw['score'] else '')
        print('    %d회차: %d점%s  %s' % (
            attempt, result['score'], gain,
            result['findings'][0][:48] if result['findings'] else '지적 없음'))

        if best is None or result['score'] > best['score']:
            best = {'score': result['score'], 'moves': moves,
                    'say': motion.get('say'), 'result': result}

        done = result['ok'] and result['score'] >= GOOD_SCORE and \
            not result['findings']
        if done or attempt >= fix_rounds:
            break

        # 지적 + 그림을 들려 보내 다시 설계하게 한다.
        image = sim_eval.render_b64(moves, title=say[:40]) if result['ok'] else None
        if keep_images and result['ok']:
            sim_eval.render(moves, os.path.join(
                keep_images, 'try-%s-%d.png' % (abs(hash(say)) % 10000, attempt)),
                title='%s (try %d, %d pts)' % (say[:30], attempt, result['score']))
        motion = ask_motion(url, token,
                            CRITIQUE.format(report=sim_eval.describe(result),
                                            say=say),
                            session, image=image)
        if motion is None:
            break

    if lessons is not None and best:
        note_lessons(lessons, best['result']['findings'])

    return {'say': say, 'scene': task.get('scene'), 'who': task.get('who'),
            'best_score': best['score'] if best else 0,
            'first_score': history[0]['score'] if history else 0,
            'say_line': best['say'] if best else None,
            'moves': best['moves'] if best else None,
            'findings': best['result']['findings'] if best else [],
            'history': history}


def report(path):
    """지난 학습 기록을 되짚는다. 누가 얼마나 기여했나.

    한 번 돌리고 끝이 아니라, 무엇이 실제로 점수를 올렸는지 봐야 다음에 뭘
    고칠지 알 수 있다. opus 를 더 부를지, 탐색을 더 돌릴지, 교훈을 손볼지.
    """
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    rows = [r for r in data['results'] if r.get('history')]
    if not rows:
        print('점수가 매겨진 과제가 없습니다.')
        return

    print('%s 기록, 과제 %d개' % (data.get('ran_at', '?'), len(data['results'])))
    print()

    # 기여도: 첫 설계 -> 탐색 -> 다시 설계, 각 단계가 얼마나 올렸나
    draft, after_opt, final = [], [], []
    opt_gains, redesign_gains = [], []
    for r in rows:
        h = r['history']
        first = h[0]
        draft.append(first.get('score_raw', first['score']))
        after_opt.append(first['score'])
        final.append(max(a['score'] for a in h))
        opt_gains.append(first['score'] - first.get('score_raw', first['score']))
        redesign_gains.append(max(a['score'] for a in h) - first['score'])

    n = len(rows)
    print('  단계별 평균 점수')
    print('    opus 첫 설계        %5.1f' % (sum(draft) / n))
    print('    + CEM 탐색          %5.1f   (+%.1f, opus 호출 0회)'
          % (sum(after_opt) / n, sum(opt_gains) / n))
    print('    + opus 재설계       %5.1f   (+%.1f)'
          % (sum(final) / n, sum(redesign_gains) / n))
    print()

    helped = sum(1 for g in opt_gains if g > 0)
    redone = sum(1 for g in redesign_gains if g > 0)
    print('  탐색이 도움된 과제 %d/%d, 재설계가 도움된 과제 %d/%d'
          % (helped, n, redone, n))

    # 남은 지적을 종류별로 - 다음에 무엇을 고쳐야 하나
    from collections import Counter
    left = Counter()
    for r in rows:
        for f in r.get('findings') or []:
            key, label = _lesson_of(f)
            left[label or f[:40]] += 1
    if left:
        print()
        print('  끝까지 남은 지적 (다음 학습이 노릴 것)')
        for label, c in left.most_common(6):
            print('    %2d회  %s' % (c, label[:66]))

    scored = [r for r in rows if r.get('best_score')]
    keepers = [r for r in scored
               if r['best_score'] >= GOOD_SCORE and not r.get('findings')]
    print()
    print('  후보로 남길 만한 것 %d개 (90점 이상, 지적 없음)' % len(keepers))
    for r in sorted(keepers, key=lambda x: -x['best_score'])[:8]:
        print('    %3d점  %s' % (r['best_score'], r['say'][:52]))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--token', default=os.environ.get('BROKER_TOKEN'))
    ap.add_argument('--rounds', type=int, default=5,
                    help='몇 개 과제를 돌릴지 (opus 호출 = 과제수 x (1+fix))')
    ap.add_argument('--fix-rounds', type=int, default=2,
                    help='한 과제를 몇 번까지 다시 설계시킬지')
    ap.add_argument('--tasks', help='미리 만들어 둔 과제 json (없으면 즉석 생성)')
    ap.add_argument('--session', default='sim-train',
                    help='opus 대화 세션. 같은 세션이면 앞 과제의 지적을 기억한다')
    ap.add_argument('--keep-images', help='궤적 그림을 이 폴더에 남긴다')
    ap.add_argument('--opt-iters', type=int, default=10,
                    help='초안 하나를 탐색으로 몇 세대 다듬을지 (0=끔)')
    ap.add_argument('--opt-pop', type=int, default=40,
                    help='탐색 한 세대의 후보 수')
    ap.add_argument('--workers', type=int, default=8,
                    help='탐색에 쓸 CPU 프로세스 수')
    ap.add_argument('--no-lessons', action='store_true',
                    help='되풀이된 지적을 다음 설계에 알려 주지 않는다')
    ap.add_argument('--report', metavar='FILE', nargs='?',
                    const=os.path.join(CONFIG, 'sim_results.json'),
                    help='지난 기록을 되짚어 보기만 한다 (새로 돌리지 않음)')
    ap.add_argument('--out', default=os.path.join(CONFIG, 'sim_results.json'))
    args = ap.parse_args()

    if args.report:
        report(args.report)
        return

    if args.keep_images:
        os.makedirs(args.keep_images, exist_ok=True)

    if args.tasks:
        with open(args.tasks, encoding='utf-8') as fh:
            tasks = json.load(fh)['tasks']
    else:
        import sim_scenarios
        print('과제를 만드는 중 (로컬 모델, 무료)...')
        tasks = sim_scenarios.build(n_personas=3, per_persona=4,
                                    n_situations=3)['tasks']
    tasks = tasks[:args.rounds]
    print('과제 %d개, 과제당 최대 %d번 설계 (opus 호출 최대 %d회)\n'
          % (len(tasks), args.fix_rounds + 1,
             len(tasks) * (args.fix_rounds + 1)))

    lessons = None if args.no_lessons else load_lessons()
    if lessons:
        known = sum(1 for v in lessons.values() if v['n'] >= LESSON_MIN)
        print('지난 학습에서 얻은 교훈 %d개를 설계에 얹습니다\n' % known)

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        print('[%d/%d] "%s"' % (i, len(tasks), task['say'][:56]))
        try:
            r = improve(task, args.url, args.token, args.session,
                        args.fix_rounds, args.keep_images, lessons=lessons,
                        opt_iters=args.opt_iters, opt_pop=args.opt_pop,
                        workers=args.workers)
        except KeyboardInterrupt:
            print('\n중단합니다.')
            break
        except Exception as e:
            print('    과제 처리 중 오류: %s: %s' % (type(e).__name__, e))
            r = {'say': task['say'], 'error': str(e), 'history': []}
        if r.get('error'):
            print('    %s' % r['error'])
        results.append(r)
        print()

    # --- 요약 -------------------------------------------------------------
    print('═' * 62)
    scored = [r for r in results if r.get('best_score')]
    improved = [r for r in scored if r['best_score'] > r['first_score']]
    print('%d개 과제, %.0f분 소요' % (len(results), (time.time() - t0) / 60.0))
    if scored:
        print('평균 첫 점수 %.0f -> 최종 %.0f  (%d개가 나아졌다)'
              % (sum(r['first_score'] for r in scored) / len(scored),
                 sum(r['best_score'] for r in scored) / len(scored),
                 len(improved)))
        print()
        for r in sorted(scored, key=lambda x: -x['best_score']):
            arrow = '%d→%d' % (r['first_score'], r['best_score'])
            print('  %-9s %-46s %s' % (arrow, r['say'][:46],
                                       '✓' if not r['findings'] else
                                       r['findings'][0][:28]))
    failed = [r for r in results if r.get('error')]
    if failed:
        print('\n설계를 못 받은 과제 %d개' % len(failed))

    if lessons is not None:
        save_lessons(lessons)
        top = sorted(((v['n'], v['label']) for v in lessons.values()),
                     reverse=True)[:3]
        if top:
            print('\n되풀이된 지적 (다음 학습의 설계 프롬프트에 얹힙니다):')
            for n, label in top:
                print('  %2d번  %s' % (n, label[:64]))

    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump({'ran_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'results': results}, fh, ensure_ascii=False, indent=2)
    print('\n전체 기록: %s' % args.out)

    # 90점 넘고 지적 없는 것만 후보로 남긴다.
    keepers = [r for r in scored
               if r['best_score'] >= GOOD_SCORE and not r['findings']]
    if keepers:
        path = os.path.join(CONFIG, 'motion_candidates.json')
        try:
            with open(path, encoding='utf-8') as fh:
                cand = json.load(fh)
        except Exception:
            cand = []
        have = {json.dumps(c.get('moves'), sort_keys=True) for c in cand}
        added = 0
        for r in keepers:
            key = json.dumps(r['moves'], sort_keys=True)
            if key in have:
                continue
            cand.append({'idea': r['say'], 'say': r['say_line'],
                         'moves': r['moves'], 'sim_score': r['best_score']})
            have.add(key)
            added += 1
        if added:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(cand, fh, ensure_ascii=False, indent=2)
            print('%d개를 motion_candidates.json 에 넣었습니다 (90점 이상, 지적 없음)'
                  % added)


if __name__ == '__main__':
    main()
