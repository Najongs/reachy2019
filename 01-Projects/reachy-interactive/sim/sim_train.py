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

CONFIG = os.path.join(HERE, '..', 'config')
GOOD_SCORE = 90          # 이 점수를 넘고 지적이 없으면 그만 고친다


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


def improve(task, url, token, session, fix_rounds, keep_images=None):
    """한 과제를 설계하고, 점수가 낮으면 지적해서 다시 받는다."""
    say = task['say']
    prompt = say if not task.get('scene') else \
        '%s\n(지금 로봇 앞 상황: %s)' % (say, task['scene'])

    history = []
    best = None

    motion = ask_motion(url, token, prompt, session)
    if motion is None:
        return {'say': say, 'error': '설계를 받지 못했습니다', 'history': []}

    for attempt in range(fix_rounds + 1):
        moves = motion.get('moves')
        if not moves:
            # 프리셋으로 답한 경우 - 손으로 다듬은 것이라 건드리지 않는다.
            return {'say': say, 'preset': motion.get('preset'),
                    'note': '프리셋으로 답했습니다 (고칠 대상 아님)',
                    'history': history}

        result = sim_eval.evaluate(moves)
        history.append({'attempt': attempt, 'score': result['score'],
                        'findings': result['findings'],
                        'error': result.get('error')})
        print('    %d회차: %d점  %s' % (
            attempt, result['score'],
            result['findings'][0][:58] if result['findings'] else '지적 없음'))

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

    return {'say': say, 'scene': task.get('scene'), 'who': task.get('who'),
            'best_score': best['score'] if best else 0,
            'first_score': history[0]['score'] if history else 0,
            'say_line': best['say'] if best else None,
            'moves': best['moves'] if best else None,
            'findings': best['result']['findings'] if best else [],
            'history': history}


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
    ap.add_argument('--out', default=os.path.join(CONFIG, 'sim_results.json'))
    args = ap.parse_args()

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

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        print('[%d/%d] "%s"' % (i, len(tasks), task['say'][:56]))
        try:
            r = improve(task, args.url, args.token, args.session,
                        args.fix_rounds, args.keep_images)
        except KeyboardInterrupt:
            print('\n중단합니다.')
            break
        except Exception as e:
            print('    과제 처리 중 오류: %s: %s' % (type(e).__name__, e))
            r = {'say': task['say'], 'error': str(e), 'history': []}
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
