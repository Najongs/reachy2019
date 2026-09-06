"""행동을 반복 개선하는 루프 - 성공은 문지기, 품질은 목표.

    평가(결정론 지표 + opus 비평) -> 파라미터 후보 생성 -> 같은 태스크로
    재평가 -> 객관 성공이 안 깨진 후보 중 품질 최고를 채택 -> 반복

강화학습의 실용판이다: 보상(품질 점수 + 자연스러움 비평)을 보고 정책
(BEHAVIOR 손잡이)을 갱신한다. 정확히 지키는 것 하나 - **객관 성공/안전은
언어모델이 아니라 계측이 판정하고, 그걸 깨는 후보는 품질이 아무리 좋아도
버린다.** 언어모델은 '어떻게 움직였는가' 만 비평한다.

후보 = 현재 + opus 조언 반영 + 무작위 흔들기. 같은 태스크 묶음으로 비교해야
운이 아니라 파라미터 차이를 잰다.

사용:
    python3 sim/sim_improve.py --iters 3 --token reachy2019
    python3 sim/sim_improve.py --iters 2                # opus 비평 없이 지표만
"""

import copy
import json
import os
import random
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_agent as A                              # noqa: E402
import sim_critic as C                             # noqa: E402
import sim_tasks as T                              # noqa: E402

# 파라미터 허용 범위 - 이 밖은 물리적으로 무의미하거나 위험.
BOUNDS = {'step_deg': (1.5, 8.0), 'gain': (0.3, 1.4), 'damping': (1.0, 10.0),
          'table_min': (0.2, 3.0), 'lift_steps': (3, 14),
          # 경로 모양 - 우회를 줄일 수 있는 손잡이. 투과는 문지기가 거른다.
          'ready_pitch': (-40.0, 15.0), 'ready_roll': (-110.0, -25.0),
          'ready_yaw': (0.0, 60.0), 'ready_elbow': (-125.0, -60.0)}
INT_PARAMS = ('lift_steps',)


def _clamp(params):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = max(lo, min(hi, params.get(k, A.BEHAVIOR[k])))
        out[k] = int(round(v)) if k in INT_PARAMS else round(float(v), 2)
    return out


def _jitter(current, rng, scale=1.0):
    """현재 주변의 무작위 후보. 각도형 파라미터는 덧셈, 배율형은 곱셈."""
    j = dict(current)
    for k in BOUNDS:
        if rng.random() < 0.5:
            continue
        if k.startswith('ready_'):
            span = (BOUNDS[k][1] - BOUNDS[k][0]) * 0.12 * scale
            j[k] = current[k] + rng.uniform(-span, span)
        else:
            j[k] = current[k] * rng.uniform(1 - 0.25 * scale, 1 + 0.3 * scale)
    return j


def episode(task, keep_frames=False, natural_min=55):
    """한 에피소드 -> 3단계 판정. (stage, 품질 dict, frames).

    stage: collision / unnatural / missed / success (sim_critic.stage_verdict).
    충돌이면 그걸로 끝, 자연스러움 문턱 미달이면 도달했어도 성공이 아니다.
    """
    world = T.build_world(task)
    frames = [] if keep_frames else None
    trace = []
    try:
        planner = A.OraclePlanner()
        view = planner.perceive(world, task)
        plan = planner.plan(world, task, view)
        A.execute(world, plan, frames=frames, trace=trace)
        reached = bool(T.success_fn(task)(world))
        path_min = min(min((t['table'] for t in trace), default=99),
                       min((t['obj'] for t in trace), default=99))
        q = C.quality(trace) if trace else {'score': 0}
        stage, why = C.stage_verdict(path_min, reached, q.get('score', 0),
                                     natural_min)
        q['stage'], q['why'] = stage, why
        return stage, q, frames
    finally:
        world.close()


def evaluate(params, tasks, natural_min=55):
    """태스크 묶음 -> {'collision','unnatural','missed','success', 'quality'}."""
    A.BEHAVIOR.update(_clamp(params))
    counts = {k: 0 for k in C.STAGES}
    scores = []
    for t in tasks:
        stage, q, _ = episode(t, natural_min=natural_min)
        counts[stage] += 1
        scores.append(q.get('score', 0))
    counts['quality'] = sum(scores) / max(len(scores), 1)
    return counts


def propose(current, advice, rng, n_random=3, scale=1.0):
    """후보들: 현재 / 조언 반영 / 무작위 흔들기(scale 로 폭 조절)."""
    cands = [('current', dict(current))]
    if advice:
        adv = dict(current)
        for k, direction in advice.items():
            if k not in BOUNDS:
                continue
            if k.startswith('ready_'):
                span = (BOUNDS[k][1] - BOUNDS[k][0]) * 0.15
                adv[k] = current[k] + (span if direction == 'up' else -span)
            else:
                adv[k] = current[k] * (1.25 if direction == 'up' else 0.8)
        cands.append(('advice', adv))
    for i in range(n_random):
        cands.append(('jitter%d' % i, _jitter(current, rng, scale)))
    return [(name, _clamp(c)) for name, c in cands]


REFRESH_EVERY = 6       # 이 횟수마다 태스크 묶음을 새로 뽑는다 (과적합 방지)
VALIDATE_EVERY = 60     # end-to-end 검증 주기
CRITIQUE_EVERY = 3      # opus 비평 주기 (10시간 기준 시간당 ~20콜 수준)


def run(iters=3, n_tasks=4, seed=0, token=None, url='http://127.0.0.1:8080',
        hours=None):
    """개선 루프. hours 를 주면 그 시간 동안 계속 돈다 (iters 무시)."""
    rng = random.Random(seed)
    deadline = time.time() + hours * 3600 if hours else None
    label = '%.1f시간' % hours if hours else '%d회' % iters
    print('개선 루프: %s, 태스크 %d개 (%d회마다 교체)'
          % (label, n_tasks, REFRESH_EVERY), flush=True)

    client = None
    if token:
        from llm_client import BrokerClient
        client = BrokerClient(url, token=token, session='sim-critic')

    def new_tasks(k):
        ts = T.generate_feasible(n_tasks, seed=seed * 977 + k, kinds=('reach',))
        return ts

    tasks = new_tasks(0)
    history = []
    current = _clamp(dict(A.BEHAVIOR))
    # 자연스러움 문턱 - 전원 성공이 이어지면 올라간다(평가도 함께 개선).
    natural_min = 55

    def rank(counts):
        """단계 우선순위 그대로: 충돌 적게 > 성공 많이 > 품질 높게.

        '성공' 은 이미 (충돌 없음 AND 자연스러움 문턱 이상 AND 도달) 이므로
        자연스러움은 성공 안에 들어 있다 - 거칠게 닿은 동작은 성공이 아니다.
        """
        return (-counts['collision'], counts['success'], counts['quality'])

    base = evaluate(current, tasks, natural_min)
    best_ever = {'quality': base['quality'], 'success': base['success'],
                 'params': dict(current), 'natural_min': natural_min}
    stagnant = 0
    perfect_streak = 0
    out = os.path.join(HERE, '..', 'sim_data',
                       'improve-%s.json' % time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def checkpoint():
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump({'history': history, 'best_ever': best_ever},
                      fh, ensure_ascii=False, indent=1)

    def fmt(c):
        return '충돌%d 부자연%d 미도달%d 성공%d 품질%.0f' % (
            c['collision'], c['unnatural'], c['missed'], c['success'],
            c['quality'])

    print('반복 0 (기준, 문턱 %d): %s' % (natural_min, fmt(base)), flush=True)
    history.append({'iter': 0, 'counts': {k: base[k] for k in C.STAGES},
                    'quality': round(base['quality'], 1),
                    'natural_min': natural_min,
                    'params': dict(current), 'ts': time.strftime('%H:%M')})

    it = 0
    while True:
        it += 1
        if deadline is not None:
            if time.time() >= deadline:
                break
        elif it > iters:
            break

        try:
            # 태스크 교체: 같은 4개에 과적합되지 않게. 교체하면 기준 재측정.
            if it % REFRESH_EVERY == 0:
                tasks = new_tasks(it)
                base = evaluate(current, tasks, natural_min)
                print('[%s] 반복 %d: 태스크 교체 -> %s'
                      % (time.strftime('%H:%M'), it, fmt(base)), flush=True)

            # opus 비평 (주기적으로; 실패해도 루프는 계속)
            advice, naturalness = None, None
            if client is not None and it % CRITIQUE_EVERY == 1:
                try:
                    A.BEHAVIOR.update(current)
                    ok, q, frames = episode(tasks[0], keep_frames=True)
                    crit = C.critique(client, C.pick_frames(frames, 3),
                                      tasks[0]['instruction'], q)
                    if crit:
                        naturalness = crit.get('naturalness')
                        advice = crit.get('advice') or None
                except Exception as e:
                    print('  비평 실패(계속 진행): %s' % e, flush=True)

            # 정체하면 탐색 폭을 넓힌다 (담금질). 개선되면 되돌린다.
            scale = min(3.0, 1.0 + stagnant * 0.35)
            n_random = 3 + min(3, stagnant // 2)

            best = (rank(base), base, 'current', dict(current))
            for name, cand in propose(current, advice, rng, n_random, scale):
                c = evaluate(cand, tasks, natural_min)
                # 문지기: 충돌은 기준보다 늘 수 없다 (안전 후퇴 금지).
                if c['collision'] > base['collision']:
                    continue
                if rank(c) > best[0]:
                    best = (rank(c), c, name, cand)

            improved = best[0] > rank(base)
            stagnant = 0 if improved else stagnant + 1
            base = best[1]
            current = best[3]
            A.BEHAVIOR.update(current)
            A.save_behavior(current)
            if (base['collision'] == 0 and base['success'] >= len(tasks)
                    and base['quality'] > best_ever['quality']):
                best_ever = {'quality': base['quality'],
                             'success': base['success'],
                             'params': dict(current), 'iter': it,
                             'natural_min': natural_min}

            # 평가 래칫: 전원 성공이 2회 이어지면 자연스러움 문턱을 올린다.
            # 행동이 좋아질수록 평가도 엄해진다 - '점점 평가 개선'.
            if base['collision'] == 0 and base['success'] >= len(tasks):
                perfect_streak += 1
                if perfect_streak >= 2 and natural_min < 85:
                    natural_min += 5
                    perfect_streak = 0
                    base = evaluate(current, tasks, natural_min)
                    print('[%s] 평가 강화: 자연스러움 문턱 -> %d (재측정 %s)'
                          % (time.strftime('%H:%M'), natural_min, fmt(base)),
                          flush=True)
            else:
                perfect_streak = 0

            history.append({'iter': it,
                            'counts': {k: base[k] for k in C.STAGES},
                            'quality': round(base['quality'], 1),
                            'natural_min': natural_min,
                            'naturalness': naturalness, 'picked': best[2],
                            'stagnant': stagnant,
                            'params': dict(current),
                            'ts': time.strftime('%H:%M')})
            nat = ' 자연 %s/10' % naturalness if naturalness else ''
            print('[%s] 반복 %d(문턱 %d): %s 채택, %s%s%s'
                  % (time.strftime('%H:%M'), it, natural_min, best[2],
                     fmt(base), nat,
                     ' (정체 %d)' % stagnant if stagnant else ''), flush=True)
            checkpoint()

            # 주기적 end-to-end 검증: opus 접지 포함 실전 배치가 여전히 되나.
            if client is not None and it % VALIDATE_EVERY == 0:
                try:
                    import sim_pipeline as PL
                    rid, summ, flags = PL.run_batch(
                        4, ('look', 'reach'), 'broker', url, token,
                        seed=it, tag='improve-check%d' % it)
                    print('[%s] end-to-end 검증: %d/%d, 감사 %s'
                          % (time.strftime('%H:%M'), summ['success'],
                             summ['n'], '깨끗' if not flags else flags[0][:40]),
                          flush=True)
                except Exception as e:
                    print('  검증 실패(계속 진행): %s' % e, flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # 장기 실행: 어떤 예외도 루프를 죽이지 않는다. 기록하고 계속.
            print('[%s] 반복 %d 예외(계속): %s: %s'
                  % (time.strftime('%H:%M'), it, type(e).__name__, e),
                  flush=True)
            time.sleep(5)

    checkpoint()
    qs = [h['quality'] for h in history]
    print('품질 추이: %.0f -> %.0f (최고 %.0f, 최종 문턱 %d, 반복 %d회)'
          % (qs[0], qs[-1], best_ever['quality'], natural_min,
             len(history) - 1))
    print('기록: %s  |  채택 파라미터: config/behavior_params.json' % out)
    return history


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--hours', type=float, help='시간 예산 (주면 iters 무시)')
    ap.add_argument('--tasks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--token')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    args = ap.parse_args()
    run(args.iters, args.tasks, args.seed, args.token, args.url,
        hours=args.hours)


if __name__ == '__main__':
    main()
