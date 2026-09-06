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
          'table_min': (0.2, 3.0), 'lift_steps': (3, 14)}


def _clamp(params):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = max(lo, min(hi, params.get(k, A.BEHAVIOR[k])))
        out[k] = int(round(v)) if k == 'lift_steps' else round(float(v), 2)
    return out


def episode(task, keep_frames=False):
    """오라클 계획으로 한 에피소드. (성공, 품질 dict, frames)."""
    world = T.build_world(task)
    frames = [] if keep_frames else None
    trace = []
    try:
        planner = A.OraclePlanner()
        view = planner.perceive(world, task)
        plan = planner.plan(world, task, view)
        A.execute(world, plan, frames=frames, trace=trace)
        ok = bool(T.success_fn(task)(world))
        path_tab = min((t['table'] for t in trace), default=99)
        path_obj = min((t['obj'] for t in trace), default=99)
        if path_tab < -0.5 or path_obj < -0.5:
            ok = False                          # 경로 투과는 무조건 실패
        q = C.quality(trace) if trace else {'score': 0}
        return ok, q, frames
    finally:
        world.close()


def evaluate(params, tasks):
    """이 파라미터로 태스크 묶음을 돌린 (성공수, 평균품질)."""
    A.BEHAVIOR.update(_clamp(params))
    oks, scores = 0, []
    for t in tasks:
        ok, q, _ = episode(t)
        oks += ok
        scores.append(q.get('score', 0))
    return oks, sum(scores) / max(len(scores), 1)


def propose(current, advice, rng, n_random=3):
    """후보들: 현재 / 조언 반영 / 무작위 흔들기."""
    cands = [('current', dict(current))]
    if advice:
        adv = dict(current)
        for k, direction in advice.items():
            if k not in BOUNDS:
                continue
            factor = 1.25 if direction == 'up' else 0.8
            adv[k] = current[k] * factor
        cands.append(('advice', adv))
    for i in range(n_random):
        j = dict(current)
        for k in BOUNDS:
            if rng.random() < 0.5:
                j[k] = current[k] * rng.uniform(0.75, 1.3)
        cands.append(('jitter%d' % i, j))
    return [(name, _clamp(c)) for name, c in cands]


def run(iters=3, n_tasks=4, seed=0, token=None, url='http://127.0.0.1:8080'):
    rng = random.Random(seed)
    print('개선 루프: 반복 %d회, 태스크 %d개 고정(비교 가능하게)' % (iters, n_tasks))
    tasks = T.generate_feasible(n_tasks, seed=seed, kinds=('reach',))
    for t in tasks:
        print('  -', t['instruction'][:40])

    client = None
    if token:
        from llm_client import BrokerClient
        client = BrokerClient(url, token=token, session='sim-critic')

    history = []
    current = _clamp(dict(A.BEHAVIOR))
    base_ok, base_q = evaluate(current, tasks)
    print('\n반복 0 (기준): 성공 %d/%d, 품질 %.0f  %s'
          % (base_ok, len(tasks), base_q, current))
    history.append({'iter': 0, 'ok': base_ok, 'quality': round(base_q, 1),
                    'params': dict(current)})

    for it in range(1, iters + 1):
        # 1) 대표 에피소드의 프레임으로 opus 비평 (반복당 1회)
        advice, naturalness = None, None
        if client is not None:
            A.BEHAVIOR.update(current)
            ok, q, frames = episode(tasks[0], keep_frames=True)
            crit = C.critique(client, C.pick_frames(frames, 3),
                              tasks[0]['instruction'], q)
            if crit:
                naturalness = crit.get('naturalness')
                advice = crit.get('advice') or None
                print('반복 %d 비평: 자연스러움 %s/10  지적 %s  조언 %s'
                      % (it, naturalness,
                         (crit.get('issues') or ['-'])[0][:44], advice))

        # 2) 후보 생성 -> 같은 태스크로 평가
        best = (base_ok, base_q, 'current', dict(current))
        for name, cand in propose(current, advice, rng):
            ok, qual = evaluate(cand, tasks)
            gate = ok >= base_ok            # 객관 성공은 후퇴 금지 (문지기)
            mark = '통과' if gate else '탈락(성공 후퇴)'
            print('   %-8s 성공 %d/%d 품질 %.0f  %s'
                  % (name, ok, len(tasks), qual, mark))
            if gate and (qual > best[1] or (qual == best[1] and ok > best[0])):
                best = (ok, qual, name, cand)

        base_ok, base_q = best[0], best[1]
        current = best[3]
        A.BEHAVIOR.update(current)
        A.save_behavior(current)
        history.append({'iter': it, 'ok': base_ok, 'quality': round(base_q, 1),
                        'naturalness': naturalness, 'picked': best[2],
                        'params': dict(current)})
        print('반복 %d 채택: %s -> 성공 %d/%d 품질 %.0f\n'
              % (it, best[2], base_ok, len(tasks), base_q))

    out = os.path.join(HERE, '..', 'sim_data',
                       'improve-%s.json' % time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'tasks': [t['id'] for t in tasks], 'history': history},
                  fh, ensure_ascii=False, indent=1)
    print('품질 추이: %s' % ' -> '.join('%.0f' % h['quality'] for h in history))
    print('기록: %s  |  채택 파라미터: config/behavior_params.json' % out)
    return history


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--tasks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--token')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    args = ap.parse_args()
    run(args.iters, args.tasks, args.seed, args.token, args.url)


if __name__ == '__main__':
    main()
