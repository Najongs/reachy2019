"""opus 가 만든 초안의 '숫자'를 탐색으로 다듬는다.

역할을 나눈다. 큰 모델은 **무엇을 할지**를 잘 정한다 - 어느 관절을 어떤 순서로
쓸지, 몇 개의 키프레임으로 이야기를 만들지. 반면 "어깨를 3도 더 들면 점수가
오른다" 같은 것은 언어로 풀 일이 아니다. 그건 그냥 찾으면 된다.

그래서 구조는 opus 가 잡고, 각도와 시간은 여기서 교차엔트로피법(CEM)으로
찾는다. 채점기가 한 번에 28ms 라 1분이면 수천 번 굴릴 수 있고, opus 호출은
한 번도 늘지 않는다.

**의미를 지키는 것이 중요하다.** 점수만 올리면 손 흔들기가 큰 팔 휘두르기로
변한다. 그래서 원래 각도에서 MAX_DRIFT 도 이상 벗어나지 못하게 묶는다 -
고쳐 쓰는 게 아니라 다듬는 것이다.

사용:
    python3 sim/sim_opt.py --preset wave
    python3 sim/sim_opt.py --json moves.json --iters 15
"""

import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                          # noqa: E402
import sim_eval                                   # noqa: E402


MAX_DRIFT = 25.0        # 원래 각도에서 이만큼(도) 넘게 벗어나지 않는다
MIN_DURATION = 0.25
MAX_DURATION = 6.0
SIGMA_DEG = 7.0         # 처음 탐색 폭(도)
SIGMA_DUR = 0.25        # 처음 탐색 폭(초)
SIGMA_FLOOR_DEG = 0.8   # 너무 좁아져 제자리걸음이 되지 않게
SIGMA_FLOOR_DUR = 0.03


def _flatten(moves):
    """(각도들, 시간들, 자리표) 로 편다. grasp 단계는 건드리지 않는다."""
    angles, durations, slots = [], [], []
    for i, frame in enumerate(moves):
        if not isinstance(frame, dict) or not frame.get('pose'):
            continue
        for joint in sorted(frame['pose']):
            angles.append(float(frame['pose'][joint]))
            slots.append((i, joint))
        durations.append(float(frame.get('duration') or 1.0))
    return angles, durations, slots


def _rebuild(moves, angles, durations, slots):
    out = copy.deepcopy(moves)
    for value, (i, joint) in zip(angles, slots):
        out[i]['pose'][joint] = round(value, 1)
    k = 0
    for frame in out:
        if isinstance(frame, dict) and frame.get('pose'):
            frame['duration'] = round(durations[k], 2)
            k += 1
    return out


def _limits_for(slots):
    """각 자리의 (하한, 상한). 관절 한계와 원래 값 근처를 함께 지킨다."""
    out = []
    for _i, joint in slots:
        lo, hi = me.ALL_LIMITS.get(joint, (-180.0, 180.0))
        out.append((lo, hi))
    return out


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def optimize(moves, iters=12, pop=40, elite=8, seed=0, workers=None,
             max_drift=MAX_DRIFT, verbose=False):
    """CEM 으로 각도·시간을 다듬는다. (개선된 moves, 기록) 을 돌려준다."""
    rng = random.Random(seed)

    base_a, base_d, slots = _flatten(moves)
    if not base_a:
        return moves, {'reason': '다듬을 각도가 없습니다', 'curve': []}

    limits = _limits_for(slots)
    start = sim_eval.evaluate(moves, fast=True)
    if not start['ok']:
        return moves, {'reason': '시작 동작이 이미 실행 불가입니다', 'curve': []}

    mean_a, mean_d = list(base_a), list(base_d)
    sig_a = [SIGMA_DEG] * len(base_a)
    sig_d = [SIGMA_DUR] * len(base_d)

    best = {'score': start['score'], 'moves': moves, 'result': start}
    curve = [start['score']]

    pool = None
    if workers and workers > 1:
        try:
            import multiprocessing as mp
            pool = mp.Pool(workers)
        except Exception:
            pool = None

    try:
        for it in range(iters):
            cands = []
            for _ in range(pop):
                a = [_clip(_clip(rng.gauss(mean_a[k], sig_a[k]),
                                 base_a[k] - max_drift, base_a[k] + max_drift),
                           limits[k][0], limits[k][1])
                     for k in range(len(base_a))]
                d = [_clip(rng.gauss(mean_d[k], sig_d[k]),
                           MIN_DURATION, MAX_DURATION)
                     for k in range(len(base_d))]
                cands.append(_rebuild(moves, a, d, slots))

            if pool is not None:
                scored = pool.map(_score_one, cands)
            else:
                scored = [_score_one(c) for c in cands]

            ranked = sorted(zip(scored, range(len(cands))),
                            key=lambda x: -x[0])
            top = [i for sc, i in ranked[:elite] if sc > 0]
            if not top:
                # 전멸했다 - 탐색 폭을 줄이고 다시.
                sig_a = [max(SIGMA_FLOOR_DEG, s * 0.6) for s in sig_a]
                sig_d = [max(SIGMA_FLOOR_DUR, s * 0.6) for s in sig_d]
                curve.append(best['score'])
                continue

            elites_a, elites_d = [], []
            for i in top:
                a, d, _ = _flatten(cands[i])
                elites_a.append(a)
                elites_d.append(d)

            mean_a = [sum(e[k] for e in elites_a) / len(elites_a)
                      for k in range(len(base_a))]
            mean_d = [sum(e[k] for e in elites_d) / len(elites_d)
                      for k in range(len(base_d))]
            sig_a = [max(SIGMA_FLOOR_DEG, _std([e[k] for e in elites_a]))
                     for k in range(len(base_a))]
            sig_d = [max(SIGMA_FLOOR_DUR, _std([e[k] for e in elites_d]))
                     for k in range(len(base_d))]

            top_score, top_i = ranked[0]
            if top_score > best['score']:
                best = {'score': top_score, 'moves': cands[top_i],
                        'result': sim_eval.evaluate(cands[top_i], fast=True)}
            curve.append(best['score'])
            if verbose:
                print('    탐색 %2d회차: 최고 %d점 (이번 세대 %d점)'
                      % (it + 1, best['score'], top_score))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # 채택 전에는 반드시 실물이 쓰는 검증기로 다시 본다.
    final = sim_eval.evaluate(best['moves'])
    if not final['ok'] or final['score'] < start['score']:
        return moves, {'curve': curve, 'kept': False,
                       'reason': '정식 검증에서 원본보다 낫지 않았습니다',
                       'from': curve[0], 'to': curve[0]}

    return best['moves'], {'curve': curve, 'kept': True,
                           'from': curve[0], 'to': final['score'],
                           'evaluations': iters * pop}


def _score_one(moves):
    r = sim_eval.evaluate(moves, fast=True)
    return r['score'] if r['ok'] else 0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset')
    ap.add_argument('--json')
    ap.add_argument('--iters', type=int, default=12)
    ap.add_argument('--pop', type=int, default=40)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out')
    args = ap.parse_args()

    if args.preset:
        from motion_presets import PRESETS
        moves = PRESETS[args.preset]['moves']
        name = args.preset
    elif args.json:
        with open(args.json, encoding='utf-8') as fh:
            data = json.load(fh)
        moves = data.get('moves', data)
        name = os.path.basename(args.json)
    else:
        ap.error('--preset 또는 --json')

    print('── %s 다듬기 (탐색 %d회 x %d개 = 채점 %d번)'
          % (name, args.iters, args.pop, args.iters * args.pop))
    t = time.time()
    out, info = optimize(moves, iters=args.iters, pop=args.pop,
                         workers=args.workers, verbose=True)
    print('  %d점 -> %d점  (%.0f초)' % (info.get('from', 0), info.get('to', 0),
                                        time.time() - t))
    if not info.get('kept'):
        print('  남기지 않음: %s' % info.get('reason'))
        return
    print()
    print(sim_eval.describe(sim_eval.evaluate(out)))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'moves': out}, fh, ensure_ascii=False, indent=2)
        print('\n저장: %s' % args.out)


if __name__ == '__main__':
    main()
