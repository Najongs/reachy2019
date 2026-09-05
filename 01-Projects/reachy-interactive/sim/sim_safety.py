"""시연 전에 위험한 동작을 걸러 낸다 - 선분 대 선분으로 정확히.

실행기의 충돌 검사는 팔을 **점 몇 개**로 줄여서 본다. 링크마다 t=0, 0.34,
0.67 세 점씩이라, 가장 긴 상완(28cm)에서는 점 간격이 9.4cm 다. 그런데 두 팔
사이의 최소 간격 기준도 9cm 다. 두 링크가 열십자로 교차해도, 각자의 표본점이
서로 9cm 넘게 떨어져 있으면 통과한다. 실제로는 부딪힌다.

여기서는 팔을 **선분의 사슬**로 보고 선분 대 선분 최단거리를 직접 푼다.
표본점 간격과 무관하게 정확하다. 시간 방향으로도 훨씬 촘촘히 본다 -
빠른 동작은 10Hz 표본 사이로 빠져나갈 수 있다.

링크 두께도 넣는다. 실행기는 중심선만 보므로 '9cm 떨어졌다' 는 실제 공기
간격이 아니다. 팔 반지름을 빼야 사람이 보는 여유가 나온다.

등급:
    안전    표면 간격 5cm 이상
    주의    2~5cm - 조립 오차와 처짐을 생각하면 시연에서는 피하는 게 낫다
    위험    2cm 미만 - 실물에서 닿을 수 있다

사용:
    python3 sim/sim_safety.py --all-presets
    python3 sim/sim_safety.py --candidates
    python3 sim/sim_safety.py --json moves.json
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                          # noqa: E402
import sim_eval                                   # noqa: E402


# 팔의 굵기(반지름, m). **3D 모델에서 잰 값이다** - reachy.glb 가 순기구학과
# 같은 좌표계의 휴식 자세로 되어 있어서, 링크 축 둘레의 정점까지 거리를 재면
# 그게 반지름이다 (sim/measure_arm.py). 처음에는 눈대중 두 값(3.5/4.5cm)을
# 썼는데, 판정이 그 숫자에 그대로 뒤집혀서(pick_to_tray: 3.0cm 안전 /
# 3.5cm 위험) 추측으로 둘 수 없었다.
#
# 이게 중요한 이유: 실행기의 검증기는 팔을 **굵기 없는 중심선**으로만 본다.
# 중심선이 금지 영역에서 5cm 떨어져 있어도, 전완 반지름이 3.6cm 면 실제
# 공기 틈은 1.4cm 다. 시연에서 조립 오차와 처짐까지 생각하면 아슬아슬하다.
try:
    import measure_arm
    RADII, RADII_SOURCE = measure_arm.measure()
except Exception:
    RADII = {'upper_arm': 0.044, 'forearm': 0.037, 'hand': 0.041}
    RADII_SOURCE = '재어 둔 값'

# 링크 순서대로 (길이 0 인 링크는 _links 에서 이미 빠진다)
_ORDER = ['upper_arm', 'forearm', 'hand']

AUDIT_HZ = 100          # 시간 방향 표본. 실행기의 10Hz 보다 촘촘히 본다
SAFE_CM = 5.0
WARN_CM = 2.0


def _seg_seg_distance(p1, q1, p2, q2):
    """두 선분 사이의 최단거리. (거리, 선분1 위 점, 선분2 위 점).

    Ericson, Real-Time Collision Detection 의 표준 해법. 점을 촘촘히 찍어
    비교하는 방식과 달리 표본 간격에 좌우되지 않는다.
    """
    d1 = [q1[i] - p1[i] for i in range(3)]
    d2 = [q2[i] - p2[i] for i in range(3)]
    r = [p1[i] - p2[i] for i in range(3)]
    a = sum(x * x for x in d1)
    e = sum(x * x for x in d2)
    f = sum(d2[i] * r[i] for i in range(3))
    eps = 1e-12

    if a <= eps and e <= eps:
        s = t = 0.0
    elif a <= eps:
        s = 0.0
        t = min(1.0, max(0.0, f / e))
    else:
        c = sum(d1[i] * r[i] for i in range(3))
        if e <= eps:
            t = 0.0
            s = min(1.0, max(0.0, -c / a))
        else:
            b = sum(d1[i] * d2[i] for i in range(3))
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))

    c1 = [p1[i] + d1[i] * s for i in range(3)]
    c2 = [p2[i] + d2[i] * t for i in range(3)]
    return math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3))), c1, c2


def _links(chain, pose):
    """이 팔을 (시작점, 끝점, 반지름) 선분 목록으로. 길이 0 인 링크는 뺀다."""
    p = (0.0, 0.0, 0.0)
    r = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    nodes = [(0.0, chain[0][1][1], 0.0)]
    for name, translation, axis, _limits in chain:
        tv = me._mat_vec(r, translation)
        p = (p[0] + tv[0], p[1] + tv[1], p[2] + tv[2])
        r = me._mat_mul(r, me._rot(axis, pose.get(name, 0.0)))
        nodes.append(p)

    segs = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        if me._dist(a, b) < 1e-6:
            continue
        key = _ORDER[min(len(segs), len(_ORDER) - 1)]
        segs.append((a, b, RADII[key]))
    return segs


def _seg_point_distance(a, b, point):
    d = [b[i] - a[i] for i in range(3)]
    L = sum(x * x for x in d)
    if L < 1e-12:
        return me._dist(a, point)
    t = min(1.0, max(0.0, sum((point[i] - a[i]) * d[i] for i in range(3)) / L))
    c = [a[i] + d[i] * t for i in range(3)]
    return me._dist(c, point)


def _seg_box_gap(a, b, radius):
    """선분과 몸통 상자의 표면 간격(m). 음수면 파고들었다."""
    box_x, box_y, box_z = (me.TORSO_BOX['x_max'], me.TORSO_BOX['y_abs_max'],
                           me.TORSO_BOX['z_min'])
    best = 1e9
    for k in range(21):                      # 선분을 잘게 나눠 상자까지 거리
        t = k / 20.0
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        z = a[2] + (b[2] - a[2]) * t
        # 상자 밖으로 나가는 각 축 방향 여유 중 가장 큰 것이 실제 여유
        gap = max(x - box_x, abs(y) - box_y, box_z - z)
        best = min(best, gap)
    return best - radius


# 몸통 상자는 '손이 들어오면 안 되는 영역' 이지 몸의 외곽선이 아니다. 팔을
# 가만히 늘어뜨리기만 해도 상자 옆면에서 0.5cm 다(어깨 장착점 y=±0.19, 상자
# |y|<=0.14). 그래서 몸통·머리는 절대값이 아니라 **휴식 자세보다 얼마나 더
# 가까워졌는가** 로 본다. 가만히 서 있는 것보다 가깝지 않으면 그 동작이 새로
# 만든 위험은 없다.
#
# 반대로 두 팔 사이와 같은 팔 안쪽은 절대값이 의미 있다 - 쉴 때는 38cm 떨어져
# 있으므로, 가까워졌다면 전부 동작이 만든 것이다.
_BASELINE = {}


def _baseline():
    if _BASELINE:
        return _BASELINE
    rest = dict(sim_eval.REST)
    for key, gap in _pose_gaps(rest).items():
        _BASELINE[key] = gap
    return _BASELINE


def _pose_gaps(pose):
    """이 자세 하나에서 쌍별 최소 표면 간격."""
    segs = {side: _links(chain, pose) for side, chain in me.CHAINS.items()}
    out = {'arm_arm': 9e9, 'torso': 9e9, 'head': 9e9, 'self': 9e9}

    for sa, ea, ra in segs['right_arm']:
        for sb, eb, rb in segs['left_arm']:
            d, _, _ = _seg_seg_distance(sa, ea, sb, eb)
            out['arm_arm'] = min(out['arm_arm'], d - ra - rb)

    for links in segs.values():
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                d, _, _ = _seg_seg_distance(links[i][0], links[i][1],
                                            links[j][0], links[j][1])
                out['self'] = min(out['self'], d - links[i][2] - links[j][2])
        for a, b, rad in links:
            out['torso'] = min(out['torso'], _seg_box_gap(a, b, rad))
            d = _seg_point_distance(a, b, me.HEAD_SPHERE_CENTER)
            out['head'] = min(out['head'], d - me.HEAD_SPHERE_RADIUS - rad)
    return out


def audit(moves, hz=AUDIT_HZ):
    """궤적 전체를 훑어 가장 아슬아슬한 순간들을 찾는다."""
    sim = sim_eval.simulate(moves, hz=hz, check_collision=False)
    if not sim['ok']:
        return {'ok': False, 'error': sim['error']}

    worst = {'arm_arm': (9e9, None), 'torso': (9e9, None),
             'head': (9e9, None), 'self': (9e9, None)}

    for s in sim['samples']:
        pose = s['pose']
        segs = {side: _links(chain, pose) for side, chain in me.CHAINS.items()}

        # 팔 대 팔 - 선분끼리 직접 푼다
        for sa, ea, ra in segs['right_arm']:
            for sb, eb, rb in segs['left_arm']:
                d, _, _ = _seg_seg_distance(sa, ea, sb, eb)
                gap = d - ra - rb
                if gap < worst['arm_arm'][0]:
                    worst['arm_arm'] = (gap, s['t'])

        for side, links in segs.items():
            # 같은 팔 안에서 서로 닿는 경우 (이웃 링크는 원래 붙어 있으니 뺀다)
            for i in range(len(links)):
                for j in range(i + 2, len(links)):
                    d, _, _ = _seg_seg_distance(links[i][0], links[i][1],
                                                links[j][0], links[j][1])
                    gap = d - links[i][2] - links[j][2]
                    if gap < worst['self'][0]:
                        worst['self'] = (gap, s['t'])

            for a, b, rad in links:
                g = _seg_box_gap(a, b, rad)
                if g < worst['torso'][0]:
                    worst['torso'] = (g, s['t'])
                d = _seg_point_distance(a, b, me.HEAD_SPHERE_CENTER)
                g = d - me.HEAD_SPHERE_RADIUS - rad
                if g < worst['head'][0]:
                    worst['head'] = (g, s['t'])

    base = _baseline()
    gaps = {k: (round(v[0] * 100, 1), v[1]) for k, v in worst.items()}

    # 절대값이 의미 있는 쌍 (쉴 때는 멀리 떨어져 있다)
    absolute = min(gaps['arm_arm'][0], gaps['self'][0])
    # 구조적으로 원래 가까운 쌍 - 휴식보다 얼마나 더 가까워졌나
    closed_in = {k: round((base[k] - worst[k][0]) * 100, 1)
                 for k in ('torso', 'head')}
    worst_close = max(closed_in.values())
    pierced = min(gaps['torso'][0], gaps['head'][0]) < 0

    if pierced or absolute < WARN_CM:
        grade = '위험'
    elif absolute < SAFE_CM or worst_close > 3.0:
        grade = '주의'
    else:
        grade = '안전'

    return {'ok': True, 'grade': grade, 'gaps': gaps,
            'closed_in_cm': closed_in, 'pierced': pierced,
            'tightest_cm': min(absolute, gaps['torso'][0], gaps['head'][0]),
            'arm_arm_cm': gaps['arm_arm'][0], 'self_cm': gaps['self'][0],
            # 굵기를 빼기 전 값. 실행기가 보는 것이 이쪽이라, 두 숫자를
            # 나란히 놓아야 '두께 때문에 좁아진 것' 이 드러난다.
            'centerline_cm': {k: round((v[0] + 2 * RADII['forearm']) * 100, 1)
                              for k, v in gaps.items()}}


LABELS = {'arm_arm': '두 팔 사이', 'torso': '몸통', 'head': '머리',
          'self': '같은 팔 안'}


def describe(a):
    if not a['ok']:
        return '실행 불가: %s' % a.get('error')
    bits = []
    if a['arm_arm_cm'] < 30:
        bits.append('두 팔 %.1fcm(%.1f초)'
                    % (a['arm_arm_cm'], a['gaps']['arm_arm'][1] or 0))
    if a['self_cm'] < 30:
        bits.append('같은 팔 %.1fcm' % a['self_cm'])
    for key, ko in (('torso', '몸통'), ('head', '머리')):
        closed = a['closed_in_cm'][key]
        if closed > 0.5 or a['gaps'][key][0] < 0:
            bits.append('%s 휴식보다 %.1fcm 접근(%.1f초, 남은 틈 %.1fcm)'
                        % (ko, closed, a['gaps'][key][1] or 0,
                           a['gaps'][key][0]))
    return '%-4s %s' % (a['grade'], ', '.join(bits) or '여유 충분')


def main():
    import argparse

    global RADII

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset')
    ap.add_argument('--all-presets', action='store_true')
    ap.add_argument('--candidates', nargs='?',
                    const=os.path.join(HERE, '..', 'config',
                                       'motion_candidates.json'))
    ap.add_argument('--json')
    ap.add_argument('--hz', type=int, default=AUDIT_HZ)
    ap.add_argument('--percentile', type=float,
                    help='모델에서 다시 잰다 (기본 90, 보수적으로 보려면 95)')
    args = ap.parse_args()

    if args.percentile:
        import measure_arm
        RADII, source = measure_arm.measure(percentile=args.percentile)
        globals()['RADII'] = RADII
    else:
        source = RADII_SOURCE
    _BASELINE.clear()
    print('팔 굵기 (%s): 상완 %.1fcm, 전완 %.1fcm, 손 %.1fcm\n'
          % (source, RADII['upper_arm'] * 100, RADII['forearm'] * 100,
             RADII['hand'] * 100))

    items = []
    if args.all_presets or args.preset:
        from motion_presets import PRESETS
        names = [args.preset] if args.preset else list(PRESETS)
        items = [(n, PRESETS[n]['moves']) for n in names]
    elif args.candidates:
        with open(args.candidates, encoding='utf-8') as fh:
            for i, c in enumerate(json.load(fh)):
                items.append((c.get('idea', '후보%d' % i)[:34], c['moves']))
    elif args.json:
        with open(args.json, encoding='utf-8') as fh:
            d = json.load(fh)
        items = [(os.path.basename(args.json), d.get('moves', d))]
    else:
        ap.error('--all-presets / --preset / --candidates / --json 중 하나')

    risky = []
    for name, moves in items:
        a = audit(moves, hz=args.hz)
        print('  %-36s %s' % (name[:36], describe(a)))
        if a['ok'] and a['grade'] != '안전':
            risky.append((name, a))

    print()
    if risky:
        print('시연 전에 손봐야 할 동작 %d개:' % len(risky))
        for name, a in sorted(risky, key=lambda x: x[1]['tightest_cm']):
            print('  · %s — %s (%.1fcm)' % (name[:40], a['grade'],
                                            a['tightest_cm']))
    else:
        print('전부 안전 등급입니다 (표면 간격 %.0fcm 이상).' % SAFE_CM)


if __name__ == '__main__':
    main()
