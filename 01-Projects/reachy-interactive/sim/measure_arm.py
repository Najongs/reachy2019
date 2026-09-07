"""팔 굵기를 3D 모델에서 직접 잰다 - 추측 대신 실측.

sim_safety 의 판정은 팔 반지름 가정에 그대로 좌우된다(pick_to_tray 는 3.0cm 로
보면 안전, 3.5cm 면 위험). 그런데 자를 대지 않고도 잴 방법이 있었다:
03-Resources/3d-models/reachy.glb 가 **motion_exec 의 순기구학과 같은 좌표계의
휴식 자세**로 되어 있다. 경계가 y ±0.248(어깨 ±0.19 + 팔 두께), z 바닥 -0.687
(손끝 -0.64)로 정확히 맞는다.

그래서 어깨->팔꿈치, 팔꿈치->손목 축을 알고 있으니 그 축 둘레 정점까지의 거리를
재면 그게 링크 반지름이다.

바깥 정점 몇 개에 끌려가지 않도록 90 백분위를 쓴다. 상완의 최대값(12cm)은 어깨
브래킷이 섞여 든 것이라 대표값이 될 수 없다.

사용:
    python3 sim/measure_arm.py
    python3 sim/measure_arm.py --percentile 95
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import para  # PARA 기준 경로
DEFAULT_GLB = os.path.join(para.RESOURCES,
                           '3d-models', 'reachy.glb')

# 재어 둔 값 (reachy.glb, 95 백분위 / 관절 여백 5cm). 모델을 못 읽을 때
# 되돌아갈 값이자 sim_safety 의 기본값이다.
#
# 관절 여백이 중요하다. 2cm 로 두면 어깨 브래킷이 상완 표본에 섞여 백분위를
# 올릴수록 반지름이 4.6 -> 6.9 -> 10.1cm 로 튀고, 그러면 가만히 서 있는 자세가
# '위험' 으로 나온다. 4cm 이상 잘라내면 3.9~5.0cm 로 안정된다. 전완과 손은
# 어느 설정에서도 3.6~3.9 / 4.1~4.5cm 로 흔들리지 않는다 - 믿을 수 있다.
MEASURED = {'upper_arm': 0.044, 'forearm': 0.037, 'hand': 0.041}

# 휴식 자세에서의 링크 축 (torso 프레임, m)
SEGMENTS = [
    ('upper_arm', '상완', (0, -0.19, 0.0), (0, -0.19, -0.28)),
    ('forearm', '전완', (0, -0.19, -0.28), (0, -0.19, -0.53)),
    ('hand', '손/그리퍼', (0, -0.20, -0.53), (0, -0.20, -0.64)),
]


def measure(glb=None, percentile=95, joint_pad=0.05, max_radius=0.12):
    """각 링크의 반지름(m) 과 어떻게 얻었는지. 못 읽으면 재어 둔 값."""
    try:
        import numpy as np
        import trimesh
    except ImportError:
        return dict(MEASURED), '라이브러리 없음 - 재어 둔 값 사용'

    path = glb or DEFAULT_GLB
    if not os.path.exists(path):
        return dict(MEASURED), '모델 파일 없음 - 재어 둔 값 사용'

    try:
        scene = trimesh.load(path)
        chunks = []
        for node in scene.graph.nodes_geometry:
            T, name = scene.graph[node]
            g = scene.geometry[name]
            if max(g.extents) >= 1.0:
                continue                    # 바닥판/배경
            chunks.append(trimesh.transform_points(g.vertices, T))
        V = np.vstack(chunks)
    except Exception as e:
        return dict(MEASURED), '모델을 읽지 못했습니다(%s) - 재어 둔 값' % e

    out = {}
    for key, _ko, p0, p1 in SEGMENTS:
        a, b = np.array(p0), np.array(p1)
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / L
        w = V - a
        t = w @ u
        sel = (t > joint_pad) & (t < L - joint_pad)   # 관절 근처는 다른 부품
        if sel.sum() < 50:
            out[key] = MEASURED[key]
            continue
        perp = w[sel] - np.outer(t[sel], u)
        r = np.linalg.norm(perp, axis=1)
        r = r[r < max_radius]
        out[key] = float(np.percentile(r, percentile))
    return out, '%s 에서 %g 백분위로 실측' % (os.path.basename(path), percentile)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--glb', default=None)
    ap.add_argument('--percentile', type=float, default=95)
    ap.add_argument('--joint-pad', type=float, default=0.05,
                    help='관절 근처 이만큼(m)은 빼고 잰다 - 브래킷이 섞인다')
    args = ap.parse_args()

    radii, how = measure(args.glb, args.percentile, args.joint_pad)
    print('팔 굵기 (%s)' % how)
    for key, ko, _p0, _p1 in SEGMENTS:
        print('  %-10s 반지름 %.1f cm' % (ko, radii[key] * 100))


if __name__ == '__main__':
    main()
