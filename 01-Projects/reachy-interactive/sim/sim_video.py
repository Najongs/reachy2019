"""동작을 영상으로 남긴다 - 로봇 없이, 순기구학만으로.

정지 3면도로는 "손이 어디를 지나갔나" 는 알 수 있어도 "어떻게 움직였나" 는
안 보인다. 빠른지 느린지, 어느 대목에서 멈칫하는지, 두 팔이 언제 스치는지는
움직이는 것을 봐야 한다.

로봇도 브라우저도 필요 없다. motion_exec 의 순기구학으로 프레임마다 팔의
링크 위치를 구해 그리고, 실측한 팔 굵기(sim/measure_arm.py)로 두께를 준다.
금지 구역(몸통 상자, 머리 구)도 함께 그려서 어디에 가까워지는지 보이게 한다.

사용:
    python3 sim/sim_video.py --preset wave -o wave.mp4
    python3 sim/sim_video.py --candidate 3 -o out.mp4
    python3 sim/sim_video.py --json a.json --compare b.json -o before_after.mp4
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                          # noqa: E402
import sim_eval                                   # noqa: E402
import sim_safety                                 # noqa: E402

FPS = 20
TRAIL = 25          # 손 궤적 꼬리를 몇 프레임이나 남길지


def _frames(moves, fps=FPS):
    """프레임마다 (시각, 양팔 링크 노드, 손 위치)."""
    sim = sim_eval.simulate(moves, hz=fps, check_collision=False)
    if not sim['ok']:
        return None, sim['error']
    out = []
    for s in sim['samples']:
        arms = {}
        for side, chain in me.CHAINS.items():
            segs = sim_safety._links(chain, s['pose'])
            nodes = [segs[0][0]] + [b for _a, b, _r in segs] if segs else []
            arms[side] = (nodes, [r for _a, _b, r in segs])
        out.append({'t': s['t'], 'arms': arms, 'hands': s['hands'],
                    'pose': s['pose']})
    return out, None


def _draw_keepouts(ax):
    """몸통 상자와 머리 구를 옅게. 어디에 가까워지는지 보이게."""
    import numpy as np

    b = me.TORSO_BOX
    x0, x1 = -0.25, b['x_max']
    y0, y1 = -b['y_abs_max'], b['y_abs_max']
    z0, z1 = b['z_min'], 0.20
    for zs in (z0, z1):
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zs] * 5,
                color='0.75', lw=0.8)
    for xs, ys in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        ax.plot([xs, xs], [ys, ys], [z0, z1], color='0.75', lw=0.8)

    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 9)
    r = me.HEAD_SPHERE_RADIUS
    c = me.HEAD_SPHERE_CENTER
    xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='0.8', lw=0.5)


def _setup(ax, title):
    ax.set_xlim(-0.15, 0.62)
    ax.set_ylim(-0.52, 0.52)
    ax.set_zlim(-0.72, 0.32)
    ax.set_box_aspect((0.77, 1.04, 1.04))
    ax.set_xlabel('x fwd', fontsize=7, labelpad=-4)
    ax.set_ylabel('y left', fontsize=7, labelpad=-4)
    ax.set_zlabel('z up', fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6)
    # 사람이 로봇을 살짝 옆에서 보는 각도. 정면(azim -90)은 팔이 겹쳐 보인다.
    ax.view_init(elev=12, azim=-72)
    ax.set_title(title, fontsize=9)


def _setup_front(ax, title):
    """사람이 로봇 앞에 서서 보는 그대로 (y-z). 제스처는 이쪽이 읽기 쉽다."""
    ax.set_xlim(0.55, -0.55)          # y 를 뒤집어야 화면 왼쪽이 로봇의 오른팔
    ax.set_ylim(-0.72, 0.32)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25)
    b = me.TORSO_BOX
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((-b['y_abs_max'], b['z_min']),
                           2 * b['y_abs_max'], 0.2 - b['z_min'],
                           color='0.9', zorder=0))
    ax.add_patch(Circle((me.HEAD_SPHERE_CENTER[1], me.HEAD_SPHERE_CENTER[2]),
                        me.HEAD_SPHERE_RADIUS, color='0.82', zorder=0))


def render_video(moves_list, path, labels=None, fps=FPS, header=None):
    """한 개 또는 여러 개(비교)의 동작을 나란히 놓고 영상으로 만든다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import imageio.v2 as imageio

    seqs = []
    for mv in moves_list:
        fr, err = _frames(mv, fps)
        if fr is None:
            return None, err
        seqs.append(fr)
    n = len(seqs)
    total = max(len(s) for s in seqs)
    labels = labels or [''] * n

    # 3D 와 정면 뷰를 같이 봐야 '어떻게 움직였나' 가 읽힌다. 하나면 좌우로,
    # 둘을 비교할 때는 위아래로 놓는다 (같은 시각을 나란히 보게).
    if n == 1:
        fig = plt.figure(figsize=(10.4, 5.4), dpi=105)
        axes = [fig.add_subplot(1, 2, 1, projection='3d')]
        fronts = [fig.add_subplot(1, 2, 2)]
    else:
        fig = plt.figure(figsize=(5.0 * n, 8.6), dpi=100)
        axes = [fig.add_subplot(2, n, i + 1, projection='3d') for i in range(n)]
        fronts = [fig.add_subplot(2, n, n + i + 1) for i in range(n)]

    colors = {'right_arm': 'tab:red', 'left_arm': 'tab:blue'}
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for k in range(total):
            for ax, fr_ax, seq, lab in zip(axes, fronts, seqs, labels):
                ax.clear()
                fr_ax.clear()
                j = min(k, len(seq) - 1)
                f = seq[j]
                _setup(ax, '%s   t=%.1fs' % (lab, f['t']))
                _setup_front(fr_ax, 'front view')
                _draw_keepouts(ax)
                for side, (nodes, radii) in f['arms'].items():
                    if not nodes:
                        continue
                    xs = [p[0] for p in nodes]
                    ys = [p[1] for p in nodes]
                    zs = [p[2] for p in nodes]
                    # 링크 두께를 선 굵기로 (실측 반지름 기준)
                    lw = 2 + 60 * radii[0]
                    ax.plot(xs, ys, zs, '-', color=colors[side], lw=lw,
                            solid_capstyle='round', alpha=0.9)
                    ax.scatter(xs, ys, zs, s=14, color=colors[side], zorder=5)
                    fr_ax.plot(ys, zs, '-', color=colors[side], lw=lw,
                               solid_capstyle='round', alpha=0.9, zorder=3)
                    fr_ax.plot(ys, zs, 'o', color=colors[side], ms=3, zorder=4)
                    # 손이 지나온 자리
                    lo = max(0, j - TRAIL)
                    tail = [seq[q]['hands'][side] for q in range(lo, j + 1)]
                    if len(tail) > 1:
                        ax.plot([p[0] for p in tail], [p[1] for p in tail],
                                [p[2] for p in tail], '-', color=colors[side],
                                lw=1.2, alpha=0.5)
                        fr_ax.plot([p[1] for p in tail], [p[2] for p in tail],
                                   '-', color=colors[side], lw=1.4, alpha=0.5,
                                   zorder=2)
            if header:
                fig.suptitle(_ascii(header), fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95), pad=1.6)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            # libx264 + yuv420p 는 가로·세로가 짝수여야 한다. matplotlib 이
            # 홀수(예: 1092x567)를 내면 ffmpeg 이 파이프를 끊어 버린다.
            h, w = buf.shape[:2]
            buf = buf[:h - (h % 2), :w - (w % 2)]
            writer.append_data(buf)
    finally:
        writer.close()
        plt.close(fig)
    return path, None


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset')
    ap.add_argument('--candidate', type=int, help='motion_candidates.json 의 번호')
    ap.add_argument('--json')
    ap.add_argument('--compare', help='이 동작과 나란히 놓고 비교')
    ap.add_argument('-o', '--out', default='motion.mp4')
    ap.add_argument('--fps', type=int, default=FPS)
    args = ap.parse_args()

    def load(spec):
        if spec is None:
            return None, None
        with open(spec, encoding='utf-8') as fh:
            d = json.load(fh)
        return d.get('moves', d), os.path.basename(spec)

    if args.preset:
        from motion_presets import PRESETS
        moves, name = PRESETS[args.preset]['moves'], args.preset
    elif args.candidate is not None:
        path = os.path.join(HERE, '..', 'config', 'motion_candidates.json')
        with open(path, encoding='utf-8') as fh:
            c = json.load(fh)[args.candidate]
        moves, name = c['moves'], c.get('idea', '후보')[:30]
    elif args.json:
        moves, name = load(args.json)
    else:
        ap.error('--preset / --candidate / --json 중 하나')

    lst, labs = [moves], [_ascii(name)]
    if args.compare:
        other, oname = load(args.compare)
        lst, labs = [other, moves], [_ascii(oname), _ascii(name)]

    score = sim_eval.evaluate(moves)
    grade = sim_safety.audit(moves, hz=40)
    # 그림 안의 글자는 아스키만 (한글 폰트가 없어 네모로 깨진다)
    GRADE_EN = {'안전': 'safe', '주의': 'caution', '위험': 'risky'}
    out, err = render_video(lst, args.out, labels=labs, fps=args.fps,
                            header='%d pts   %s' % (
                                score['score'],
                                GRADE_EN.get(grade.get('grade'), '?')))
    if err:
        print('영상 실패: %s' % err)
        return
    print('영상: %s (%.1fMB)' % (out, os.path.getsize(out) / 1e6))


def _ascii(text):
    """이 서버에 한글 폰트가 없다 - 그림 안의 글자는 아스키만."""
    out = ''.join(c if ord(c) < 128 else '' for c in (text or ''))
    return ' '.join(out.split()) or 'motion'


if __name__ == '__main__':
    main()
