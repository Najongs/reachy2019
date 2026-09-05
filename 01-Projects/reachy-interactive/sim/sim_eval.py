"""동작을 실물 없이 돌려 보고 점수를 매긴다.

로봇도 GPU도 필요 없다. robot/motion_exec.py 의 검증기와 순기구학을 그대로
빌려 쓰므로, 여기서 통과한 동작은 실물에서도 같은 판정을 받는다 - 시뮬레이터를
따로 만들면 시뮬에서만 되는 동작이 나온다.

무엇을 보는가:

  * 안전       validate() 가 통과시키나. 각도를 얼마나 깎았나(clamp).
               속도 상한 때문에 시간을 얼마나 늘렸나(stretch).
  * 크기       손이 실제로 얼마나 움직였나. 복도 끝에서 보이는 동작인가.
  * 여유       금지 구역(몸통·머리·반대팔)에 얼마나 가까이 갔나.
  * 성격       손이 몇 번 방향을 바꾸나(손 흔들기는 왕복해야 하고, 가리키기는
               아니다). 쓰지도 않는 관절을 키프레임에 넣었나.
  * 마무리     마지막 자세가 휴식 자세에서 얼마나 떨어져 있나.

점수 하나로 줄이지 않고 '문제 목록'을 함께 돌려준다. opus 에게 고치라고 할 때
점수보다 "오른손이 3cm 밖에 안 움직입니다" 가 훨씬 잘 먹힌다.

사용:
    python3 sim/sim_eval.py --preset wave
    python3 sim/sim_eval.py --all-presets
    python3 sim/sim_eval.py --json moves.json --render out.png
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'robot'))

import motion_exec as me            # noqa: E402


# 이 로봇에 실제로 있는 관절. 왼쪽 그리퍼와 양쪽 wrist_roll 은 없다
# (base_pose.connect 의 custom hands 와 같은 목록).
JOINTS = [
    'right_arm.shoulder_pitch', 'right_arm.shoulder_roll', 'right_arm.arm_yaw',
    'right_arm.elbow_pitch', 'right_arm.hand.forearm_yaw',
    'right_arm.hand.wrist_pitch', 'right_arm.hand.gripper',
    'left_arm.shoulder_pitch', 'left_arm.shoulder_roll', 'left_arm.arm_yaw',
    'left_arm.elbow_pitch', 'left_arm.hand.forearm_yaw',
    'left_arm.hand.wrist_pitch',
    'head.left_antenna', 'head.right_antenna',
]

REST = {j: 0.0 for j in JOINTS}

SAMPLE_HZ = 25          # 궤적 샘플링. 실행기는 50Hz 지만 평가는 절반이면 충분하다
MIN_REACH_CM = 12.0     # 이보다 작으면 복도에서 안 보인다 (실측 감각)
NEAR_MISS_CM = 4.0      # 금지 구역에 이만큼 붙으면 아슬아슬하다
GRASP_SECONDS = 1.2     # 힘센서로 쥐거나 펴는 한 단계에 걸리는 시간(실측 감각)

# 팔 관절 접두사. 안테나만 움직이는 동작에 '손이 안 움직인다' 고 하면 안 된다.
ARM_PREFIXES = ('right_arm.', 'left_arm.')


def _clearance(point, side):
    """이 점이 금지 구역에서 얼마나 떨어져 있나(m). 음수면 이미 안쪽."""
    x, y, z = point
    gaps = []

    # 몸통 상자: 밖으로 나가는 가장 가까운 면까지의 거리
    if (x < me.TORSO_BOX['x_max'] and abs(y) < me.TORSO_BOX['y_abs_max']
            and z > me.TORSO_BOX['z_min']):
        gaps.append(-min(me.TORSO_BOX['x_max'] - x,
                         me.TORSO_BOX['y_abs_max'] - abs(y),
                         z - me.TORSO_BOX['z_min']))
    else:
        gaps.append(max(x - me.TORSO_BOX['x_max'],
                        abs(y) - me.TORSO_BOX['y_abs_max'],
                        me.TORSO_BOX['z_min'] - z))

    # 머리 구
    d = math.sqrt(sum((point[i] - me.HEAD_SPHERE_CENTER[i]) ** 2
                      for i in range(3)))
    gaps.append(d - me.HEAD_SPHERE_RADIUS)

    # 정중선
    if x < me.MIDLINE_FREE_X:
        if side == 'right_arm':
            gaps.append(me.MIDLINE_MARGIN - y)
        else:
            gaps.append(y + me.MIDLINE_MARGIN)

    return min(gaps)


def simulate(moves, seed=None, hz=SAMPLE_HZ):
    """검증하고 궤적을 샘플링한다.

    Returns:
        dict(ok, error, segments, samples, stretched_s, clamped)
        samples = [{'t':초, 'pose':{...}, 'hands':{side:(x,y,z)},
                    'clearance':{side:m}}, ...]
    """
    seed = dict(seed or REST)
    out = {'ok': False, 'error': None, 'segments': [], 'samples': [],
           'stretched_s': 0.0, 'clamped': []}

    asked = sum(m.get('duration', 0) or 0 for m in moves
                if isinstance(m, dict))

    try:
        segments = me.validate(moves, seed, JOINTS)
    except me.ValidationError as e:
        out['error'] = str(e)
        return out
    except Exception as e:                      # 형식 오류 등
        out['error'] = '{}: {}'.format(type(e).__name__, e)
        return out

    out['ok'] = True
    out['segments'] = segments
    # validate 는 그리퍼 단계를 ('grasp', 'close'|'open') 으로 내보낸다 -
    # duration 자리에 숫자가 아니라 명령이 온다. 각도 키프레임이 아니므로
    # 궤적에서는 빼고, 실제로 걸리는 시간만 얹는다.
    out['grasps'] = [a for kind, a in segments if kind == 'grasp']
    got = sum(d for kind, d in segments if kind != 'grasp')
    got += GRASP_SECONDS * len(out['grasps'])
    out['stretched_s'] = round(max(0.0, got - asked), 2)

    # 각도를 얼마나 깎았나: 요청한 값과 검증 뒤 값의 차이
    angle_segs = [seg for seg in segments if seg[0] != 'grasp']
    for raw, (pose, _d) in zip(
            [m for m in moves if isinstance(m, dict) and m.get('pose')],
            angle_segs):
        for joint, want in (raw.get('pose') or {}).items():
            if joint not in pose:
                continue
            gap = abs(float(want) - pose[joint])
            if gap > 0.5:
                out['clamped'].append({'joint': joint, 'asked': float(want),
                                       'used': round(pose[joint], 1),
                                       'cut_deg': round(gap, 1)})

    # 궤적 샘플링 - 실행기와 같은 minjerk 보간
    current = dict(seed)
    t = 0.0
    for kind, duration in segments:
        if kind == 'grasp':
            t += GRASP_SECONDS      # 힘센서로 쥐고 펴는 데 걸리는 시간
            continue
        pose = kind
        targets = dict(current)
        targets.update(pose)
        steps = max(2, int(duration * hz))
        for step in range(1, steps + 1):
            k = me.minjerk(step / float(steps))
            sample = {j: current.get(j, 0.0)
                      + (targets.get(j, 0.0) - current.get(j, 0.0)) * k
                      for j in set(current) | set(targets)}
            hands, clear = {}, {}
            for side, chain in me.CHAINS.items():
                hands[side] = me.forward_kinematics(chain, sample)
                clear[side] = min(_clearance(p, side)
                                  for p in me._arm_points(chain, sample))
            out['samples'].append({
                't': round(t + duration * step / float(steps), 3),
                'pose': sample, 'hands': hands, 'clearance': clear})
        current = targets
        t += duration

    return out


def _joint_reversals(samples, side):
    """이 팔 관절들이 방향을 바꾼 횟수(가장 많은 관절 기준).

    손끝 위치만 보면 비틀림 왕복을 통째로 놓친다. 실제로 wave 프리셋은
    forearm_yaw 를 ±30 으로 흔드는데, 그 축 위에 있는 손끝은 거의 제자리라
    위치 기준으로는 '방향전환 0회' 로 나왔다 - 손 흔들기인지 아닌지를
    가려내야 하는 지표가 정작 손 흔들기를 못 알아본 셈이다.
    """
    best = 0
    for joint in JOINTS:
        if not joint.startswith(side + '.'):
            continue
        series = [s['pose'].get(joint, 0.0) for s in samples]
        deltas = [b - a for a, b in zip(series[:-1], series[1:])
                  if abs(b - a) > 0.15]          # 미세한 흔들림은 뺀다
        best = max(best, sum(1 for a, b in zip(deltas[:-1], deltas[1:])
                             if a * b < 0))
    return best


def _path_stats(samples, side):
    """이 팔의 손이 그린 궤적에서 길이·최대이동·방향전환 횟수."""
    pts = [s['hands'][side] for s in samples]
    if len(pts) < 2:
        return {'path_cm': 0.0, 'reach_cm': 0.0,
                'reversals': _joint_reversals(samples, side)}

    rest = pts[0]
    length = sum(me._dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
    reach = max(me._dist(rest, p) for p in pts)

    # 방향 전환: 연속한 이동 벡터의 내적이 음수로 바뀌는 횟수. 미세한
    # 흔들림은 세지 않도록 일정 길이 이상 움직인 구간만 본다.
    vecs = []
    for a, b in zip(pts[:-1], pts[1:]):
        v = tuple(b[i] - a[i] for i in range(3))
        if math.sqrt(sum(c * c for c in v)) > 0.004:
            vecs.append(v)
    reversals = 0
    for a, b in zip(vecs[:-1], vecs[1:]):
        if sum(a[i] * b[i] for i in range(3)) < 0:
            reversals += 1

    return {'path_cm': round(length * 100, 1),
            'reach_cm': round(reach * 100, 1),
            'reversals': max(reversals, _joint_reversals(samples, side))}


def evaluate(moves, intent=None):
    """동작을 평가한다. dict(ok, score, metrics, findings)."""
    sim = simulate(moves)
    findings = []

    if not sim['ok']:
        return {'ok': False, 'score': 0, 'error': sim['error'],
                'metrics': {}, 'findings': ['검증에서 막혔습니다: ' + sim['error']]}

    samples = sim['samples']
    metrics = {
        'duration_s': round(samples[-1]['t'], 2) if samples else 0.0,
        'keyframes': len(sim['segments']),
        'stretched_s': sim['stretched_s'],
        'clamped': sim['clamped'],
    }
    for side in ('right_arm', 'left_arm'):
        metrics[side] = _path_stats(samples, side)
        metrics[side]['min_clearance_cm'] = round(
            min(s['clearance'][side] for s in samples) * 100, 1)

    # 마무리: 마지막 자세가 휴식에서 얼마나 떨어져 있나
    last = samples[-1]['pose'] if samples else REST
    metrics['end_offset_deg'] = round(
        max(abs(last.get(j, 0.0)) for j in JOINTS), 1)

    # 한 번도 안 움직인 관절 = 키프레임에 넣을 이유가 없던 것
    asked = set()   # 이 동작이 건드리려 한 관절들
    for m in moves:
        if isinstance(m, dict):
            asked |= set((m.get('pose') or {}).keys())
    still = [j for j in sorted(asked)
             if max(abs(s['pose'].get(j, 0.0)) for s in samples) < 0.5]
    metrics['unused_joints'] = still

    # --- 사람이 읽을 수 있는 문제 목록 ------------------------------------
    uses_arms = any(j.startswith(ARM_PREFIXES) for j in asked)
    metrics['uses_arms'] = uses_arms
    moved = max(metrics['right_arm']['reach_cm'], metrics['left_arm']['reach_cm'])
    if uses_arms and moved < MIN_REACH_CM:
        findings.append(
            '손이 가장 멀리 가 봐야 %.0fcm 입니다. 복도 건너에서는 움직이는 '
            '줄도 모릅니다 - 어깨를 더 크게 쓰세요.' % moved)

    for side, ko in (('right_arm', '오른팔'), ('left_arm', '왼팔')):
        gap = metrics[side]['min_clearance_cm']
        if gap < NEAR_MISS_CM:
            findings.append(
                '%s이 금지 구역에 %.1fcm 까지 붙습니다. 통과는 했지만 실물에서는 '
                '조립 오차만으로도 닿습니다 - 여유를 두세요.' % (ko, gap))

    if sim['stretched_s'] > 0.5:
        findings.append(
            '속도 상한 때문에 전체 시간이 %.1f초 늘어났습니다. 요청한 duration '
            '이 그 각도 변화에는 너무 짧습니다 - 처음부터 늘려 잡으세요.'
            % sim['stretched_s'])

    if sim['clamped']:
        worst = max(sim['clamped'], key=lambda c: c['cut_deg'])
        findings.append(
            '한계를 넘은 각도가 %d개 깎였습니다 (가장 큰 것: %s %.0f도 -> %.0f도). '
            '깎이면 의도한 모양이 아니게 됩니다.'
            % (len(sim['clamped']), worst['joint'], worst['asked'], worst['used']))

    if still:
        findings.append(
            '한 번도 움직이지 않는 관절이 키프레임에 있습니다: %s. '
            '빼면 동작이 읽기 쉬워집니다.' % ', '.join(still[:4]))

    if metrics['end_offset_deg'] > 25:
        findings.append(
            '마지막 자세가 휴식에서 %.0f도 떨어져 있습니다. 끝나면 자동으로 '
            '돌아가지만, 그 복귀가 커서 동작이 뚝 끊긴 느낌이 됩니다.'
            % metrics['end_offset_deg'])

    if metrics['duration_s'] > 12:
        findings.append('%.1f초는 지나가는 사람에게 깁니다.'
                        % metrics['duration_s'])

    # --- 점수 ------------------------------------------------------------
    score = 100
    if uses_arms:
        score -= min(35, max(0, (MIN_REACH_CM - moved)) * 3)
    score -= min(20, max(0, (NEAR_MISS_CM - min(
        metrics['right_arm']['min_clearance_cm'],
        metrics['left_arm']['min_clearance_cm']))) * 5)
    score -= min(15, sim['stretched_s'] * 5)
    score -= min(15, len(sim['clamped']) * 5)
    score -= min(10, len(still) * 3)
    score -= min(10, max(0, metrics['end_offset_deg'] - 25) * 0.3)
    # 길이도 점수에 넣는다. 안 그러면 '100점인데 지적 있음' 이 나와서
    # 어느 쪽을 믿어야 할지 알 수 없다.
    score -= min(15, max(0, metrics['duration_s'] - 12) * 2)

    return {'ok': True, 'score': int(max(0, round(score))),
            'metrics': metrics, 'findings': findings}


# --- 그림 -------------------------------------------------------------------
#
# opus 에게 "이렇게 움직였다" 를 글로만 설명하면 한계가 있다. 손이 그린 궤적을
# 세 방향에서 보여 주면 '팔이 몸을 스치며 지나간다', '왕복이 아니라 한 번만
# 갔다' 같은 것을 바로 짚는다. 라벨은 영문으로 둔다 - 이 서버에 한글 폰트가
# 없어서 한글을 쓰면 네모로 깨진다.

def render(moves, path, title=None):
    """궤적을 3면도 + 관절 곡선 그림으로 저장한다. 실패하면 None."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception:
        return None

    sim = simulate(moves)
    if not sim['ok'] or not sim['samples']:
        return None

    # 이 서버에는 한글 폰트가 없다. 제목에 한글을 넣으면 네모로 깨지고
    # 저장할 때마다 경고가 쏟아진다. 요청 내용은 본문으로 따로 가므로,
    # 그림 제목은 아스키만 남긴다.
    if title:
        title = ''.join(c if ord(c) < 128 else '.' for c in title).strip('. ')
    title = title or 'trajectory'

    S = sim['samples']
    ts = [s['t'] for s in S]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=11)

    views = [
        (axes[0][0], 0, 1, 'top view  x-y  (x=forward, y=left)', 'x [m]', 'y [m]'),
        (axes[0][1], 0, 2, 'side view  x-z  (z=up)', 'x [m]', 'z [m]'),
        (axes[1][0], 1, 2, 'front view  y-z', 'y [m]', 'z [m]'),
    ]
    colors = {'right_arm': 'tab:red', 'left_arm': 'tab:blue'}

    for ax, i, j, name, xl, yl in views:
        # 금지 구역을 옅게 깔아 둔다 - 어디를 피해야 하는지가 보여야 한다.
        if (i, j) == (0, 2):
            ax.add_patch(Rectangle(
                (-0.4, me.TORSO_BOX['z_min']), 0.4 + me.TORSO_BOX['x_max'], 1.0,
                color='0.85', zorder=0, label='keep-out (torso)'))
            ax.add_patch(Circle((me.HEAD_SPHERE_CENTER[0], me.HEAD_SPHERE_CENTER[2]),
                                me.HEAD_SPHERE_RADIUS, color='0.75', zorder=0))
        if (i, j) == (0, 1):
            ax.add_patch(Rectangle(
                (-0.4, -me.TORSO_BOX['y_abs_max']),
                0.4 + me.TORSO_BOX['x_max'], 2 * me.TORSO_BOX['y_abs_max'],
                color='0.85', zorder=0))

        for side, c in colors.items():
            pts = [s['hands'][side] for s in S]
            ax.plot([p[i] for p in pts], [p[j] for p in pts], '-',
                    color=c, lw=1.6, label=side.replace('_arm', ''))
            ax.plot(pts[0][i], pts[0][j], 'o', color=c, ms=7, mfc='white')
            ax.plot(pts[-1][i], pts[-1][j], 's', color=c, ms=6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, loc='best')

    ax = axes[1][1]
    changing = [j for j in JOINTS
                if max(abs(s['pose'].get(j, 0.0)) for s in S) > 1.0]
    for j in changing[:9]:
        ax.plot(ts, [s['pose'].get(j, 0.0) for s in S], lw=1.2,
                label=j.replace('_arm', '').replace('.hand', ''))
    ax.set_title('joint angles [deg] vs time [s]', fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, ncol=2, loc='best')

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def render_b64(moves, title=None):
    """그림을 base64 jpeg 로. 브로커 /motion 에 그대로 실어 보낼 수 있다."""
    import base64
    import tempfile

    fd, png = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        if render(moves, png, title=title) is None:
            return None
        try:
            from PIL import Image
            jpg = png.replace('.png', '.jpg')
            Image.open(png).convert('RGB').save(jpg, quality=72)
            data = open(jpg, 'rb').read()
            os.unlink(jpg)
        except Exception:
            data = open(png, 'rb').read()
        return base64.b64encode(data).decode('ascii')
    finally:
        try:
            os.unlink(png)
        except OSError:
            pass


def describe(result):
    """평가 결과를 opus 에게 보여 줄 짧은 글로."""
    if not result['ok']:
        return '실행 자체가 막혔습니다: %s' % result.get('error')
    m = result['metrics']
    lines = [
        '점수 %d/100, 길이 %.1f초, 키프레임 %d개'
        % (result['score'], m['duration_s'], m['keyframes']),
    ]
    for side, ko in (('right_arm', '오른손'), ('left_arm', '왼손')):
        lines.append('%s: 최대 %.0fcm 이동, 총 경로 %.0fcm, 방향전환 %d회, '
                     '금지구역 여유 %.1fcm'
                     % (ko, m[side]['reach_cm'], m[side]['path_cm'],
                        m[side]['reversals'], m[side]['min_clearance_cm']))
    if result['findings']:
        lines.append('지적된 문제:')
        lines += ['  - ' + f for f in result['findings']]
    else:
        lines.append('지적된 문제 없음.')
    return '\n'.join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset', help='motion_presets 의 이름')
    ap.add_argument('--all-presets', action='store_true')
    ap.add_argument('--json', help='moves 가 든 json 파일')
    ap.add_argument('--render', help='궤적 그림을 이 경로에 저장')
    args = ap.parse_args()

    def show(name, moves):
        r = evaluate(moves)
        print('── %s' % name)
        print(describe(r))
        print()
        if args.render:
            out = args.render if not args.all_presets else \
                args.render.replace('.png', '-%s.png' % name)
            if render(moves, out, title=name):
                print('  그림: %s' % out)

    if args.all_presets:
        from motion_presets import PRESETS
        for name in PRESETS:
            show(name, PRESETS[name]['moves'])
    elif args.preset:
        from motion_presets import PRESETS
        show(args.preset, PRESETS[args.preset]['moves'])
    elif args.json:
        with open(args.json, encoding='utf-8') as fh:
            data = json.load(fh)
        show(os.path.basename(args.json), data.get('moves', data))
    else:
        ap.error('--preset / --all-presets / --json 중 하나')


if __name__ == '__main__':
    main()
