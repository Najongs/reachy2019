"""MuJoCo 로 로봇을 실제로 그린다 - 브라우저 없이, 헤드리스로.

sim_viewer.html 은 reachy.glb 를 three.js 로 띄우지만 브라우저가 필요하다.
서버에는 브라우저가 없다. MuJoCo 는 EGL 로 화면 없이 렌더링하므로 같은 그림을
파일로 뽑을 수 있다.

모델은 손으로 쓰지 않고 **motion_exec.CHAINS 에서 만들어 낸다**. 그 사슬이
실물의 기구학이고 검증기가 쓰는 것과 같은 값이므로, 따로 URDF 를 쓰면 둘이
어긋날 수 있다. 각 링크의 이동(pos)과 회전축(axis), 한계(range)를 그대로
옮기고, 굵기는 sim/measure_arm.py 로 reachy.glb 에서 잰 값을 쓴다.

몸통과 머리는 검증기가 쓰는 금지 구역(상자·구) 그대로 그린다. 실제 외형보다
조금 크지만, 로봇이 무엇을 피하며 움직이는지가 보이는 편이 낫다.

덤: MuJoCo 가 접촉을 직접 계산하므로, 기하 계산과 별개로 '정말 닿는가' 를
물어볼 수 있다.

사용:
    python3 sim/sim_mujoco.py --preset wave -o wave.mp4
    python3 sim/sim_mujoco.py --candidate 3 -o out.mp4
    python3 sim/sim_mujoco.py --json final.json --compare draft.json -o ab.mp4
    python3 sim/sim_mujoco.py --preset wave --contacts     # 접촉 검사만
"""

import json
import math
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                          # noqa: E402
import sim_eval                                   # noqa: E402

try:
    import measure_arm
    RADII, _ = measure_arm.measure()
except Exception:
    RADII = {'upper_arm': 0.044, 'forearm': 0.037, 'hand': 0.041}

FPS = 25
WIDTH, HEIGHT = 640, 640


def _body_name(joint):
    return joint.replace('.', '_')


def build_mjcf():
    """motion_exec.CHAINS 를 그대로 MJCF 로. 모델과 검증기가 갈라지지 않게."""
    box = me.TORSO_BOX
    hx = (box['x_max'] + 0.25) / 2.0        # 뒤쪽으로 조금 두께를 준다
    cx = box['x_max'] - hx
    hz = (0.20 - box['z_min']) / 2.0
    cz = box['z_min'] + hz

    out = ['<mujoco model="reachy2019">',
           '  <compiler angle="degree"/>',
           '  <option gravity="0 0 0"/>',
           '  <visual><headlight ambient=".45 .45 .45" diffuse=".7 .7 .7"/>',
           '    <map znear=".01"/><quality shadowsize="2048"/>',
           # 오프스크린 버퍼 기본값이 640x480 이라 그보다 큰 프레임을 요청하면
           # 렌더러 생성 자체가 실패한다.
           '    <global offwidth="1600" offheight="1200"/></visual>',
           '  <asset>',
           '    <texture type="skybox" builtin="gradient" rgb1=".16 .19 .24"',
           '             rgb2=".07 .08 .10" width="256" height="256"/>',
           # 몸통·머리는 실제 외형이 아니라 검증기의 금지 구역이다(뒤로 25cm
           # 나 뻗은 큰 상자). 불투명하게 그리면 팔이 통째로 가려지므로
           # 반투명으로 두어 '피해야 할 부피' 로 보이게 한다.
           '    <material name="body" rgba=".70 .74 .80 .30"/>',
           '    <material name="core" rgba=".55 .58 .64 1" specular=".3"/>',
           '    <material name="rightm" rgba=".85 .28 .25 1" specular=".4"/>',
           '    <material name="leftm"  rgba=".23 .45 .82 1" specular=".4"/>',
           '    <material name="ant"    rgba=".95 .78 .25 1"/>',
           '  </asset>',
           '  <worldbody>',
           '    <light pos="0.8 0.6 1.2" dir="-.5 -.4 -1" directional="true"/>',
           '    <light pos="-0.6 -0.8 0.9" dir=".4 .5 -1" diffuse=".3 .3 .3"/>',
           '    <geom name="torso_keepout" type="box" material="body"',
           '          contype="4" conaffinity="3"',
           '          pos="%.4f 0 %.4f" size="%.4f %.4f %.4f"/>'
           % (cx, cz, hx, box['y_abs_max'], hz),
           '    <geom name="head_keepout" type="sphere" material="body"',
           '          contype="4" conaffinity="3"',
           '          pos="%.4f %.4f %.4f" size="%.4f"/>'
           % (me.HEAD_SPHERE_CENTER + (me.HEAD_SPHERE_RADIUS,)),
           # 안쪽에 실제 몸통에 가까운 크기의 불투명한 심을 둔다. 로봇이
           # 어디 있는지 눈으로 잡히게 하는 용도라 충돌에는 넣지 않는다.
           '    <geom name="core" type="box" material="core" contype="0" '
           'conaffinity="0" pos="-0.02 0 -0.22" size=".055 .085 .26"/>',
           '    <geom name="headcore" type="sphere" material="core" contype="0" '
           'conaffinity="0" pos="0 0 %.3f" size=".075"/>'
           % (me.HEAD_SPHERE_CENTER[2] + 0.02),
           # 안테나는 사슬에 없다(머리 위 별도 모터). 감정 표현이 잘 보이므로
           # 힌지로 붙여 준다.
           '    <body name="left_antenna" pos="-0.02 0.06 %.3f">'
           % (me.HEAD_SPHERE_CENTER[2] + me.HEAD_SPHERE_RADIUS - 0.01),
           '      <inertial pos="0 0 .05" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
           '      <joint name="head.left_antenna" axis="0 1 0" range="-140 140"/>',
           '      <geom type="capsule" material="ant" contype="0" conaffinity="0" '
           'fromto="0 0 0 0 0 .1" size=".008"/>',
           '    </body>',
           '    <body name="right_antenna" pos="-0.02 -0.06 %.3f">'
           % (me.HEAD_SPHERE_CENTER[2] + me.HEAD_SPHERE_RADIUS - 0.01),
           '      <inertial pos="0 0 .05" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
           '      <joint name="head.right_antenna" axis="0 1 0" range="-140 140"/>',
           '      <geom type="capsule" material="ant" contype="0" conaffinity="0" '
           'fromto="0 0 0 0 0 .1" size=".008"/>',
           '    </body>']

    # 충돌 그룹. 같은 팔의 이웃 링크는 팔꿈치에서 늘 겹치므로(캡슐 끝이
    # 만난다) 서로 보지 않게 한다. 의미 있는 것은 '반대 팔' 과 '금지 구역'
    # 뿐이다.  right=1, left=2, keepout=4
    GROUP = {'right_arm': (1, 6), 'left_arm': (2, 5)}
    order = ['upper_arm', 'forearm', 'hand']
    for side, chain in me.CHAINS.items():
        mat = 'rightm' if side == 'right_arm' else 'leftm'
        depth = 0
        seg = 0
        for i, (jname, trans, axis, lim) in enumerate(chain):
            pad = '    ' + '  ' * depth
            out.append('%s<body name="%s" pos="%.4f %.4f %.4f">'
                       % (pad, _body_name(jname), trans[0], trans[1], trans[2]))
            depth += 1
            pad = '    ' + '  ' * depth
            # 관절이 달린 바디는 질량이 있어야 한다(mjMINVAL). 여기서는
            # 순기구학만 쓰고 중력도 0 이라 값 자체는 무의미하다.
            out.append('%s<inertial pos="0 0 0" mass="0.05" '
                       'diaginertia="1e-4 1e-4 1e-4"/>' % pad)
            if any(axis):                       # 회전축이 있으면 관절
                out.append('%s<joint name="%s" axis="%d %d %d" range="%g %g"/>'
                           % (pad, jname, axis[0], axis[1], axis[2],
                              lim[0], lim[1]))
            # 다음 링크까지 길이가 있으면 그 구간을 캡슐로 그린다
            nxt = chain[i + 1][1] if i + 1 < len(chain) else None
            if nxt and any(abs(v) > 1e-6 for v in nxt):
                r = RADII[order[min(seg, len(order) - 1)]]
                ct, ca = GROUP[side]
                out.append('%s<geom name="%s_%s" type="capsule" '
                           'material="%s" contype="%d" conaffinity="%d" '
                           'fromto="0 0 0 %.4f %.4f %.4f" size="%.4f"/>'
                           % (pad, side, order[min(seg, len(order) - 1)],
                              mat, ct, ca, nxt[0], nxt[1], nxt[2], r))
                seg += 1
        # 마지막(손)은 작은 구로
        pad = '    ' + '  ' * depth
        ct, ca = GROUP[side]
        out.append('%s<geom name="%s_tip" type="sphere" material="%s" '
                   'contype="%d" conaffinity="%d" size="%.4f"/>'
                   % (pad, side, mat, ct, ca, RADII['hand']))
        while depth > 0:
            depth -= 1
            out.append('    ' + '  ' * depth + '</body>')

    out += ['  </worldbody>', '</mujoco>']
    return '\n'.join(out)


def _load():
    import mujoco
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    return mujoco, model


def _set_pose(data, idx, pose):
    """각도(도)를 qpos 에 넣는다.

    **qpos 는 항상 라디안이다.** MJCF 의 compiler angle="degree" 는 XML 안의
    숫자를 읽을 때만 적용되고 런타임 값에는 관계없다. 이걸 놓쳐서 40도를
    40라디안(=132도)으로 돌렸고, 팔이 엉뚱한 데로 갔다.
    """
    for joint, adr in idx.items():
        data.qpos[adr] = math.radians(pose.get(joint, 0.0))


def _qpos_index(mujoco, model):
    """관절 이름 -> qpos 위치."""
    out = {}
    for j in sim_eval.JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0:
            out[j] = model.jnt_qposadr[jid]
    return out


def frames_of(moves, fps=FPS):
    sim = sim_eval.simulate(moves, hz=fps, check_collision=False)
    if not sim['ok']:
        return None, sim['error']
    return sim['samples'], None


# 로봇은 +x 를 바라본다. MuJoCo 카메라 azimuth 0 이 +x 쪽에서 보는 각도라,
# 사람이 로봇 앞에 서서 보는 그림이 된다. 살짝 틀어 3/4 로 본다.
def render_video(moves_list, path, labels=None, fps=FPS, header=None,
                 width=WIDTH, height=HEIGHT, azimuth=45, elevation=-10,
                 distance=1.7):
    """동작을 MuJoCo 로 렌더링해 mp4 로. 여러 개를 주면 가로로 붙인다."""
    import imageio.v2 as imageio
    import numpy as np

    mujoco, model = _load()
    data = mujoco.MjData(model)
    idx = _qpos_index(mujoco, model)

    seqs = []
    for mv in moves_list:
        fr, err = frames_of(mv, fps)
        if fr is None:
            return None, err
        seqs.append(fr)
    total = max(len(s) for s in seqs)
    labels = labels or [''] * len(seqs)

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = [0.05, 0.0, -0.22]

    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for k in range(total):
            panes = []
            for seq in seqs:
                s = seq[min(k, len(seq) - 1)]
                _set_pose(data, idx, s['pose'])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                panes.append(renderer.render())
            img = np.concatenate(panes, axis=1) if len(panes) > 1 else panes[0]
            img = _label(img, labels, seqs, k, header)
            h, w = img.shape[:2]
            writer.append_data(img[:h - h % 2, :w - w % 2])
    finally:
        writer.close()
        renderer.close()
    return path, None


def _label(img, labels, seqs, k, header):
    """프레임 위에 글자를 얹는다. 한글 폰트가 없으므로 아스키만."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    n = len(seqs)
    w = im.width // n
    for i, (lab, seq) in enumerate(zip(labels, seqs)):
        t = seq[min(k, len(seq) - 1)]['t']
        d.text((i * w + 12, 10), '%s  t=%.1fs' % (_ascii(lab), t),
               fill=(235, 235, 235))
    if header:
        d.text((12, im.height - 20), _ascii(header), fill=(190, 200, 215))
    import numpy as np
    return np.asarray(im)


def contacts(moves, fps=60):
    """MuJoCo 로 실제 접촉을 센다. 기하 계산과 별개의 확인이다."""
    mujoco, model = _load()
    data = mujoco.MjData(model)
    idx = _qpos_index(mujoco, model)
    samples, err = frames_of(moves, fps)
    if samples is None:
        return None, err

    hits = {}
    for s in samples:
        _set_pose(data, idx, s['pose'])
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            key = tuple(sorted([g1 or '?', g2 or '?']))
            prev = hits.get(key)
            if prev is None or con.dist < prev[0]:
                hits[key] = (con.dist, s['t'])
    return hits, None


def _ascii(text):
    out = ''.join(c if ord(c) < 128 else '' for c in (text or ''))
    return ' '.join(out.split())


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset')
    ap.add_argument('--candidate', type=int)
    ap.add_argument('--json')
    ap.add_argument('--compare')
    ap.add_argument('-o', '--out', default='motion.mp4')
    ap.add_argument('--fps', type=int, default=FPS)
    ap.add_argument('--size', type=int, default=WIDTH)
    ap.add_argument('--azimuth', type=float, default=45,
                    help='카메라 각도. 0=정면, 45=3/4(기본), 90=로봇의 왼쪽')
    ap.add_argument('--contacts', action='store_true',
                    help='영상 대신 접촉 검사만 한다')
    ap.add_argument('--dump-xml', action='store_true', help='MJCF 만 출력')
    args = ap.parse_args()

    if args.dump_xml:
        print(build_mjcf())
        return

    def load(spec):
        with open(spec, encoding='utf-8') as fh:
            d = json.load(fh)
        return d.get('moves', d), os.path.basename(spec)

    if args.preset:
        from motion_presets import PRESETS
        moves, name = PRESETS[args.preset]['moves'], args.preset
    elif args.candidate is not None:
        with open(os.path.join(HERE, '..', 'config',
                               'motion_candidates.json'), encoding='utf-8') as fh:
            c = json.load(fh)[args.candidate]
        moves, name = c['moves'], c.get('idea', 'cand')[:30]
    elif args.json:
        moves, name = load(args.json)
    else:
        ap.error('--preset / --candidate / --json 중 하나')

    if args.contacts:
        hits, err = contacts(moves)
        if err:
            print('실패: %s' % err)
            return
        if not hits:
            print('접촉 없음 (MuJoCo 기준)')
        else:
            print('접촉/근접 %d쌍:' % len(hits))
            for (a, b), (dist, t) in sorted(hits.items(), key=lambda x: x[1][0]):
                print('  %-22s ~ %-22s  간격 %+.1fmm  (%.1f초)'
                      % (a, b, dist * 1000, t))
        return

    lst, labs = [moves], [name]
    if args.compare:
        other, oname = load(args.compare)
        lst, labs = [other, moves], [oname, name]

    score = sim_eval.evaluate(moves)
    out, err = render_video(lst, args.out, labels=labs, fps=args.fps,
                            width=args.size, height=args.size,
                            azimuth=args.azimuth,
                            header='%d pts' % score['score'])
    if err:
        print('영상 실패: %s' % err)
        return
    print('영상: %s (%.2fMB)' % (out, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
