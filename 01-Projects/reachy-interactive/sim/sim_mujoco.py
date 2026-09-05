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


MESH_DIR = os.path.join(HERE, 'meshes')


def _meshes():
    """링크별 CAD 메시가 뽑혀 있으면 그 목록. 없으면 빈 dict."""
    if not os.path.isdir(MESH_DIR):
        return {}
    return {f[:-4]: os.path.join(MESH_DIR, f)
            for f in os.listdir(MESH_DIR) if f.endswith('.obj')}


def build_mjcf(use_mesh=True, keepout=True):
    """motion_exec.CHAINS 를 그대로 MJCF 로. 모델과 검증기가 갈라지지 않게.

    CAD 메시(sim/extract_meshes.py 로 reachy.glb 에서 뽑은 것)가 있으면 그것을
    **보이는 용도로만** 붙인다. 충돌은 계속 캡슐로 본다 - 캡슐은 실측 반지름을
    쓰고 sim_safety 의 기하 계산과 일치가 확인된 것이라, 보기 좋자고 바꿀
    이유가 없다. 메시는 MuJoCo 에서 볼록 껍질로 취급되기도 해서 충돌용으로는
    오히려 부정확하다.
    """
    mesh = _meshes() if use_mesh else {}
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
           '    <material name="body" rgba=".70 .74 .80 %s"/>'
           % ('.30' if keepout else '0'),
           '    <material name="core" rgba=".55 .58 .64 1" specular=".3"/>',
           '    <material name="rightm" rgba=".85 .28 .25 1" specular=".4"/>',
           '    <material name="leftm"  rgba=".23 .45 .82 1" specular=".4"/>',
           '    <material name="ant"    rgba=".95 .78 .25 1"/>',
           '    <material name="cad"    rgba=".78 .80 .84 1" specular=".35" '
           'shininess=".4"/>',
           # 메시를 쓰면 충돌용 캡슐은 안 보이게 한다 (둘 다 그리면 겹쳐 보인다)
           '    <material name="hidden" rgba="0 0 0 0"/>']
    for name, path in sorted(mesh.items()):
        out.append('    <mesh name="m_%s" file="%s"/>' % (name, path))
    out += ['  </asset>',
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
           # 몸통·머리의 실제 모습. CAD 메시가 있으면 그것을, 없으면 대충의
           # 상자와 구를. 어느 쪽이든 충돌에는 넣지 않는다(금지 구역이 본다).
           ('    <geom name="core" type="mesh" mesh="m_world" material="cad" '
            'contype="0" conaffinity="0"/>' if 'world' in mesh else
            '    <geom name="core" type="box" material="core" contype="0" '
            'conaffinity="0" pos="-0.02 0 -0.22" size=".055 .085 .26"/>'),
           ('' if 'world' in mesh else
            '    <geom name="headcore" type="sphere" material="core" '
            'contype="0" conaffinity="0" pos="0 0 %.3f" size=".075"/>'
            % (me.HEAD_SPHERE_CENTER[2] + 0.02)),
           # 로봇 눈 카메라. 실물은 Orbita 목이 시선 (y, z at x=0.5) 로
           # 움직이므로, 팬(z축)과 틸트(y축) 관절 두 개로 같은 시선을 흉내낸다.
           # 위치는 실물 카메라 자리(목 위 머리 앞면, NECK_POS 기준).
           '    <body name="eye_pan" pos="%.3f 0 %.3f">' % (0.0, 0.09),
           '      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
           '      <joint name="eye.pan" axis="0 0 1" range="-90 90"/>',
           '      <body name="eye_tilt" pos="0.06 0 0.02">',
           '        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
           '        <joint name="eye.tilt" axis="0 1 0" range="-60 60"/>',
           # zaxis 가 카메라의 '뒤' 방향이다. -x 를 zaxis 로 주면 +x(정면)를 본다.
           '        <camera name="robot_eye" pos="0 0 0" zaxis="-1 0 0" fovy="58"/>',
           '      </body>',
           '    </body>',
           # 안테나는 사슬에 없다(머리 위 별도 모터). 감정 표현이 잘 보이므로
           # 힌지로 붙여 준다.
           '    <body name="left_antenna" pos="-0.02 0.06 %.3f">'
           % (me.HEAD_SPHERE_CENTER[2] + me.HEAD_SPHERE_RADIUS - 0.01),
           '      <inertial pos="0 0 .05" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
           '      <joint name="head.left_antenna" axis="0 1 0" range="-140 140"/>',
           ('      <geom type="mesh" mesh="m_left_antenna" material="cad" '
            'contype="0" conaffinity="0"/>' if 'left_antenna' in mesh else
            '      <geom type="capsule" material="ant" contype="0" '
            'conaffinity="0" fromto="0 0 0 0 0 .1" size=".008"/>'),
           '    </body>',
           '    <body name="right_antenna" pos="-0.02 -0.06 %.3f">'
           % (me.HEAD_SPHERE_CENTER[2] + me.HEAD_SPHERE_RADIUS - 0.01),
           '      <inertial pos="0 0 .05" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
           '      <joint name="head.right_antenna" axis="0 1 0" range="-140 140"/>',
           ('      <geom type="mesh" mesh="m_right_antenna" material="cad" '
            'contype="0" conaffinity="0"/>' if 'right_antenna' in mesh else
            '      <geom type="capsule" material="ant" contype="0" '
            'conaffinity="0" fromto="0 0 0 0 0 .1" size=".008"/>'),
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
            # CAD 메시는 보이는 용도로만. 충돌은 아래 캡슐이 계속 맡는다.
            bname = _body_name(jname)
            if bname in mesh:
                out.append('%s<geom type="mesh" mesh="m_%s" material="cad" '
                           'contype="0" conaffinity="0"/>' % (pad, bname))
            # 다음 링크까지 길이가 있으면 그 구간을 캡슐로 그린다
            nxt = chain[i + 1][1] if i + 1 < len(chain) else None
            if nxt and any(abs(v) > 1e-6 for v in nxt):
                r = RADII[order[min(seg, len(order) - 1)]]
                ct, ca = GROUP[side]
                out.append('%s<geom name="%s_%s" type="capsule" '
                           'material="%s" contype="%d" conaffinity="%d" '
                           'fromto="0 0 0 %.4f %.4f %.4f" size="%.4f"/>'
                           % (pad, side, order[min(seg, len(order) - 1)],
                              'hidden' if mesh else mat, ct, ca,
                              nxt[0], nxt[1], nxt[2], r))
                seg += 1
        # 마지막(손)은 작은 구로
        pad = '    ' + '  ' * depth
        ct, ca = GROUP[side]
        out.append('%s<geom name="%s_tip" type="sphere" material="%s" '
                   'contype="%d" conaffinity="%d" size="%.4f"/>'
                   % (pad, side, 'hidden' if mesh else mat, ct, ca,
                      RADII['hand']))
        while depth > 0:
            depth -= 1
            out.append('    ' + '  ' * depth + '</body>')

    out += ['  </worldbody>', '</mujoco>']
    return '\n'.join(out)


_USE_MESH = True
_KEEPOUT = True


def _load():
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(use_mesh=_USE_MESH, keepout=_KEEPOUT))
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


# 카메라 각도는 실측으로 정했다. 로봇이 +x 를 본다고 azimuth 0 이 정면일
# 거라 짐작했는데 반대였다 - 오른팔을 앞으로 뻗어 놓고 네 각도를 찍어 보니
# **azimuth 180 이 정면**(얼굴의 카메라 렌즈 두 개가 보인다)이고 0 은 등이다.
VIEWS = {'front': 180, 'side': 90, 'back': 0, 'other_side': 270}
DEFAULT_VIEWS = ('front', 'side')      # 평가에는 이 둘이면 충분하다


def render_video(moves_list, path, labels=None, fps=FPS, header=None,
                 width=WIDTH, height=HEIGHT, azimuth=None, elevation=-14,
                 distance=1.62, views=None):
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

    # azimuth 를 직접 주면 그 한 각도만, 아니면 정면+측면 두 화면.
    angles = ([azimuth] if azimuth is not None
              else [VIEWS[v] for v in (views or DEFAULT_VIEWS)])
    names = ([''] if azimuth is not None else list(views or DEFAULT_VIEWS))

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.elevation, cam.distance = elevation, distance
    # 로봇은 안테나 끝(z≈+0.35)부터 늘어뜨린 손끝(z≈-0.64)까지 걸친다.
    # 그 가운데를 봐야 위아래가 잘리지 않는다.
    cam.lookat[:] = [0.06, 0.0, -0.14]

    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for k in range(total):
            panes = []
            for seq in seqs:
                s = seq[min(k, len(seq) - 1)]
                _set_pose(data, idx, s['pose'])
                mujoco.mj_forward(model, data)
                for az in angles:
                    cam.azimuth = az
                    renderer.update_scene(data, camera=cam)
                    panes.append(renderer.render())
            img = np.concatenate(panes, axis=1) if len(panes) > 1 else panes[0]
            img = _label(img, labels, seqs, k, header, names)
            h, w = img.shape[:2]
            writer.append_data(img[:h - h % 2, :w - w % 2])
    finally:
        writer.close()
        renderer.close()
    return path, None


def _label(img, labels, seqs, k, header, view_names=None):
    """프레임 위에 글자를 얹는다. 한글 폰트가 없으므로 아스키만."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    views = view_names or ['']
    panes = len(seqs) * len(views)
    w = im.width // panes
    for i in range(panes):
        seq = seqs[i // len(views)]
        lab = labels[i // len(views)]
        view = views[i % len(views)]
        t = seq[min(k, len(seq) - 1)]['t']
        d.text((i * w + 12, 10), '%s  t=%.1fs' % (_ascii(lab), t),
               fill=(235, 235, 235))
        if view:
            d.text((i * w + 12, 26), view, fill=(150, 170, 200))
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
    ap.add_argument('--azimuth', type=float,
                    help='한 각도만 쓴다 (180=정면, 90=측면, 0=등). '
                         '기본은 정면+측면 두 화면')
    ap.add_argument('--views', default='front,side',
                    help='보여 줄 시점: front/side/back/other_side 중 쉼표로')
    ap.add_argument('--contacts', action='store_true',
                    help='영상 대신 접촉 검사만 한다')
    ap.add_argument('--dump-xml', action='store_true', help='MJCF 만 출력')
    ap.add_argument('--no-mesh', action='store_true',
                    help='CAD 메시 대신 캡슐 모형으로 그린다 (가볍다)')
    ap.add_argument('--no-keepout', action='store_true',
                    help='금지 구역(반투명 상자·구)을 그리지 않는다 - 발표용')
    args = ap.parse_args()

    global _USE_MESH, _KEEPOUT
    _USE_MESH = not args.no_mesh
    _KEEPOUT = not args.no_keepout
    if args.dump_xml:
        print(build_mjcf(use_mesh=_USE_MESH, keepout=_KEEPOUT))
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
                            views=[v.strip() for v in args.views.split(',')
                                   if v.strip() in VIEWS],
                            header='%d pts' % score['score'])
    if err:
        print('영상 실패: %s' % err)
        return
    print('영상: %s (%.2fMB)' % (out, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
