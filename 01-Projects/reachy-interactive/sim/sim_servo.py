"""로봇 눈으로 보면서 조준을 스스로 고친다 - visual servoing 을 시뮬로.

실물 로봇은 이미 '보고 고치기' 를 한다: object_vision.look_at_object 가
카메라에서 물체의 화면 오차를 재서 시선을 옮긴다. 여기서는 그 다음 단계 -
**팔을** 화면 오차로 움직이는 것 - 을 실물에 넣기 전에 시뮬로 검증한다.

구성:
  * 장면에 표적(공)을 놓는다.
  * 머리의 robot_eye 카메라로 렌더링해 표적을 색으로 찾는다 - 진짜 검출기
    (SSD)를 흉내내는 자리다. 실물에서는 object_vision 의 DNN 이 이 역할.
  * 화면 오차(표적이 화면 중앙/그리퍼 마커에서 얼마나 벗어났나)로
    관절을 조금씩 움직인다. 자코비안은 수치 미분으로 얻는다 - 실물과 같은
    forward_kinematics 를 쓰므로 실물에서도 같은 값이 나온다.
  * 매 스텝을 실행기의 안전 검사(금지 구역)로 거른다.

사용:
    python3 sim/sim_servo.py --target 0.30 -0.10 -0.15 -o servo.mp4
    python3 sim/sim_servo.py --target 0.28 -0.14 -0.20 --noise 2 -o servo.mp4
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
import sim_mujoco as sm                           # noqa: E402

# 서보에 쓰는 관절 - 오른팔의 위치 자유도 4개.
SERVO_JOINTS = ['right_arm.shoulder_pitch', 'right_arm.shoulder_roll',
                'right_arm.arm_yaw', 'right_arm.elbow_pitch']
EYE = (0.06, 0.0, 0.11)          # 카메라 위치(팬 축 기준 오프셋 포함)
STEP_DEG = 4.0                   # 한 스텝의 최대 관절 변화
DONE_CM = 3.0                    # 손끝-표적 거리가 이 안이면 성공


def look_at(pose, point):
    """시선을 그 점으로 - 실물 Orbita 기구학 그대로 (롤 포함).

    None 이면 실물 목이 그 방향을 못 본다는 뜻이다.
    """
    return sm.set_gaze(pose, point[1], point[2], x=max(point[0], 0.05))


def hand_of(pose):
    return me.forward_kinematics(me.CHAINS['right_arm'], pose)


def pixel_of(model, data, mujoco, cam_name, point, width, height):
    """3D 점이 robot_eye 화면의 어느 픽셀에 오는가."""
    import numpy as np

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cid]
    cam_mat = np.array(data.cam_xmat[cid]).reshape(3, 3)
    rel = cam_mat.T @ (np.array(point) - cam_pos)
    if rel[2] > -1e-6:           # 카메라 뒤
        return None
    fovy = math.radians(model.cam_fovy[cid])
    f = (height / 2.0) / math.tan(fovy / 2.0)
    u = width / 2.0 + f * rel[0] / -rel[2]
    v = height / 2.0 - f * rel[1] / -rel[2]
    return (u, v)


def in_bounds(pose):
    """이 자세가 실행기의 금지 구역 검사를 통과하나."""
    try:
        for side, chain in me.CHAINS.items():
            for p in me._arm_points(chain, pose):
                me._check_point(p, side)
        return True
    except me.ValidationError:
        return False


def servo(target, steps=60, noise_px=0.0, seed=0, record=None,
          width=640, height=480):
    """화면 오차만으로 손끝을 표적까지. (성공 여부, 궤적 기록)."""
    import numpy as np
    import mujoco

    rng = np.random.RandomState(seed)
    model = mujoco.MjModel.from_xml_string(sm.build_mjcf(keepout=False))
    data = mujoco.MjData(model)
    idx = sm._qpos_index(mujoco, model)

    # 표적은 모델에 없으므로 렌더링에만 얹는다 (아래 _draw_marker).
    pose = {j: 0.0 for j in idx}
    pose['right_arm.shoulder_pitch'] = -40      # 손이 화면에 들어오는 시작 자세
    pose['right_arm.elbow_pitch'] = -60

    frames = []
    history = []
    renderer = mujoco.Renderer(model, height=height, width=width) if record else None

    # 화면 (u,v) 두 축만으로는 깊이가 안 보인다. 처음 만든 판은 5스텝 만에
    # 화면 오차를 0 으로 만들고 26cm 뒤에서 멈췄다 - 손이 표적과 같은 시선상에
    # 있었을 뿐이다. 한 눈(단안)의 정직한 한계다.
    #
    # 실물에서도 쓸 수 있는 단서로 세 번째 축을 만든다: **화면에서의 크기**.
    # 같은 물체는 가까울수록 크게 찍힌다. 알려진 크기의 표적(공)과 그리퍼의
    # 화면 폭 비율이 곧 깊이 비율이다. 여기서는 그 비율이 주는 정보량만
    # 흉내내면 되므로, 카메라-표적 / 카메라-손 거리의 로그 차를 세 번째
    # 오차로 쓴다 (잡음도 같이 준다).
    SIZE_GAIN = 220.0        # 픽셀 축과 규모를 맞추는 계수

    def screen_error(p, measured=True):
        """(u오차, v오차, 크기오차). 시선은 표적을 따라간다.

        measured=False 는 자코비안 프로브용 - 잡음을 넣지 않는다. 유한차분을
        잡음 낀 측정으로 하면 1도 차이의 몇 픽셀에 2px 잡음이 얹혀 미분이
        엉망이 된다(실제로 성공률이 절반으로 떨어졌다). 실물도 마찬가지다:
        검출은 흔들려도 자기 팔의 기구학(FK)은 정확히 아니까, 화면 자코비안은
        모델에서 깨끗하게 뽑고 잡음은 오차 측정에만 있다.
        """
        look_at(p, target)
        sm._set_pose(data, idx, p)
        mujoco.mj_forward(model, data)
        tp = pixel_of(model, data, mujoco, 'robot_eye', target, width, height)
        hp = pixel_of(model, data, mujoco, 'robot_eye', hand_of(p), width, height)
        if tp is None or hp is None:
            return None
        cam = data.cam_xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot_eye')]
        dt_ = me._dist(tuple(cam), target)
        dh_ = me._dist(tuple(cam), hand_of(p))
        size = SIZE_GAIN * math.log(dh_ / max(dt_, 1e-6))
        n = rng.normal(0, noise_px, 5) if (noise_px and measured) else (0,) * 5
        return (tp[0] + n[0] - hp[0] - n[2],
                tp[1] + n[1] - hp[1] - n[3],
                size + n[4])

    ok = False
    for step in range(steps):
        err = screen_error(pose)
        dist = me._dist(hand_of(pose), target)
        history.append({'step': step, 'err_px': err, 'dist_cm': dist * 100})

        if record is not None and renderer is not None:
            renderer.update_scene(data, camera='robot_eye')
            eye = renderer.render().copy()
            _draw_marker(eye, pixel_of(model, data, mujoco, 'robot_eye',
                                       target, width, height), (240, 70, 60))
            _draw_marker(eye, pixel_of(model, data, mujoco, 'robot_eye',
                                       hand_of(pose), width, height),
                         (70, 200, 90))
            renderer.update_scene(data, camera=_third_person(mujoco))
            world = renderer.render().copy()
            _draw_dot3(world, model, data, mujoco, target, width, height)
            frames.append(np.concatenate([eye, world], axis=1))

        if dist * 100 <= DONE_CM:
            ok = True
            break
        if err is None:
            break

        # 화면 자코비안: 각 관절을 조금 움직이면 화면 오차가 얼마나 변하나.
        # 실물과 같은 FK 로 수치 미분 - 캘리브레이션 없이도 방향은 맞다.
        base = screen_error(pose, measured=False)
        if base is None:
            break
        J = []
        for j in SERVO_JOINTS:
            trial = dict(pose)
            trial[j] = pose[j] + 1.0
            e2 = screen_error(trial, measured=False)
            if e2 is None:
                J.append((0.0, 0.0, 0.0))
            else:
                J.append((e2[0] - base[0], e2[1] - base[1], e2[2] - base[2]))
        J = np.array(J).T                       # 3 x n
        # 최소제곱으로 관절 변화량. 감쇠를 넣어 특이점에서 날뛰지 않게.
        dq, *_ = np.linalg.lstsq(J.T @ J + 4.0 * np.eye(len(SERVO_JOINTS)),
                                 J.T @ (-np.array(err)), rcond=None)
        dq = np.clip(dq * 0.8, -STEP_DEG, STEP_DEG)

        trial = dict(pose)
        for j, d in zip(SERVO_JOINTS, dq):
            lo, hi = me.ALL_LIMITS[j]
            trial[j] = min(hi, max(lo, trial[j] + float(d)))
        # 안전: 실행기의 금지 구역을 침범하는 스텝은 버리고 반으로 줄여 본다.
        if in_bounds(trial):
            pose = trial
        else:
            half = dict(pose)
            for j, d in zip(SERVO_JOINTS, dq):
                lo, hi = me.ALL_LIMITS[j]
                half[j] = min(hi, max(lo, half[j] + float(d) * 0.4))
            if in_bounds(half):
                pose = half
            history[-1]['clipped'] = True

    if record and frames:
        import imageio.v2 as imageio
        w = imageio.get_writer(record, fps=10, macro_block_size=1)
        for f in frames:
            h2, w2 = f.shape[:2]
            w.append_data(f[:h2 - h2 % 2, :w2 - w2 % 2])
        w.close()
    if renderer is not None:
        renderer.close()
    return ok, history


def _third_person(mujoco):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 150, -18, 1.5
    cam.lookat[:] = [0.15, -0.05, -0.15]
    return cam


def _draw_marker(img, px, color, r=10):
    if px is None:
        return
    import numpy as np
    h, w = img.shape[:2]
    u, v = int(px[0]), int(px[1])
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            d2 = du * du + dv * dv
            if (r - 2) ** 2 <= d2 <= r * r and 0 <= v + dv < h and 0 <= u + du < w:
                img[v + dv, u + du] = color


def _draw_dot3(img, model, data, mujoco, point, width, height):
    # 3자유 시점에는 픽셀 투영 없이 근사 표시만 - 표적 위치의 눈짐작용.
    pass


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', nargs=3, type=float, default=[0.30, -0.10, -0.15],
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--noise', type=float, default=0.0,
                    help='검출 잡음 (픽셀 표준편차)')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('-o', '--out', help='과정을 영상으로 (로봇 눈 + 제3자)')
    args = ap.parse_args()

    ok, hist = servo(tuple(args.target), steps=args.steps,
                     noise_px=args.noise, record=args.out)
    first, last = hist[0], hist[-1]
    print('표적 (%.2f %.2f %.2f)%s' % (*args.target,
          '  잡음 %.0fpx' % args.noise if args.noise else ''))
    print('  %d스텝: 손끝-표적 %.1fcm -> %.1fcm  [%s]'
          % (len(hist), first['dist_cm'], last['dist_cm'],
             '도달' if ok else '미도달'))
    clipped = sum(1 for h in hist if h.get('clipped'))
    if clipped:
        print('  금지 구역에 막혀 줄인 스텝 %d개' % clipped)
    if args.out:
        print('  영상: %s' % args.out)


if __name__ == '__main__':
    main()
