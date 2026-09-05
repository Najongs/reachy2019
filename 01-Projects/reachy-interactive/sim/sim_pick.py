"""컵을 보고, 집고, 쟁반에 놓는다 - pick & place 전 과정을 시뮬로.

장면은 실측 그대로다(motion_prompt.txt): 테이블 면 z=-0.27, 컵은 몸 앞
(0.31, -0.14), 쟁반은 (0.30, +0.14). 단계마다 visual servoing(sim_servo 와
같은 방식: 로봇 눈 화면 오차 + 크기 단서, 모델 자코비안)으로 손을 옮긴다.

    상공 접근 -> 하강 -> 집기(거리 확인) -> 쟁반 상공으로 -> 놓기

집기는 물리가 아니라 판정이다: '집기' 시점에 손끝이 컵에서 GRIP_CM 안에
있으면 성공으로 보고, 이후 컵이 손끝을 따라간다. 실물에서는 힘센서가 이
판정을 한다(hand.close() 가 잡혔는지 돌려준다). 놓으면 컵은 바로 아래
표면(쟁반/테이블) 위로 내려앉는다 - 낙하 물리는 흉내내지 않는다.

사용:
    python3 sim/sim_pick.py -o pick.mp4
    python3 sim/sim_pick.py --cup 0.28 -0.10 --noise 3 -o pick.mp4
"""

import math
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                          # noqa: E402
import sim_mujoco as sm                           # noqa: E402
import sim_servo as sv                            # noqa: E402

TABLE_TOP = -0.27
CUP_H = 0.09
TRAY = (0.30, 0.14, -0.26)       # 쟁반 윗면
GRIP_CM = 7.0                    # 손끝-컵 중심이 이 안이면 잡힌 것으로 본다
                                 # (손 반지름 4.4cm + 컵 반지름 3cm)


class PickWorld(object):
    """장면 + 로봇. 컵을 옮기고, 눈/제3자 프레임을 모은다."""

    def __init__(self, cup_xy=(0.31, -0.14), width=640, height=480):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(
            sm.build_mjcf(keepout=False, scene=True))
        self.data = mujoco.MjData(self.model)
        self.idx = sm._qpos_index(mujoco, self.model)
        self.cup_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                         'cup')
        self.model.geom_pos[self.cup_gid][:2] = cup_xy
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.frames = []
        self.held = False
        self._grip_offset = None

        self.pose = {j: 0.0 for j in self.idx}
        self.pose['right_arm.shoulder_pitch'] = -40
        self.pose['right_arm.elbow_pitch'] = -60

    @property
    def cup(self):
        return tuple(float(v) for v in self.model.geom_pos[self.cup_gid])

    def hand(self):
        return me.forward_kinematics(me.CHAINS['right_arm'], self.pose)

    def sync(self, look=None):
        """자세 반영 + (집혔으면) 컵이 손을 따라가게 + 시선."""
        if look is not None:
            sv.look_at(self.pose, look)
        if self.held:
            h = self.hand()
            self.model.geom_pos[self.cup_gid] = [
                h[i] + self._grip_offset[i] for i in range(3)]
        sm._set_pose(self.data, self.idx, self.pose)
        self.mujoco.mj_forward(self.model, self.data)

    def grip(self):
        """집기 판정. 실물에서는 hand.close() 의 힘센서 판정이 이 자리다."""
        d = me._dist(self.hand(), self.cup) * 100
        if d <= GRIP_CM:
            self.held = True
            h = self.hand()
            self._grip_offset = tuple(self.cup[i] - h[i] for i in range(3))
        return d

    def release(self):
        """놓기. 컵은 바로 아래 표면 위로 내려앉는다."""
        self.held = False
        x, y, _ = self.cup
        on_tray = (abs(x - TRAY[0]) < 0.08 and abs(y - TRAY[1]) < 0.07)
        surface = TRAY[2] if on_tray else TABLE_TOP
        self.model.geom_pos[self.cup_gid][2] = surface + CUP_H / 2.0
        return on_tray

    def snap(self, note='', look=None):
        import numpy as np

        self.sync(look=look)
        cam = self.renderer
        cam.update_scene(self.data, camera='robot_eye')
        eye = cam.render().copy()
        sv._draw_marker(eye, sv.pixel_of(self.model, self.data, self.mujoco,
                                         'robot_eye', self.cup,
                                         self.width, self.height), (240, 70, 60))
        sv._draw_marker(eye, sv.pixel_of(self.model, self.data, self.mujoco,
                                         'robot_eye', self.hand(),
                                         self.width, self.height), (70, 200, 90))
        cam.update_scene(self.data, camera=sv._third_person(self.mujoco))
        world = cam.render().copy()
        frame = np.concatenate([eye, world], axis=1)
        if note:
            try:
                from PIL import Image, ImageDraw
                im = Image.fromarray(frame)
                ImageDraw.Draw(im).text((12, 10), note, fill=(235, 235, 235))
                frame = np.asarray(im)
            except Exception:
                pass
        self.frames.append(frame)

    # -- 서보 한 구간 -------------------------------------------------------

    def servo_to(self, target, label, steps=40, done_cm=3.0, noise_px=2.0,
                 seed=0):
        """화면 오차로 손을 target 까지. sim_servo 와 같은 방식."""
        import numpy as np

        rng = np.random.RandomState(seed)
        SIZE_GAIN = 220.0

        def err_of(p, measured=True):
            trial_world = dict(self.pose)
            self.pose.update(p)
            sv.look_at(self.pose, target)
            if self.held:
                self.sync()
            else:
                sm._set_pose(self.data, self.idx, self.pose)
                self.mujoco.mj_forward(self.model, self.data)
            tp = sv.pixel_of(self.model, self.data, self.mujoco, 'robot_eye',
                             target, self.width, self.height)
            hand = self.hand()          # 프로브 자세의 손 - 복원 전에 읽는다
            hp = sv.pixel_of(self.model, self.data, self.mujoco, 'robot_eye',
                             hand, self.width, self.height)
            cid = self.mujoco.mj_name2id(self.model,
                                         self.mujoco.mjtObj.mjOBJ_CAMERA,
                                         'robot_eye')
            cam = tuple(float(v) for v in self.data.cam_xpos[cid])
            # data 도 원래 자세로 되돌린다 (다음 프로브가 이 위에 안 얹히게).
            self.pose = trial_world
            sm._set_pose(self.data, self.idx, self.pose)
            self.mujoco.mj_forward(self.model, self.data)
            if tp is None or hp is None:
                return None
            size = SIZE_GAIN * math.log(
                me._dist(cam, hand) / max(me._dist(cam, target), 1e-6))
            n = rng.normal(0, noise_px, 5) if (noise_px and measured) else (0,) * 5
            return (tp[0] + n[0] - hp[0] - n[2], tp[1] + n[1] - hp[1] - n[3],
                    size + n[4])

        for step in range(steps):
            d = me._dist(self.hand(), target) * 100
            self.snap('%s  %.1fcm' % (label, d), look=target)
            if d <= done_cm:
                return True
            err = err_of({})
            base = err_of({}, measured=False)
            if err is None or base is None:
                return False
            J = []
            for j in sv.SERVO_JOINTS:
                e2 = err_of({j: self.pose[j] + 1.0}, measured=False)
                J.append((0.0, 0.0, 0.0) if e2 is None else
                         tuple(e2[k] - base[k] for k in range(3)))
            J = np.array(J).T
            dq, *_ = np.linalg.lstsq(J.T @ J + 4.0 * np.eye(len(sv.SERVO_JOINTS)),
                                     J.T @ (-np.array(err)), rcond=None)
            dq = np.clip(dq * 0.8, -sv.STEP_DEG, sv.STEP_DEG)
            trial = dict(self.pose)
            for j, dv in zip(sv.SERVO_JOINTS, dq):
                lo, hi = me.ALL_LIMITS[j]
                trial[j] = min(hi, max(lo, trial[j] + float(dv)))
            if sv.in_bounds(trial):
                self.pose = trial
            else:
                for j, dv in zip(sv.SERVO_JOINTS, dq):
                    lo, hi = me.ALL_LIMITS[j]
                    self.pose[j] = min(hi, max(lo, self.pose[j] + float(dv) * 0.4))
        return me._dist(self.hand(), target) * 100 <= done_cm

    def hold(self, label, n=6):
        for _ in range(n):
            self.snap(label, look=self.cup)


def run(cup_xy=(0.31, -0.14), noise_px=2.0, record=None):
    w = PickWorld(cup_xy=cup_xy)
    log = []

    cup = w.cup
    above_cup = (cup[0], cup[1], cup[2] + 0.10)
    grasp_at = (cup[0], cup[1], cup[2] + 0.02)
    above_tray = (TRAY[0], TRAY[1] - 0.02, TRAY[2] + CUP_H / 2 + 0.10)
    place_at = (TRAY[0], TRAY[1] - 0.02, TRAY[2] + CUP_H / 2 + 0.03)

    ok1 = w.servo_to(above_cup, 'approach', noise_px=noise_px, seed=1)
    log.append(('상공 접근', ok1, me._dist(w.hand(), above_cup) * 100))
    ok2 = w.servo_to(grasp_at, 'descend', done_cm=2.5, noise_px=noise_px, seed=2)
    log.append(('하강', ok2, me._dist(w.hand(), grasp_at) * 100))

    d = w.grip()
    log.append(('집기 (손끝-컵 %.1fcm, 기준 %.0fcm)' % (d, GRIP_CM),
                w.held, d))
    w.hold('grip!' if w.held else 'grip FAILED')

    placed = False
    if w.held:
        ok3 = w.servo_to(above_tray, 'carry', done_cm=3.5,
                         noise_px=noise_px, seed=3)
        log.append(('쟁반 상공으로', ok3, me._dist(w.hand(), above_tray) * 100))
        ok4 = w.servo_to(place_at, 'place', done_cm=3.0,
                         noise_px=noise_px, seed=4)
        log.append(('내려놓을 위치', ok4, me._dist(w.hand(), place_at) * 100))
        placed = w.release()
        log.append(('놓기 -> %s' % ('쟁반 위' if placed else '쟁반 밖'),
                    placed, 0.0))
        w.hold('released' if placed else 'missed tray')

    if record and w.frames:
        import imageio.v2 as imageio

        wr = imageio.get_writer(record, fps=10, macro_block_size=1)
        for f in w.frames:
            h, ww = f.shape[:2]
            wr.append_data(f[:h - h % 2, :ww - ww % 2])
        wr.close()
    w.renderer.close()
    return placed, log


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cup', nargs=2, type=float, default=[0.31, -0.14],
                    metavar=('X', 'Y'))
    ap.add_argument('--noise', type=float, default=2.0)
    ap.add_argument('-o', '--out')
    args = ap.parse_args()

    placed, log = run(tuple(args.cup), noise_px=args.noise, record=args.out)
    print('컵 (%.2f, %.2f), 검출 잡음 %.0fpx' % (*args.cup, args.noise))
    for name, ok, dist in log:
        print('  %-34s %s%s' % (name, '성공' if ok else '실패',
                                ('  (%.1fcm)' % dist) if dist else ''))
    print('결과: %s' % ('컵이 쟁반 위에 놓였다' if placed else '실패'))
    if args.out:
        print('영상: %s' % args.out)


if __name__ == '__main__':
    main()
