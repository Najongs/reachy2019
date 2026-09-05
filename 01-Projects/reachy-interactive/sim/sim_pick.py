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
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')   # GPU0 만 쓴다 (ollama 와 동거)

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
        self.clearances = []        # 프레임마다 (clearance cm, 부위쌍)
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

    def search(self, label='look for the cup', sweep=(-40, 40), steps=9):
        """팔은 그대로 두고 시선만 좌우로 훑어 컵을 화면에 담는다.

        실물 순서 그대로다: 먼저 눈(목)으로 물체를 찾고 시선을 두고, 그 다음에
        팔이 움직인다. 바로 팔부터 뻗으면 아직 어디 있는지 모르는 것을 향해
        움직이는 셈이다. 시야각(fovy 58도) 안에 컵이 들어오면 거기서 멈춘다.
        """
        cup = self.cup
        for k in range(steps):
            # 시선을 좌우로 쓸며 훑는다. 실물은 이때 SSD 가 매 프레임 돈다.
            # 책상 위 물건은 눈보다 훨씬 아래(z≈-0.22)에 있으므로, 훑는 시선도
            # 책상 높이를 향하게 한다(z 방향을 낮춰). 여기가 실측이라 값이
            # 아니라 '책상 표면을 겨눈다' 로 두는 게 안전하다.
            frac = k / (steps - 1)
            pan = sweep[0] + (sweep[1] - sweep[0]) * frac
            y = 0.5 * math.tan(math.radians(pan))
            sm.set_gaze(self.pose, y, TABLE_TOP - 0.08)   # 책상보다 더 아래로 겨눔
            sm._set_pose(self.data, self.idx, self.pose)
            self.mujoco.mj_forward(self.model, self.data)
            self.snap('%s' % label)
            px = sv.pixel_of(self.model, self.data, self.mujoco, 'robot_eye',
                             cup, self.width, self.height)
            # 화면 안(가장자리 5% 여백)에 컵이 들어오면 찾은 것.
            if px is not None and 0.05 * self.width < px[0] < 0.95 * self.width \
                    and 0.05 * self.height < px[1] < 0.95 * self.height:
                self.hold('found it', n=4)       # 시선을 컵에 고정하고 응시
                return True
        return False

    def _arm_gids(self):
        if getattr(self, '_agids', None) is None:
            mj = self.mujoco
            self._agids = []
            for i in range(self.model.ngeom):
                n = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
                if n and n.startswith('right_arm'):
                    self._agids.append((i, n))
        return self._agids

    def clearance(self, exclude_held=True):
        """팔의 모든 지오메트리와 장면(컵·테이블) 사이 최소 표면거리(cm).

        손끝 한 점이 아니라 팔 전체를 본다 - 팔뚝이나 손 옆면이 컵을 스치는
        것도 잡아야 하기 때문이다(그게 못 잡던 부분이다). MuJoCo 의 정확한
        지오메트리간 거리(mj_geomDistance)를 쓴다. 음수면 파고든 것.

        컵을 잡고 있으면(exclude_held) 손과 컵의 접촉은 정상이므로 뺀다.
        """
        mj = self.mujoco
        cup = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'cup')
        table = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'table')
        worst = (9.0, None)
        for gid, name in self._arm_gids():
            for oid, oname in ((cup, 'cup'), (table, 'table')):
                if oid < 0:
                    continue
                if self.held and exclude_held and oname == 'cup':
                    continue
                d = mj.mj_geomDistance(self.model, self.data, gid, oid, 1.0, None)
                if d < worst[0]:
                    worst = (d * 100, '%s~%s' % (name.replace('right_arm_', ''),
                                                 oname))
        return worst

    def penetrating(self, margin_cm=0.0):
        """팔 어느 부위든 컵/테이블을 파고들었으면 그 쌍 이름, 아니면 None.

        margin_cm 을 주면 '이만큼 가까워도 위험' 으로 본다.
        """
        d, pair = self.clearance()
        return pair if d < margin_cm else None

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
        self.clearances.append(self.clearance())    # 이 프레임의 팔<->장면 여유
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
        # 이 프레임에서 팔이 장면을 파고들면 테두리를 붉게 - 프레임을 넘겨
        # 보다가 바로 눈에 띄게.
        if self.clearances and self.clearances[-1][0] < 0:
            for img in (eye, world):
                img[:6, :] = [200, 40, 40]; img[-6:, :] = [200, 40, 40]
                img[:, :6] = [200, 40, 40]; img[:, -6:] = [200, 40, 40]
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
                 seed=0, stop_gap=None):
        """화면 오차로 손을 target 까지. sim_servo 와 같은 방식.

        stop_gap(cm) 을 주면 팔이 컵/테이블에 그만큼 가까워지는 순간 멈춘다 -
        위에서 하강할 때 손이 컵을 감싸는 높이가 여기다. 그 이상 내려가면
        파고든다. clearance() 로 팔 전체를 보므로 손끝뿐 아니라 손 옆면·
        팔뚝이 스치는 것도 잡는다.
        """
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
            if stop_gap is not None:
                gap, _ = self.clearance()
                if gap <= stop_gap:
                    return True                 # 이미 감쌀 높이 - 여기서 집는다
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
            applied_stop = False
            if not sv.in_bounds(trial):
                for j, dv in zip(sv.SERVO_JOINTS, dq):
                    lo, hi = me.ALL_LIMITS[j]
                    trial[j] = min(hi, max(lo, self.pose[j] + float(dv) * 0.4))
            # 이 스텝을 적용하면 컵/테이블을 파고드나? 미리 본다.
            if stop_gap is not None:
                saved = dict(self.pose)
                self.pose = trial
                self.sync()
                gap, _ = self.clearance()
                if gap < stop_gap:
                    # 파고든다 - 스텝을 취소하고 여기서 멈춘다(닿기 직전).
                    self.pose = saved
                    self.sync()
                    return True
            else:
                self.pose = trial
        return me._dist(self.hand(), target) * 100 <= done_cm

    def hold(self, label, n=6):
        for _ in range(n):
            self.snap(label, look=self.cup)


def run(cup_xy=(0.31, -0.14), noise_px=2.0, record=None):
    w = PickWorld(cup_xy=cup_xy)
    log = []

    cup = w.cup
    # 접근은 반드시 컵 '위' 를 경유한다. 컵 옆으로 곧장 가면 컵을 쳐서
    # 넘어뜨린다(투과). 위에서 수직으로 내려와 집는다.
    GRASP_GAP = 0.3    # 팔<->컵 표면이 이만큼 남으면 잡는다 (cm)
    above_cup = (cup[0], cup[1], cup[2] + 0.15)
    grasp_at = (cup[0], cup[1], cup[2] - 0.04)
    above_tray = (TRAY[0], TRAY[1] - 0.02, TRAY[2] + CUP_H / 2 + 0.12)
    place_at = (TRAY[0], TRAY[1] - 0.02, TRAY[2] + CUP_H / 2 + 0.03)

    # 0) 팔을 움직이기 전에 눈으로 먼저 컵을 찾는다.
    seen = w.search()
    log.append(('눈으로 컵 찾기', seen, 0.0))

    ok1 = w.servo_to(above_cup, 'approach', done_cm=3.0, noise_px=noise_px, seed=1)
    log.append(('컵 상공(위)', ok1, me._dist(w.hand(), above_cup) * 100))
    ok2 = w.servo_to(grasp_at, 'descend', done_cm=1.0, noise_px=noise_px, seed=2,
                     stop_gap=GRASP_GAP)
    gap, pair = w.clearance()
    log.append(('수직 하강 (팔<->컵 %.1fcm 남기고 정지)' % gap, gap >= -0.5, 0.0))

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

    # 프레임별 최소 clearance - '어느 프레임에서 얼마나 파고들었나' 를 자동
    # 보고한다. 이게 원래 부탁받은 것: 결과만 보지 말고 중간 프레임을 훑어라.
    pierce = [(i, c, p) for i, (c, p) in enumerate(w.clearances) if c < 0]
    if pierce:
        worst = min(pierce, key=lambda x: x[1])
        log.append(('투과: %d/%d 프레임에서 팔이 장면을 파고듦 (최악 프레임 %d: '
                    '%s %.1fcm)' % (len(pierce), len(w.clearances),
                                    worst[0], worst[2], worst[1]),
                    False, 0.0))

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
