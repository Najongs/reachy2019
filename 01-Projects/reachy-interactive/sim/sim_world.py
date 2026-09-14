"""임의의 테이블 위 장면 하나 = 로봇 + 물체들. 파이프라인의 '환경' 층.

sim_mujoco 가 실물 기구학으로 만든 로봇에, 테이블과 물체 몇 개를 얹은 것이다.
물체는 종류·크기·색·위치를 자유롭게 준다(sim_tasks 가 만든다). 검증기가 쓰는
것과 같은 순기구학이므로, 여기서 안전하면 실물에서도 안전하다.

제공하는 것:
  * render_eye()   로봇 눈 카메라 프레임 (접지 모델이 보는 그림)
  * render_third() 제3자 프레임 (사람이 보는 그림)
  * set_gaze(y,z)  실물 Orbita 기구학으로 시선을 둔다 (오차 0도)
  * set_arm(pose)  팔 관절 각도
  * object_pos(n)  물체 위치 (참값 - 실물에서는 카메라로 추정할 것)
  * pixel_of(p)    3D 점이 눈 화면의 어느 픽셀에 오는가
  * clearance()    팔 전체와 물체들 사이 최소 표면거리 (투과 검사)
  * move_object(n, pos)  물체를 옮긴다 (집어서 놓기)
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

TABLE_TOP = -0.27       # 기본 높이. 태스크가 높이를 정해 올 수 있다.


def _table(table_top=TABLE_TOP):
    """높이 table_top 의 테이블. 낮을수록 팔이 넘을 턱이 낮아진다."""
    return {'name': 'table', 'type': 'box',
            'pos': (0.37, 0.0, table_top - 0.015),
            'size': (0.19, 0.35, 0.015), 'rgba': (0.55, 0.42, 0.28, 1),
            'collide': True, 'movable': False}

# 물체 원형. size 는 MuJoCo 규약(원기둥=[반지름,반높이], 상자=[반폭...], 구=[반지름]).
OBJECT_LIBRARY = {
    'cup':   {'type': 'cylinder', 'size': (0.02, 0.05),  'rgba': (0.85, 0.35, 0.20, 1)},
    'block': {'type': 'box',      'size': (0.03, 0.03, 0.03), 'rgba': (0.30, 0.45, 0.85, 1)},
    'ball':  {'type': 'sphere',   'size': (0.03,),       'rgba': (0.90, 0.75, 0.20, 1)},
    'can':   {'type': 'cylinder', 'size': (0.033, 0.06), 'rgba': (0.60, 0.60, 0.65, 1)},
    # 실물풍 사물들 - 색·비율이 실제와 비슷하게.
    'bottle': {'type': 'cylinder', 'size': (0.026, 0.09), 'rgba': (0.20, 0.55, 0.30, 1)},
    'book':  {'type': 'box',      'size': (0.055, 0.04, 0.012), 'rgba': (0.55, 0.20, 0.15, 1)},
    'apple': {'type': 'sphere',   'size': (0.035,),      'rgba': (0.80, 0.12, 0.10, 1)},
    'tray':  {'type': 'box',      'size': (0.07, 0.05, 0.005), 'rgba': (0.25, 0.55, 0.35, 1),
              'movable': False},
    # 바구니: 바닥 + 벽 4면. 테두리(rim) 위에서 넣어야 한다 - 옆에서
    # 밀어 넣으면 벽에 걸린다. size 는 (외반폭x, 외반폭y, 벽높이반).
    'basket': {'type': 'box', 'size': (0.06, 0.06, 0.03),
               'rgba': (0.65, 0.45, 0.25, 1), 'movable': False,
               'parts': True},
}

BASKET_WALL = 0.008


def _basket_parts(size):
    hx, hy, hz = size
    w = BASKET_WALL
    return [
        {'type': 'box', 'size': (hx, hy, w), 'off': (0, 0, w)},            # 바닥
        {'type': 'box', 'size': (w, hy, hz), 'off': (-(hx - w), 0, hz)},   # 벽들
        {'type': 'box', 'size': (w, hy, hz), 'off': (hx - w, 0, hz)},
        {'type': 'box', 'size': (hx, w, hz), 'off': (0, -(hy - w), hz)},
        {'type': 'box', 'size': (hx, w, hz), 'off': (0, hy - w, hz)},
    ]


def make_object(name, kind, xy, z=None, table_top=TABLE_TOP):
    """이름 kind 의 물체를 (x, y) 에 놓는다. z 는 테이블 위에 얹히도록 자동."""
    tpl = OBJECT_LIBRARY[kind]
    if z is None:
        # 테이블 면 위에 밑바닥이 닿게. 반높이만큼 올린다.
        half = tpl['size'][-1] if tpl['type'] == 'box' else tpl['size'][-1] \
            if tpl['type'] == 'cylinder' else tpl['size'][0]
        if tpl.get('parts'):
            half = 0.0                       # 복합 물체는 pos 가 바닥면 기준
        elif tpl['type'] == 'cylinder':
            half = tpl['size'][1]
        elif tpl['type'] == 'sphere':
            half = tpl['size'][0]
        else:
            half = tpl['size'][2]
        z = table_top + half
    out = {'name': name, 'kind': kind, 'type': tpl['type'],
           'pos': (xy[0], xy[1], z), 'size': tpl['size'], 'rgba': tpl['rgba'],
           'collide': True, 'movable': tpl.get('movable', True)}
    if tpl.get('parts'):
        out['parts'] = _basket_parts(tpl['size'])
    return out


class World(object):
    """물체 목록 하나로 정의되는 장면."""

    def __init__(self, objects, width=640, height=480, table_top=TABLE_TOP,
                 light=None):
        import mujoco

        self.mujoco = mujoco
        self.table_top = table_top
        self.objects = [_table(table_top)] + list(objects)
        self.model = mujoco.MjModel.from_xml_string(
            sm.build_mjcf(keepout=False, objects=self.objects, light=light))
        self.data = mujoco.MjData(self.model)
        self.idx = sm._qpos_index(mujoco, self.model)
        self.width, self.height = width, height
        self._renderer = None
        self._gid = {}
        for i in range(self.model.ngeom):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if n:
                self._gid[n] = i
        self._arm_gids = [(i, n) for n, i in self._gid.items()
                          if n.startswith('right_arm')]
        # 복합 물체: name -> 부품 지오메트리 id 목록, 논리 중심은 따로 든다.
        self._parts = {}
        self._centers = {}
        for o in self.objects:
            if o.get('parts'):
                self._parts[o['name']] = [
                    self._gid['%s_p%d' % (o['name'], k)]
                    for k in range(len(o['parts']))]
                self._centers[o['name']] = tuple(o['pos'])

        # 시작 자세 = 실물 휴식(모든 관절 0, 팔을 늘어뜨림). 예전 '대기'
        # 자세는 손을 테이블 밑(z-0.44, x0.36)에 넣고 있어서, 서보가 상판을
        # 뚫고 올라왔다 - 실물이면 모서리에 팔을 박는 경로다. 휴식은 테이블
        # 여유 13.6cm 로 안전하고, 팔을 쓰는 동작은 execute 가 '들어올리기'
        # 단계를 먼저 거친다.
        self._dofadr = {}
        for jid in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if n:
                self._dofadr[n] = self.model.jnt_dofadr[jid]
        self._free = {o['name']: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, o['name'])
            for o in self.objects if o.get('movable')}
        self.pose = {j: 0.0 for j in self.idx}
        self._forward()
        self.step_physics(40)              # 물체들이 테이블에 안착

    # -- 상태 --------------------------------------------------------------

    def _forward(self):
        sm._set_pose(self.data, self.idx, self.pose)
        self.mujoco.mj_forward(self.model, self.data)

    def step_physics(self, substeps=8, prev=None):
        """물리를 적분한다. 팔은 위치 구동, 자유 바디(물체)만 중력·접촉.

        prev 를 주면 팔 자세를 하위스텝에 걸쳐 '보간' 한다 - 최종 자세로
        한 번에 점프하면 얇은 조/링크가 물체 안으로 텔레포트해 솔버가
        물체를 박힌 채 끌고 다닌다 (자석처럼 보이는 터널링 유착, 실측:
        공이 손목에 매달려 상승). 잘게 쓸면 밀려날 뿐 삼켜지지 않는다."""
        mj = self.mujoco
        for k in range(substeps):
            if prev is not None:
                u = (k + 1) / substeps
                pose_k = {j: prev.get(j, 0.0)
                          + (self.pose.get(j, 0.0) - prev.get(j, 0.0)) * u
                          for j in self.pose}
            else:
                pose_k = self.pose
            sm._set_pose(self.data, self.idx, pose_k)
            for joint, adr in self.idx.items():
                if not joint.endswith('_free'):
                    dof = self._dofadr.get(joint)
                    if dof is not None:
                        self.data.qvel[dof] = 0.0
            mj.mj_step(self.model, self.data)
        sm._set_pose(self.data, self.idx, self.pose)
        mj.mj_forward(self.model, self.data)

    def set_arm(self, pose):
        prev = dict(self.pose)
        self.pose.update(pose)
        self._forward()
        self.step_physics(prev=prev)

    def set_gaze(self, y, z, x=0.5):
        """시선을 (y, z at x) 로 즉시. 실물 Orbita 가 못 보는 방향이면 None.

        즉시 이동은 내부 프로브/미세 되먹임용이다. 눈에 보이는 시선 이동은
        glide_gaze 를 써라 - 실물 목은 순간이동하지 않는다.
        """
        disks = sm.set_gaze(self.pose, y, z, x)
        self._forward()
        self._gaze_pt = (max(x, 0.05), y, z)
        return disks

    def glide_gaze(self, y, z, x=0.5, on_step=None, max_step_deg=6.0):
        """시선을 현재 방향에서 목표로 '돌려서' 가져간다.

        실물 목에는 속도 상한이 있다(실기 커밋과 같은 원칙). 시뮬이 시선을
        한 스텝에 박아 넣으면 눈 카메라 뷰가 프레임 사이에서 뚝 바뀌어
        영상이 끊겨 보이고, 자연스러움 평가도 왜곡된다. 시선 방향을
        max_step_deg 씩 나눠 보간한다. on_step(i, n) 으로 중간 프레임을
        찍을 수 있다.
        """
        target = (max(x, 0.05), y, z)
        cur = getattr(self, '_gaze_pt', None) or (0.5, 0.0, 0.0)

        def _norm(v):
            n = math.sqrt(sum(c * c for c in v)) or 1.0
            return tuple(c / n for c in v)

        a, b = _norm(cur), _norm(target)
        dot = max(-1.0, min(1.0, sum(x1 * x2 for x1, x2 in zip(a, b))))
        ang = math.degrees(math.acos(dot))
        n = max(1, int(math.ceil(ang / max(max_step_deg, 0.5))))
        for i in range(1, n + 1):
            t = i / n
            pt = tuple(c0 + (c1 - c0) * t for c0, c1 in zip(cur, target))
            self.set_gaze(pt[1], pt[2], x=pt[0])
            if on_step is not None:
                on_step(i, n)
        return n

    def nudge_gaze(self, point, max_step_deg=6.0):
        """시선을 point 쪽으로 '한 스텝만' 돌린다. 도달하면 True.

        팔이 움직이는 동안 머리가 손/물체를 부드럽게 따라가는 용도다.
        glide_gaze 는 목표까지 다 돌지만, 이건 서보 스텝마다 한 번씩
        불러 목 속도 상한을 지키며 추적하게 한다.
        """
        target = (max(point[0], 0.05), point[1], point[2])
        cur = getattr(self, '_gaze_pt', None) or (0.5, 0.0, 0.0)

        def _norm(v):
            n = math.sqrt(sum(c * c for c in v)) or 1.0
            return tuple(c / n for c in v)

        a, b = _norm(cur), _norm(target)
        dot = max(-1.0, min(1.0, sum(x1 * x2 for x1, x2 in zip(a, b))))
        ang = math.degrees(math.acos(dot))
        if ang <= max_step_deg:
            self.set_gaze(target[1], target[2], x=target[0])
            return True
        t = max_step_deg / ang
        pt = tuple(c0 + (c1 - c0) * t for c0, c1 in zip(cur, target))
        self.set_gaze(pt[1], pt[2], x=pt[0])
        return False

    def look_at(self, point):
        return self.set_gaze(point[1], point[2], x=max(point[0], 0.05))

    def center_on(self, point, tol_px=None, iters=6):
        """물체가 눈 화면 '중앙' 에 오도록 시선을 맞춘다.

        look_at 은 목을 물체 '방향' 으로 두지만, 카메라가 목보다 6cm 앞·
        아래에 있어 물체가 화면 아래에 맺힌다. 픽셀 오차를 보고 시선 (y,z) 를
        몇 번 되먹임해 실제로 화면 한가운데 오게 한다 - 이게 실물에서 '대상을
        응시' 하는 것에 해당한다.
        """
        y, z = point[1], point[2]
        x = max(point[0], 0.05)
        for _ in range(iters):
            self.set_gaze(y, z, x)
            px = self.pixel_of(point)
            if px is None:
                return False
            du = (px[0] - self.width / 2) / self.width
            dv = (px[1] - self.height / 2) / self.height
            if abs(du) < 0.03 and abs(dv) < 0.03:
                return True
            # 화면 오른쪽(u 큼)=시선을 오른쪽(y 작게), 아래(v 큼)=시선 아래(z 작게)
            y -= du * 0.35
            z -= dv * 0.35
        return abs(du) < 0.06 and abs(dv) < 0.06

    def hand(self):
        return me.forward_kinematics(me.CHAINS['right_arm'], self.pose)

    def object_pos(self, name):
        if name in self._free:
            return tuple(float(v) for v in self.data.xpos[self._free[name]])
        if name in self._centers:
            return self._centers[name]
        return tuple(float(v) for v in self.model.geom_pos[self._gid[name]])

    def movable_objects(self):
        return [o['name'] for o in self.objects if o.get('movable')]

    def move_object(self, name, pos):
        if name in self._free:
            adr = self.idx[name + '_free']
            self.data.qpos[adr:adr + 3] = pos
            self.data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
            dof = self._dofadr[name + '_free']
            self.data.qvel[dof:dof + 6] = 0
            self.mujoco.mj_forward(self.model, self.data)
            return
        if name in self._parts:
            old = self._centers[name]
            d = tuple(pos[i] - old[i] for i in range(3))
            for gid in self._parts[name]:
                self.model.geom_pos[gid] = [
                    self.model.geom_pos[gid][i] + d[i] for i in range(3)]
            self._centers[name] = tuple(pos)
        else:
            self.model.geom_pos[self._gid[name]] = pos
        self._forward()

    # -- 카메라 ------------------------------------------------------------

    def _rend(self):
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(
                self.model, height=self.height, width=self.width)
        return self._renderer

    def render_eye(self):
        r = self._rend()
        r.update_scene(self.data, camera='robot_eye')
        return r.render().copy()

    def render_third(self):
        r = self._rend()
        r.update_scene(self.data, camera=sv._third_person(self.mujoco))
        return r.render().copy()

    def pixel_of(self, point):
        """3D 점 -> 눈 화면 픽셀 (u, v). 화면 밖/뒤면 None."""
        return sv.pixel_of(self.model, self.data, self.mujoco, 'robot_eye',
                           point, self.width, self.height)

    def visible(self, name, margin=0.06):
        """이 물체가 지금 시선에서 화면 안에 보이나."""
        px = self.pixel_of(self.object_pos(name))
        if px is None:
            return False
        return (margin * self.width < px[0] < (1 - margin) * self.width
                and margin * self.height < px[1] < (1 - margin) * self.height)

    # -- 안전/투과 ---------------------------------------------------------

    def jaw_clearance(self, name, jaw=None):
        """그리퍼 조(움직/고정)와 이 물체 사이 최소 표면거리(cm).

        조임 판정 전용 - 팔 전체 최소를 쓰면 손목 캡슐 접촉이 조임으로
        오인된다 (실측: 조가 안 닿았는데 잡힘 처리)."""
        mj = self.mujoco
        names = {'moving': ('right_arm_jaw_moving',),
                 'fixed': ('right_arm_jaw_fixed',)}.get(
            jaw, ('right_arm_jaw_moving', 'right_arm_jaw_fixed',
                  'right_arm_palm'))
        jids = [self._gid[n] for n in names if n in self._gid]
        oids = self._parts.get(name) or [self._gid[name]]
        return min(mj.mj_geomDistance(self.model, self.data, g, o, 1.0, None)
                   for g in jids for o in oids) * 100

    def pinch_tilt(self):
        """집게면 기울기(도): 조 분리 방향이 수평에서 벗어난 각."""
        gm = self._gid.get('right_arm_jaw_moving')
        gf = self._gid.get('right_arm_jaw_fixed')
        if gm is None or gf is None:
            return 0.0
        v = self.data.geom_xpos[gm] - self.data.geom_xpos[gf]
        n = float((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5) or 1.0
        return abs(math.degrees(math.asin(abs(float(v[2])) / n)))

    def ghost_eval(self, pose_delta):
        """물리 교란 없는 유령 프로브: 자세를 임시 적용해 (기울기, 슬롯)
        을 재고 원상복구한다. 물체 qpos 는 건드리지 않는다."""
        saved_qpos = self.data.qpos.copy()
        merged = dict(self.pose); merged.update(pose_delta)
        sm._set_pose(self.data, self.idx, merged)
        self.mujoco.mj_forward(self.model, self.data)
        out = (self.pinch_tilt(), self.grip_slot())
        self.data.qpos[:] = saved_qpos
        self.mujoco.mj_forward(self.model, self.data)
        return out

    def grip_slot(self):
        """두 조 사이 슬롯의 세계좌표 중심 - 물체가 '여기' 에 와야 집힌다.
        손목 FK 점과 어긋나 있어(자세 따라 회전) 정렬은 이걸 기준으로."""
        mj = self.mujoco
        gm = self._gid.get('right_arm_jaw_moving')
        gf = self._gid.get('right_arm_jaw_fixed')
        if gm is None or gf is None:
            return self.hand()
        a = self.data.geom_xpos[gm]
        b = self.data.geom_xpos[gf]
        return tuple(float((a[i] + b[i]) / 2) for i in range(3))

    def clearance_of(self, name, exclude_jaws=False):
        """팔 전체와 이 물체 하나 사이 최소 표면거리(cm).

        exclude_jaws: 조/손바닥 제외 - 파지 하강 때 조 끝이 테이블에
        접근하는 건 올바른 동작이라, 테이블 가드가 이를 재면 낮은 물체
        (책 등)는 구조적으로 접근 불가가 된다 (충돌 과회피의 뿌리)."""
        mj = self.mujoco
        oids = self._parts.get(name) or [self._gid[name]]
        skip = {'right_arm_jaw_moving', 'right_arm_jaw_fixed',
                'right_arm_palm'} if exclude_jaws else set()
        return min(mj.mj_geomDistance(self.model, self.data, gid, oid, 1.0,
                                      None)
                   for gid, gn in self._arm_gids if gn not in skip
                   for oid in oids) * 100

    def clearance(self, ignore=()):
        """팔 전체와 (충돌하는) 물체들 사이 최소 표면거리(cm), 부위쌍.

        음수면 파고든 것. ignore 에 든 물체 이름은 뺀다(집고 있는 것 등).
        MuJoCo 의 정확한 지오메트리간 거리를 쓴다.
        """
        mj = self.mujoco
        targets = []
        for o in self.objects:
            if not o.get('collide') or o['name'] in ignore:
                continue
            for oid in (self._parts.get(o['name']) or [self._gid[o['name']]]):
                targets.append((oid, o['name']))
        worst = (99.0, None)
        for gid, gname in self._arm_gids:
            for oid, oname in targets:
                d = mj.mj_geomDistance(self.model, self.data, gid, oid, 1.0, None)
                if d < worst[0]:
                    worst = (d * 100,
                             '%s~%s' % (gname.replace('right_arm_', ''), oname))
        return worst

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
