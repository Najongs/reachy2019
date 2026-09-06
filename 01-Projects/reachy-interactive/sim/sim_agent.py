"""태스크 하나를 인지->계획->행동->검증으로 수행한다. 파이프라인의 '몸통'.

    1) 인지    로봇 눈으로 장면을 본다. 필요하면 시선을 돌려 대상을 화면에
               담는다(카메라 조정 = 상황 파악).
    2) 계획    planner 가 '무엇을 볼지 / 어떻게 움직일지' 를 정한다.
               - BrokerPlanner: opus 에게 눈 그림 + 지시를 보내 행동 JSON 을 받음
               - OraclePlanner: 참값을 써서 계획(opus 없이 harness 검증용)
    3) 행동    계획을 월드에서 실행한다(시선 이동 / 팔 서보 / 집어 놓기).
    4) 검증    task.success(world) 로 됐는지 판정하고, 매 프레임 투과를 검사.

planner 를 갈아 끼우는 구조라, opus 를 붙이기 전에 harness 가 옳게 도는지
OraclePlanner 로 먼저 확인할 수 있다. opus 로 바꾸면 그 자리에 지능이 들어온다.

사용:
    python3 sim/sim_agent.py --tasks tasks.json --planner oracle
    python3 sim/sim_agent.py --tasks tasks.json --planner broker --token reachy2019
"""

import json
import math
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')   # GPU0 만 쓴다 (ollama 와 동거)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                           # noqa: E402
import sim_servo as sv                             # noqa: E402
import sim_tasks as T                              # noqa: E402
import sim_world as W                              # noqa: E402


# ============================================================ 계획자 ==========

class OraclePlanner(object):
    """참값으로 계획한다. opus 자리에 들어갈 것의 '정답' 판.

    이걸로 harness(인지/행동/검증)가 옳게 도는지 먼저 확인한다. 여기서 안
    풀리는 태스크는 opus 로도 안 풀린다 - 기구학이나 안전의 한계다.
    """

    name = 'oracle'

    def perceive(self, world, task):
        """대상을 화면 중앙에 담는다. (대상의 참 위치를 안다)"""
        ok = world.center_on(world.object_pos(task['target']))
        return {'seen': ok}

    def plan(self, world, task, view):
        """유형별로 목표를 돌려준다.  {mode, target_xyz?, ...}"""
        obj = world.object_pos(task['target'])
        if task['kind'] == 'look':
            return {'mode': 'gaze', 'point': obj}
        if task['kind'] in ('point', 'reach'):
            return {'mode': 'reach', 'point': obj,
                    'done_cm': 8.0 if task['kind'] == 'reach' else 12.0}
        if task['kind'] == 'pick':
            return {'mode': 'pick', 'object': task['target'],
                    'tray': task['tray']}
        return {'mode': 'noop'}


class BrokerPlanner(object):
    """opus 시각 접지로 계획한다 - 참값 없이 카메라와 기억만.

    분업이 핵심이다. opus 는 **어느 박스가 대상인지**만 고른다(시각·언어 판단).
    박스를 3D 목표로 바꾸는 기하(픽셀+크기 -> 방향+거리)와 서보는 결정론이다.
    opus 에게 좌표 계산을 시키면 그림만 보고 지어내게 된다.

    인지: sim_perceive.sweep 으로 시선을 훑어 SceneMemory 를 채우고, 대상
    종류가 기억에 잡히면 그 방향으로 시선을 되돌린 뒤 현재 프레임의 검출로
    접지를 요청한다.
    """

    name = 'broker'

    def __init__(self, url='http://127.0.0.1:8080', token=None,
                 session='sim-agent', noise_px=2.0, dropout=0.05, seed=0):
        import random
        sys.path.insert(0, os.path.join(HERE, '..', 'robot'))
        from llm_client import BrokerClient
        import sim_perceive as P
        self.P = P
        self.client = BrokerClient(url, token=token, session=session)
        self.noise_px, self.dropout = noise_px, dropout
        self.rng = random.Random(seed)
        self.lessons = []          # 배치 안에서 누적되는 지적 (7->4 되먹임)

    def _jpeg(self, img):
        import base64
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.fromarray(img).convert('RGB').save(buf, format='JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    def perceive(self, world, task):
        """시선을 훑어 기억을 채우고, 대상 종류 방향으로 시선을 되돌린다."""
        mem = self.P.SceneMemory()
        self.P.sweep(world, mem, noise_px=self.noise_px,
                     dropout=self.dropout, rng=self.rng)
        kind = next((o['kind'] for o in task['scene']
                     if o['name'] == task['target']), None)
        item = mem.find(kind) if kind else None
        if item is not None:
            self.P.gaze_toward(world, item)
        dets = self.P.detect(world, self.noise_px, self.dropout, self.rng)
        return {'seen': bool(dets), 'memory': mem, 'dets': dets}

    def plan(self, world, task, view):
        """opus 에게 '몇 번 박스인가' 를 묻고, 그 박스로 3D 목표를 만든다."""
        dets = view.get('dets') or []
        mem = view.get('memory')
        img = self.P.overlay(world, dets)
        prompt = '지시: %s' % task['instruction']
        if self.lessons:
            prompt = ('지난 시도의 지적: %s\n' % ' / '.join(self.lessons[-3:])
                      ) + prompt
        if mem is not None:
            prompt += '\n' + mem.summary()
        raw = self.client.ask_vision(prompt, image=self._jpeg(img))
        choice = _parse_grounding(raw)
        if choice is None:
            return {'mode': 'noop', 'raw': raw, 'why': '접지 응답 해석 불가'}

        box = choice.get('box')
        if box is not None and 0 <= int(box) < len(dets):
            det = dets[int(box)]
            point = _det_to_3d(world, det)
            mode = 'gaze' if task['kind'] == 'look' else 'reach'
            done = 12.0 if task['kind'] == 'point' else 8.0
            return {'mode': mode, 'point': point, 'done_cm': done,
                    'raw': raw, 'grounded_det': det}
        if choice.get('memory') and mem is not None:
            item = mem.find(choice['memory'])
            if item is not None:
                self.P.gaze_toward(world, item)
                dets2 = self.P.detect(world, self.noise_px, self.dropout,
                                      self.rng)
                if dets2:
                    det = max(dets2, key=lambda d: d['size_px'])
                    point = _det_to_3d(world, det)
                    mode = 'gaze' if task['kind'] == 'look' else 'reach'
                    return {'mode': mode, 'point': point, 'done_cm': 8.0,
                            'raw': raw, 'grounded_det': det}
        return {'mode': 'noop', 'raw': raw, 'why': '대상을 찾지 못함'}


def _parse_grounding(raw):
    """접지 응답 JSON 을 관대하게 파싱. {'box':.., 'memory':..} 또는 None."""
    if not raw:
        return None
    start = raw.find('{')
    end = raw.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        out = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _det_to_3d(world, det):
    """검출(픽셀+크기) -> 대략적 3D 점. 참값을 쓰지 않는다.

    방향은 시선+픽셀 오프셋으로, 거리는 화면 크기(가까울수록 크다)로 추정 -
    servo 의 크기 단서와 같은 원리다. 서보가 화면 오차로 계속 고치므로 이
    초기 추정은 대충이어도 된다.
    """
    import sim_perceive as P
    f = (world.height / 2.0) / math.tan(math.radians(58 / 2.0))
    dist = P.KIND_SIZE.get(det['kind'], 0.08) * f / max(det['size_px'], 8.0)
    du = math.atan2(det['u'] - world.width / 2, f)
    dv = math.atan2(det['v'] - world.height / 2, f)
    pan = math.radians(world.pose.get('eye.pan', 0.0)) - du
    tilt = math.radians(world.pose.get('eye.tilt', 0.0)) + dv
    # 카메라 위치에서 그 방향으로 dist 만큼
    mj = world.mujoco
    cid = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_CAMERA, 'robot_eye')
    cam = world.data.cam_xpos[cid]
    return (float(cam[0] + dist * math.cos(tilt) * math.cos(pan)),
            float(cam[1] + dist * math.cos(tilt) * math.sin(pan)),
            float(cam[2] - dist * math.sin(tilt)))


# ============================================================ 실행 ============

# 테이블 앞모서리 '위' 의 준비 자세 (탐색으로 찾음: 손끝 (0.28,-0.20,-0.13),
# 테이블 여유 9.4cm, 금지구역 통과). 휴식(팔 늘어뜨림)에서 바로 물체로 서보하면
# 팔이 상판을 밑에서 뚫고 올라온다 - 실물이면 모서리에 박는 경로다. 그래서
# 실물 사람이 하듯 '먼저 들어 올리고, 위에서 접근' 한다.
READY = {'right_arm.shoulder_pitch': -8.0, 'right_arm.shoulder_roll': -66.0,
         'right_arm.arm_yaw': 32.0, 'right_arm.elbow_pitch': -124.0}
TABLE_MIN_CM = 0.5      # 서보 중 팔<->테이블 표면이 이 밑으로 못 내려간다


def _lift_ready(world, frames=None, trace=None, steps=6):
    """휴식 -> 준비 자세, 3단계로: 벌리고 -> 굽히고 -> 돌려 넣기.

    관절을 한꺼번에 보간하면 전완이 테이블 앞모서리를 스친다(실측 -5cm).
    사람이 하듯 팔을 옆으로 벌려 테이블 옆면 밖에서 굽힌 뒤 위에서 돌려
    넣으면 경로 최소 여유 9.5cm, 투과 0 이다.
    """
    stages = [
        {'right_arm.shoulder_roll': READY['right_arm.shoulder_roll']},
        {'right_arm.elbow_pitch': READY['right_arm.elbow_pitch'],
         'right_arm.shoulder_pitch': READY['right_arm.shoulder_pitch']},
        {'right_arm.arm_yaw': READY['right_arm.arm_yaw']},
    ]
    cur = {j: world.pose.get(j, 0.0) for j in READY}
    for stage in stages:
        tgt = dict(cur)
        tgt.update(stage)
        for k in range(1, steps + 1):
            u = k / float(steps)
            wgt = u * u * (3 - 2 * u)           # smoothstep
            pose = {j: cur[j] + (tgt[j] - cur[j]) * wgt for j in READY}
            if not sv.in_bounds({**world.pose, **pose}):
                continue
            world.set_arm(pose)
            if trace is not None:
                trace.append({'table': world.clearance_of('table'),
                              'obj': world.clearance(ignore=('table',))[0]})
            if frames is not None:
                frames.append(_snap(world, 'lift'))
        cur = tgt
    return True

def _servo(world, target, done_cm, noise_px=1.0, steps=45, seed=0,
           stop_gap=None, obj_stop=None, frames=None, note='', trace=None,
           table_min=TABLE_MIN_CM):
    """화면 오차로 손을 target 까지. sim_servo 와 같은 방식, World 위에서."""
    import numpy as np

    rng = np.random.RandomState(seed)
    SIZE_GAIN = 220.0
    mj = world.mujoco

    def err(p, measured=True):
        saved = dict(world.pose)
        world.pose.update(p)
        sv.look_at(world.pose, target)
        world._forward()
        tp = world.pixel_of(target)
        hand = world.hand()
        hp = world.pixel_of(hand)
        cid = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_CAMERA, 'robot_eye')
        cam = tuple(float(v) for v in world.data.cam_xpos[cid])
        world.pose = saved
        world._forward()
        if tp is None or hp is None:
            return None
        size = SIZE_GAIN * math.log(me._dist(cam, hand)
                                    / max(me._dist(cam, target), 1e-6))
        n = rng.normal(0, noise_px, 5) if (noise_px and measured) else (0,) * 5
        return (tp[0] + n[0] - hp[0] - n[2], tp[1] + n[1] - hp[1] - n[3],
                size + n[4])

    blocked = 0
    for step in range(steps):
        world.look_at(target)
        if trace is not None:
            trace.append({'table': world.clearance_of('table'),
                          'obj': world.clearance(ignore=('table',))[0]})
        if frames is not None:
            frames.append(_snap(world, '%s %.1fcm' % (note,
                          me._dist(world.hand(), target) * 100)))
        if me._dist(world.hand(), target) * 100 <= done_cm:
            return True
        if stop_gap is not None and world.clearance()[0] <= stop_gap:
            return True
        if obj_stop is not None and world.clearance(ignore=('table',))[0] <= obj_stop:
            return True                     # 물체에 닿기 직전 - 여기서 멈춘다
        e = err({})
        base = err({}, measured=False)
        if e is None or base is None:
            return False
        J = []
        for j in sv.SERVO_JOINTS:
            e2 = err({j: world.pose[j] + 1.0}, measured=False)
            J.append((0.0, 0.0, 0.0) if e2 is None else
                     tuple(e2[k] - base[k] for k in range(3)))
        J = np.array(J).T
        dq, *_ = np.linalg.lstsq(J.T @ J + 4.0 * np.eye(len(sv.SERVO_JOINTS)),
                                 J.T @ (-np.array(e)), rcond=None)
        dq = np.clip(dq * 0.8, -sv.STEP_DEG, sv.STEP_DEG)
        trial = dict(world.pose)
        for j, dv in zip(sv.SERVO_JOINTS, dq):
            lo, hi = me.ALL_LIMITS[j]
            trial[j] = min(hi, max(lo, trial[j] + float(dv)))
        if not sv.in_bounds(trial):
            for j, dv in zip(sv.SERVO_JOINTS, dq):
                lo, hi = me.ALL_LIMITS[j]
                trial[j] = min(hi, max(lo, world.pose[j] + float(dv) * 0.4))
        # 한 스텝 앞을 본다: 물체를 뚫으면 도착으로 치고, 테이블을 뚫으면
        # 스텝을 줄여 보고 그래도 안 되면 버린다(실물은 상판 위로만 다닌다).
        saved = dict(world.pose)
        world.set_arm(trial)
        tab = world.clearance_of('table')
        obj = world.clearance(ignore=('table',))[0]
        if obj_stop is not None and obj < obj_stop:
            world.set_arm(saved)
            return True                       # 물체에 닿기 직전 - 여기서 멈춘다
        if tab < table_min:
            half = {j: saved[j] + (trial[j] - saved[j]) * 0.4 for j in trial}
            world.set_arm(half)
            if world.clearance_of('table') < table_min:
                world.set_arm(saved)          # 테이블을 뚫는 스텝 - 버린다
                blocked += 1
                if blocked >= 5:
                    break                     # 계속 막히면 더 못 간다
                continue
        blocked = 0
    return me._dist(world.hand(), target) * 100 <= done_cm


def _snap(world, note=''):
    import numpy as np
    eye = world.render_eye()
    third = world.render_third()
    pierced = world.clearance()[0] < 0
    if pierced:
        for img in (eye, third):
            img[:6, :] = [200, 40, 40]; img[-6:, :] = [200, 40, 40]
            img[:, :6] = [200, 40, 40]; img[:, -6:] = [200, 40, 40]
    frame = np.concatenate([eye, third], axis=1)
    if note:
        try:
            from PIL import Image, ImageDraw
            im = Image.fromarray(frame)
            ImageDraw.Draw(im).text((12, 10), note, fill=(235, 235, 235))
            frame = np.asarray(im)
        except Exception:
            pass
    return frame


def execute(world, plan, frames=None, trace=None):
    """계획을 월드에서 실행한다. (성공했다고 주장하지 않음 - 검증은 따로)"""
    mode = plan.get('mode')
    if mode == 'gaze':
        world.center_on(plan['point'])
        if frames is not None:
            frames.append(_snap(world, 'gaze'))
        return True
    if mode == 'reach':
        # 실물 순서: 팔을 먼저 테이블 위로 들어 올리고, 위에서 접근한다.
        _lift_ready(world, frames=frames, trace=trace)
        return _servo(world, plan['point'], plan['done_cm'], obj_stop=1.0,
                      frames=frames, note='reach', trace=trace)
    if mode == 'pick':
        _lift_ready(world, frames=frames, trace=trace)
        return _pick(world, plan['object'], plan['tray'], frames=frames)
    return False


def _pick(world, obj, tray, frames=None):
    """집어 쟁반에 놓기. (현재 기구학 한계로 위에서 집기는 어려움 - 기록됨)"""
    cup = world.object_pos(obj)
    above = (cup[0], cup[1], cup[2] + 0.14)
    _servo(world, above, 3.0, obj_stop=1.0, frames=frames,
           note='approach', seed=1)
    reached = _servo(world, (cup[0], cup[1], cup[2] - 0.02), 1.0,
                     obj_stop=0.5, frames=frames, note='descend', seed=2)
    gap, _ = world.clearance()
    grabbed = me._dist(world.hand(), cup) * 100 < 7.0 and gap > -0.5
    if grabbed:
        tp = world.object_pos(tray)
        world.move_object(obj, (cup[0], cup[1], cup[2]))   # 집힘
        # 쟁반 위로 옮긴다(손을 따라).
        _servo(world, (tp[0], tp[1] - 0.02, tp[2] + 0.12), 3.5,
               frames=frames, note='carry', seed=3)
        h = world.hand()
        world.move_object(obj, (tp[0], tp[1], world.TABLE_TOP + 0.05))
        if frames is not None:
            frames.append(_snap(world, 'placed'))
    return grabbed


# ============================================================ 루프 ===========

def run_task(task, planner, record_dir=None):
    """한 태스크: 인지 -> 계획 -> 행동 -> 검증. 결과 dict."""
    world = T.build_world(task)
    frames = [] if record_dir else None
    result = {'id': task['id'], 'kind': task['kind']}

    try:
        view = planner.perceive(world, task)
        result['perceived'] = bool(view.get('seen'))
        plan = planner.plan(world, task, view)
        result['plan'] = plan.get('mode')
        execute(world, plan, frames=frames)
        result['success'] = bool(T.success_fn(task)(world))
        # 투과를 둘로 나눈다: 대상/다른 물체를 뚫으면 진짜 실패(물체를 쳐서
        # 넘어뜨린다). 테이블을 스치는 건 접근에 따르는 것이라 따로 센다.
        obj_gap, obj_pair = world.clearance(ignore=('table',))
        tbl_gap, _ = world.clearance()
        result['object_clearance_cm'] = round(obj_gap, 1)
        result['table_clearance_cm'] = round(tbl_gap, 1)
        result['final_clearance_cm'] = round(obj_gap, 1)   # 물체 기준
        if obj_gap < -0.5:
            result['hit_object'] = obj_pair
    except Exception as e:
        result['error'] = '%s: %s' % (type(e).__name__, e)
        result['success'] = False
    finally:
        if record_dir and frames:
            _write_video(frames, os.path.join(record_dir, task['id'] + '.mp4'))
        world.close()
    return result


def _write_video(frames, path):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = imageio.get_writer(path, fps=10, macro_block_size=1)
    for f in frames:
        h, ww = f.shape[:2]
        w.append_data(f[:h - h % 2, :ww - ww % 2])
    w.close()


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', help='태스크 json (없으면 즉석 생성)')
    ap.add_argument('-n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--planner', choices=['oracle', 'broker'], default='oracle')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--token')
    ap.add_argument('--record', metavar='DIR', help='태스크별 영상 저장 폴더')
    args = ap.parse_args()

    if args.tasks:
        with open(args.tasks, encoding='utf-8') as fh:
            tasks = json.load(fh)['tasks']
    else:
        tasks = T.generate(args.n, args.seed)

    if args.planner == 'broker':
        planner = BrokerPlanner(args.url, token=args.token)
    else:
        planner = OraclePlanner()

    print('계획자: %s, 태스크 %d개\n' % (planner.name, len(tasks)))
    ok = 0
    by_kind = {}
    for t in tasks:
        r = run_task(t, planner, record_dir=args.record)
        ok += r.get('success', False)
        by_kind.setdefault(t['kind'], [0, 0])
        by_kind[t['kind']][0] += r.get('success', False)
        by_kind[t['kind']][1] += 1
        flag = '✓' if r.get('success') else '✗'
        pierce = ''
        if r.get('hit_object'):
            pierce = '  [물체투과 %s %.1fcm]' % (r['hit_object'],
                                             r['object_clearance_cm'])
        err = '  (%s)' % r['error'] if r.get('error') else ''
        print('  %s [%-5s] %-40s 인지=%s%s%s' % (
            flag, t['kind'], t['instruction'][:40],
            'O' if r.get('perceived') else 'X', pierce, err))

    print('\n성공 %d/%d' % (ok, len(tasks)))
    for k, (s, n) in sorted(by_kind.items()):
        print('  %-6s %d/%d' % (k, s, n))
    if args.record:
        print('\n영상: %s/<task id>.mp4' % args.record)


if __name__ == '__main__':
    main()
