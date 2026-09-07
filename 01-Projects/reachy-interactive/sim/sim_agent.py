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
            return {'mode': 'gaze', 'point': obj, 'target': task['target']}
        if task['kind'] in ('point', 'reach'):
            return {'mode': 'reach', 'point': obj, 'target': task['target'],
                    'approach': 'top',
                    'done_cm': 8.0 if task['kind'] == 'reach' else 12.0}
        if task['kind'] == 'pick':
            return {'mode': 'pick', 'object': task['target'],
                    'target': task['target'], 'tray': task['tray']}
        if task['kind'] == 'lift':
            return {'mode': 'lift_hold', 'object': task['target'],
                    'point': obj, 'target': task['target']}
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

        # 행동 유형도 opus 가 지시문에서 추론한다 (이론상 인지의 몫).
        # 유형 메타데이터는 판정에만 쓰고, 실행은 추론을 따른다 - 어긋나면
        # 기록에 남아 감사 대상이 된다. pick 은 슬라이스 밖이라 reach 로.
        action = choice.get('action')
        if action not in ('look', 'point', 'reach', 'pick', 'lift'):
            action = task['kind']
        act_mismatch = (action != task['kind']) or None
        approach = choice.get('approach')
        if approach not in ('top', 'side'):
            approach = 'top'

        box = choice.get('box')
        if box is not None and 0 <= int(box) < len(dets):
            det = dets[int(box)]
            point = _det_to_3d(world, det)
            if action == 'lift':
                return {'mode': 'lift_hold', 'point': point,
                        'action': action, 'action_mismatch': act_mismatch,
                        'raw': raw, 'grounded_det': det}
            if action == 'pick':
                tray_pt = None
                tb = choice.get('tray_box')
                if tb is not None and 0 <= int(tb) < len(dets):
                    tray_pt = _det_to_3d(world, dets[int(tb)])
                return {'mode': 'pick', 'point': point,
                        'tray_point': tray_pt,
                        'action': action, 'action_mismatch': act_mismatch,
                        'raw': raw, 'grounded_det': det}
            mode = 'gaze' if action == 'look' else 'reach'
            done = 12.0 if action == 'point' else 8.0
            return {'mode': mode, 'point': point, 'done_cm': done,
                    'action': action, 'action_mismatch': act_mismatch,
                    'approach': approach,
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
                    mode = 'gaze' if action == 'look' else 'reach'
                    return {'mode': mode, 'point': point, 'done_cm': 8.0,
                            'action': action,
                            'action_mismatch': act_mismatch,
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

# ── 행동 파라미터 (개선 루프가 돌리는 손잡이) ──────────────────────────────
#
# 동작의 '성격' 을 정하는 숫자들. 손으로 고치는 대신 sim_improve 가 품질
# 지표와 언어모델 비평을 보며 반복 조정한다. 객관 성공/안전은 문지기라
# 여기서 뭘 바꿔도 투과·미도달이 생기면 그 후보는 버려진다.
BEHAVIOR = {
    'step_deg': 4.0,        # 서보 한 스텝 최대 관절 변화 (작으면 부드럽고 느림)
    'gain': 0.8,            # 서보 이득 (크면 빠르고 거침)
    'damping': 4.0,         # 최소제곱 감쇠 (크면 신중)
    'table_min': 0.5,       # 테이블 표면 여유 하한 cm (크면 높이 돈다)
    'lift_steps': 6,        # 들어올리기 각 단계 보간 수 (많으면 부드러움)
    # 들어올리기 경유 자세 3개 - 경로의 '구조' 자체가 학습 대상이다.
    # 예전에는 벌리고-굽히고-돌려넣기 순서가 코드에 고정이었고 루프는
    # 도착 각도만 조율했다 (우회로 효율 0.4 상한). 이제 경유 자세들을
    # 통째로 탐색한다. 초기값 = 검증된 옛 3단계의 각 단계 끝 자세라
    # 출발점은 무투과 경로 그대로다. 투과는 문지기가 걸러 준다.
    'via1_pitch': 0.0,   'via1_roll': -66.0, 'via1_yaw': 0.0,  'via1_elbow': 0.0,
    'via2_pitch': -8.0,  'via2_roll': -66.0, 'via2_yaw': 0.0,  'via2_elbow': -124.0,
    'via3_pitch': -8.0,  'via3_roll': -66.0, 'via3_yaw': 32.0, 'via3_elbow': -124.0,
    'gaze_step_deg': 6.0,   # 시선 활강 한 스텝 각도 (작으면 목이 차분)
}


_VIA_JOINT = {'pitch': 'right_arm.shoulder_pitch',
              'roll': 'right_arm.shoulder_roll',
              'yaw': 'right_arm.arm_yaw',
              'elbow': 'right_arm.elbow_pitch'}


def via_poses():
    """BEHAVIOR 에서 들어올리기 경유 자세 3개를 만든다."""
    return [{j: BEHAVIOR['via%d_%s' % (n, k)] for k, j in _VIA_JOINT.items()}
            for n in (1, 2, 3)]
_BEHAVIOR_FILE = os.path.join(HERE, '..', 'config', 'behavior_params.json')


# 기준(문헌) 파라미터 - config 로드/루프 갱신 전의 값. 태스크 실현가능성
# 판정은 이걸로 고정한다: '가능한 태스크' 의 정의가 학습 중인 파라미터를
# 따라 움직이면, 파라미터가 나빠질수록 가능한 태스크가 사라지는 순환이 된다
# (실제로 10시간 루프가 이걸로 빈 태스크 -> 0/0 공회전에 빠졌다).
DEFAULTS = dict(BEHAVIOR)


def load_behavior():
    try:
        with open(_BEHAVIOR_FILE, encoding='utf-8') as fh:
            saved = json.load(fh)
        BEHAVIOR.update({k: v for k, v in saved.items() if k in BEHAVIOR})
        # 이관: 옛 판(ready_*)의 학습 결과를 경유 자세로 옮긴다.
        if 'ready_roll' in saved and 'via3_roll' not in saved:
            r = {k: saved.get('ready_' + k, BEHAVIOR['via3_' + k])
                 for k in ('pitch', 'roll', 'yaw', 'elbow')}
            BEHAVIOR.update({
                'via1_pitch': 0.0, 'via1_roll': r['roll'],
                'via1_yaw': 0.0, 'via1_elbow': 0.0,
                'via2_pitch': r['pitch'], 'via2_roll': r['roll'],
                'via2_yaw': 0.0, 'via2_elbow': r['elbow'],
                'via3_pitch': r['pitch'], 'via3_roll': r['roll'],
                'via3_yaw': r['yaw'], 'via3_elbow': r['elbow']})
    except Exception:
        pass
    return dict(BEHAVIOR)


def save_behavior(params=None):
    data = dict(BEHAVIOR)
    if params:
        data.update({k: v for k, v in params.items() if k in BEHAVIOR})
        BEHAVIOR.update(data)
    with open(_BEHAVIOR_FILE, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


load_behavior()


def _lift_ready(world, frames=None, trace=None, steps=None, watch=None,
                target_name=None):
    """휴식 -> 준비 자세, 3단계로: 벌리고 -> 굽히고 -> 돌려 넣기.

    관절을 한꺼번에 보간하면 전완이 테이블 앞모서리를 스친다(실측 -5cm).
    사람이 하듯 팔을 옆으로 벌려 테이블 옆면 밖에서 굽힌 뒤 위에서 돌려
    넣으면 경로 최소 여유 9.5cm, 투과 0 이다.
    """
    vias = via_poses()
    if steps is None:
        steps = int(BEHAVIOR['lift_steps'])
    cur = {j: world.pose.get(j, 0.0) for j in vias[0]}
    for tgt in vias:
        for k in range(1, steps + 1):
            u = k / float(steps)
            wgt = u * u * (3 - 2 * u)           # smoothstep
            pose = {j: cur[j] + (tgt[j] - cur[j]) * wgt for j in cur}
            if not sv.in_bounds({**world.pose, **pose}):
                continue
            world.set_arm(pose)
            # 손만 쫓으면 목표가 시야에서 사라진다 - 서보가 눈 뷰로 일해야
            # 하니, 목표를 알면 손-목표 중간을 보며 둘 다 화면에 담는다.
            h = world.hand()
            aim = h if watch is None else tuple(
                (h[i] + watch[i]) / 2.0 for i in range(3))
            world.nudge_gaze(aim, min(BEHAVIOR['gaze_step_deg'], 6.0))
            if trace is not None:
                trace.append(_trace_entry(world, watch, target_name))
            if frames is not None:
                frames.append(_snap(world, 'lift'))
        cur = tgt
    return True


def _trace_entry(world, watch=None, target=None, carried=False, ret=False):
    # 대상 물체는 따로 잰다: 집기·접근은 대상에 '닿는' 게 목표라, 대상
    # 접촉을 충돌로 세면 다가가는 것 자체가 벌점이 된다 (사용자 지적).
    # 테이블·다른 물체만 회피 대상이고, 대상은 관통(-1cm 초과)만 금지.
    ignore = ('table', target) if target else ('table',)
    e = {'table': world.clearance_of('table'),
         'obj': world.clearance(ignore=ignore)[0],
         'hand': tuple(round(v, 4) for v in world.hand()),
         'joints': {j: round(world.pose.get(j, 0.0), 2)
                    for j in sv.SERVO_JOINTS}}
    if target and not carried and not CONTACT_OK:
        # carried(운반 중)·CONTACT_OK(탐침~닫기)면 재지 않는다: 잡힌
        # 물체는 손에 붙어 있고, 조임·탐침 접촉은 파지 그 자체다 (사용자:
        # 집으면서 나는 충돌은 괜찮다). 접근·하강 중 들이받기만 잡는다.
        # 조는 뺀다: 하강·호핑에서 조가 물체 옆면을 스치며 감싸는 건
        # 파지의 일부다 (조-물체는 파지 판정·밀어냄·과조임이 감시).
        # 여기 남는 건 손목/팔뚝으로 들이받기.
        e['tgt'] = world.clearance_of(target, exclude_jaws=True)
    if ret:
        e['ret'] = 1        # 복귀 구간: 품질(효율 등)에선 빼고 충돌 검사엔 넣는다
    if watch is not None:
        # 실물은 눈 뷰만 보고 동작한다 - 목표를 시야에서 잃으면 서보가
        # 눈이 먼다. 목표가 화면 안에 있었는지를 궤적에 남겨 품질로 잰다.
        px = world.pixel_of(watch)
        e['view'] = int(px is not None and 0 <= px[0] < world.width
                        and 0 <= px[1] < world.height)
    return e

def _servo(world, target, done_cm, noise_px=1.0, steps=45, seed=0,
           stop_gap=None, obj_stop=None, frames=None, note='', trace=None,
           table_min=None, target_name=None, ignore_objs=(), target_stop=None,
           on_step=None, step_cap=None, carried=False, ret=False,
           stall_slack=3.0, grasping=False):
    """화면 오차로 손을 target 까지. sim_servo 와 같은 방식, World 위에서."""
    import numpy as np

    rng = np.random.RandomState(seed)
    SIZE_GAIN = 220.0
    mj = world.mujoco
    if table_min is None:
        table_min = BEHAVIOR['table_min']
    step_deg = BEHAVIOR['step_deg']
    if step_cap is not None:
        step_deg = min(step_deg, step_cap)   # 접촉 직전 하강 등 정밀 구간
    gain = BEHAVIOR['gain']
    damping = BEHAVIOR['damping']

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
    dist_hist = []
    recovered = False
    for step in range(steps):
        # 머리는 손과 물체의 중간을 본다 - 손이 다가갈수록 물체로 수렴.
        # 실물 visual servoing 도 손과 목표가 같이 보여야 오차를 잰다.
        h = world.hand()
        aim = tuple((h[i] + target[i]) / 2.0 for i in range(3))
        world.nudge_gaze(aim, min(BEHAVIOR['gaze_step_deg'], 6.0))
        if trace is not None:
            trace.append(_trace_entry(world, target, target_name,
                                      carried=carried, ret=ret))
        if frames is not None:
            frames.append(_snap(world, '%s %.1fcm' % (note,
                          me._dist(world.hand(), target) * 100)))
        d_now = me._dist(world.hand(), target) * 100
        if d_now <= done_cm:
            return True
        # 진행 감시: 최근 8스텝간 1cm 도 못 좁혔으면 정체다. 45스텝을
        # 소진하며 '미도달' 로 끝나는 대신, 한 번은 준비 자세 쪽으로
        # 물러났다가 다른 각도로 재진입하고, 또 정체면 일찍 포기한다
        # (밖의 재시도 층이 재인지부터 다시 하게).
        dist_hist.append(d_now)
        if len(dist_hist) >= 10 and dist_hist[-10] - d_now < 1.0:
            # 이미 코앞이면 '튕기며' 재진입하지 않는다 - 그냥 멈춘다
            # (가드가 마지막 mm 를 막는 상황: 물러나도 소용없다).
            if recovered or d_now <= done_cm + stall_slack:
                if frames is not None:
                    frames.append(_snap(world, '%s 정체' % note))
                return False
            recovered = True
            dist_hist = []
            via = via_poses()[-1]
            cur_pose = {j: world.pose.get(j, 0.0) for j in via}
            for k in range(1, 6):
                u = k / 5.0
                wgt = u * u * (3 - 2 * u)
                saved_bk = dict(world.pose)
                world.set_arm({j: cur_pose[j] + (via[j] - cur_pose[j])
                               * 0.35 * wgt for j in via})
                # 백오프 호가 대상을 긁으면 그 스텝은 물리고 그냥 멈춘다
                # (물러나다 물체를 관통하는 게 최악이다).
                if (target_name
                        and world.clearance_of(target_name) <= -0.4):
                    world.set_arm(saved_bk)
                    return me._dist(world.hand(), target) * 100 <= done_cm
                world.nudge_gaze(target, min(BEHAVIOR['gaze_step_deg'], 6.0))
                if trace is not None:
                    trace.append(_trace_entry(world, target, target_name))
                if frames is not None:
                    frames.append(_snap(world, '%s 다시 접근' % note))
            continue
        if stop_gap is not None and world.clearance()[0] <= stop_gap:
            return True
        if obj_stop is not None and world.clearance(
                ignore=('table',) + tuple(ignore_objs))[0] <= obj_stop:
            return True                     # 물체에 닿기 직전 - 여기서 멈춘다
        if (target_stop is not None and target_name
                and world.clearance_of(target_name) <= target_stop):
            return True                     # 대상 접촉 - 잡기는 여기가 목적지
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
        dq, *_ = np.linalg.lstsq(J.T @ J + damping * np.eye(len(sv.SERVO_JOINTS)),
                                 J.T @ (-np.array(e)), rcond=None)
        # 사람 팔처럼 목표 근처에서 감속한다 - 오버슛이 줄고 눈에도
        # 자연스럽다 (종 모양 속도 프로파일의 싼 근사).
        slow = max(0.35, min(1.0, me._dist(world.hand(), target) / 0.15))
        dq = np.clip(dq * gain * slow, -step_deg, step_deg)
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
        tab = world.clearance_of('table', exclude_jaws=grasping)
        obj = world.clearance(ignore=('table',) + tuple(ignore_objs))[0]
        if obj_stop is not None and obj < obj_stop:
            world.set_arm(saved)
            return True                       # 물체에 닿기 직전 - 여기서 멈춘다
        if (target_stop is not None and target_name
                and world.clearance_of(target_name) <= target_stop):
            world.set_arm(saved)              # 넘어선 스텝은 되돌리고 멈춤
            return True
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
        if on_step is not None:
            on_step()
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


def _confirm_result(world, point, frames=None, note='결과 확인'):
    """임무 끝에 결과 지점을 응시해 확인한다 - 허공을 보며 끝나지 않게."""
    if point is None:
        return
    world.glide_gaze(point[1], point[2], x=max(point[0], 0.05),
                     max_step_deg=min(BEHAVIOR['gaze_step_deg'], 6.0),
                     on_step=(lambda i, n: frames.append(
                         _snap(world, note))) if frames is not None else None)
    world.center_on(point)
    if frames is not None:
        frames.append(_snap(world, note))


def _retreat(world, frames=None, trace=None, target_name=None):
    """임무 후 복귀: 경유 통로를 역순으로 되짚어 휴식 자세로.

    닿는 순간 영상이 뚝 끝나는 게 어색하다는 지적(사용자)에서 나왔다.
    실물도 동작이 끝나면 팔을 거둬들인다. 갈 때 검증된 통로(경유 자세)를
    거꾸로 지나므로 안전하고, 복귀 중 충돌도 trace 로 계속 검사된다.
    """
    vias = via_poses()
    rest = {j: 0.0 for j in vias[0]}
    steps = max(4, int(BEHAVIOR['lift_steps']) // 2)
    cur = {j: world.pose.get(j, 0.0) for j in vias[0]}
    # 이탈: 물체 위에서 경유 자세로 4관절을 한꺼번에 보간하면 손이 낮게
    # 쓸며 작업 영역을 가로질러 방금 놓은 물체를 관통한다 (실측 -5cm).
    # 팔꿈치·어깨를 먼저 접어 손을 위로 거둬들인 '접힘' 자세를 경유하고,
    # 그 다음 통로(경유 역순)를 탄다. 서보(화면 기반)는 여기서 못 쓴다 -
    # 이탈 목표가 화면 밖이면 무동작이 된다.
    # 접기 전에 가드 있는 서보로 수직 이탈 - 방금 놓은 물체 위 안전
    # 고도까지. (접힘 보간은 개루프라 물체를 감지하지 못한다.)
    _set_gripper(world, 0.0)               # 그리퍼 중립(0도)으로 복귀
    h = world.hand()
    # 위로만 빼면 이어지는 스윙 호가 물체 위치에 따라 물체를 지난다 -
    # 몸쪽으로 당기며 올려 호 전체를 물체에서 떼어낸다.
    _servo(world, (h[0] - 0.08, h[1], h[2] + 0.16), 5.0, steps=12,
           step_cap=4.0, frames=frames, note='clear', trace=trace,
           target_name=target_name, ret=True)
    # 접근의 정확한 역순: 이탈(위) -> 옆으로 스윙아웃(롤만) -> 통로 역순.
    # 팔꿈치를 먼저 접으면 전완이 방금 놓은 물체를 휘두르며 지나간다.
    cur = {j: world.pose.get(j, 0.0) for j in vias[0]}
    swing = dict(cur)
    swing['right_arm.shoulder_roll'] = vias[-1]['right_arm.shoulder_roll']
    for tgt in [swing] + list(reversed(vias)) + [rest]:
        for k in range(1, steps + 1):
            u = k / float(steps)
            wgt = u * u * (3 - 2 * u)
            pose = {j: cur[j] + (tgt[j] - cur[j]) * wgt for j in cur}
            if not sv.in_bounds({**world.pose, **pose}):
                continue
            saved_ret = dict(world.pose)
            world.set_arm(pose)
            # 복귀 충돌 감시(사용자): 돌아가는 길에 물체를 치면 그 스텝을
            # 물리고 위로 접어 피한 뒤 계속 간다.
            if world.clearance(ignore=('table',))[0] < 0.0:
                world.set_arm(saved_ret)
                INCIDENTS.append('복귀 경로가 물체를 스침 - 회피 접기')
                fold_up = dict(world.pose)
                fold_up['right_arm.elbow_pitch'] = vias[-1][
                    'right_arm.elbow_pitch']
                world.set_arm(fold_up)
            # 복귀 중에도 팔을 본다 - 사람도 팔을 거둘 때 눈이 따라온다.
            world.nudge_gaze(world.hand(), min(
                BEHAVIOR['gaze_step_deg'], 6.0))
            if trace is not None:
                trace.append(_trace_entry(world, None, target_name, ret=True))
            if frames is not None:
                frames.append(_snap(world, 'return'))
        cur = tgt
    # 팔이 휴식에 닿았다 - 시선은 '테이블이 보이는' 기본 응시로.
    # 순정면(수평선)은 빈 허공만 보인다 (스트립 7~8 프레임 암전).
    world.glide_gaze(0.0, world.table_top + 0.05, x=0.45,
                     max_step_deg=min(BEHAVIOR['gaze_step_deg'], 6.0),
                     on_step=(lambda i, n: frames.append(
                         _snap(world, '기본 응시')))
                     if frames is not None else None)
    return True


def execute(world, plan, frames=None, trace=None, check=None, retreat=True):
    """계획을 월드에서 실행한다. (성공했다고 주장하지 않음 - 검증은 따로)"""
    mode = plan.get('mode')
    if mode == 'gaze':
        # 시선은 돌려서 간다 - 순간이동하면 눈 뷰가 프레임 사이에 뚝 바뀐다.
        p = plan['point']
        cb = None
        if frames is not None:
            cb = lambda i, n: frames.append(_snap(world, 'gaze %d/%d' % (i, n)))
        world.glide_gaze(p[1], p[2], x=max(p[0], 0.05), on_step=cb,
                         max_step_deg=BEHAVIOR['gaze_step_deg'])
        world.center_on(p)          # 미세 센터링 (잔여 오차만)
        if frames is not None:
            frames.append(_snap(world, 'gaze'))
        return True
    if mode == 'reach':
        # 실물 순서: 팔을 먼저 테이블 위로 들어 올리고, 접근 축을 정해
        # 예비점을 거쳐 진입한다. 파지는 '어디로 어떻게 들어가느냐' 가
        # 반이다 - 위 접근은 예비점이 대상 위, 옆 접근은 로봇 쪽 옆.
        tname = plan.get('target')
        pt = plan['point']
        _lift_ready(world, frames=frames, trace=trace, watch=pt,
                    target_name=tname)
        appr = plan.get('approach')
        if appr in ('top', 'side'):
            if appr == 'top':
                pre = (pt[0], pt[1], pt[2] + 0.07)
            else:
                n = math.hypot(pt[0], pt[1]) or 1.0
                pre = (pt[0] * (1 - 0.07 / n), pt[1] * (1 - 0.07 / n),
                       pt[2] + 0.02)
            _servo(world, pre, 5.0, obj_stop=1.5, steps=25,
                   frames=frames, note='pre-%s' % appr, trace=trace,
                   target_name=tname)
        ok = _servo(world, pt, plan['done_cm'], obj_stop=1.0,
                    frames=frames, note='reach', trace=trace,
                    target_name=tname)
        # 성공 판정은 복귀 전에 잰다 - 복귀하면 손이 멀어진다.
        if check is not None:
            ok = bool(check())
        _confirm_result(world, world.object_pos(tname) if tname else pt,
                        frames=frames)
        if retreat:
            _retreat(world, frames=frames, trace=trace, target_name=tname)
        return ok
    if mode == 'lift_hold':
        # 잡아서 들어 올려 보이기. 성공은 들린 순간(check) 재고, 그 뒤
        # 제자리에 내려놓고 복귀한다.
        tname = plan.get('target')
        watch = (world.object_pos(plan['object']) if plan.get('object')
                 else plan.get('point'))
        _lift_ready(world, frames=frames, trace=trace, watch=watch,
                    target_name=tname)
        bind, half, ok0, start = _grasp(world, obj=plan.get('object'),
                                        point=plan.get('point'),
                                        frames=frames, trace=trace)
        if ok0:
            h0 = world.hand()
            _servo(world, (h0[0], h0[1], h0[2] + 0.15), 3.0, steps=20,
                   ignore_objs=(bind,), carried=True,
                   step_cap=4.0, frames=frames, note='들어올림',
                   trace=trace, target_name=bind, seed=7)
            if frames is not None:
                frames.append(_snap(world, '들었다'))
        ok = bool(check()) if check is not None else ok0
        _confirm_result(world, world.object_pos(bind), frames=frames)
        if ok0:
            _servo(world, (start[0], start[1], start[2] + half + 0.03), 3.0,
                   steps=15, ignore_objs=(bind,),
                   carried=True, step_cap=4.0, frames=frames,
                   note='내려놓기', trace=trace, target_name=bind, seed=8)
            _set_gripper(world, GRIP_OPEN, frames=frames, note='그리퍼 벌림')
            world.step_physics(80)
            if frames is not None:
                frames.append(_snap(world, '내려놓음'))
        if retreat:
            _retreat(world, frames=frames, trace=trace, target_name=tname)
        return ok

    if mode == 'pick':
        watch = (world.object_pos(plan['object']) if plan.get('object')
                 else plan.get('point'))
        _lift_ready(world, frames=frames, trace=trace, watch=watch,
                    target_name=plan.get('target'))
        ok = _pick(world, obj=plan.get('object'), tray=plan.get('tray'),
                   point=plan.get('point'),
                   tray_point=plan.get('tray_point'),
                   frames=frames, trace=trace)
        if check is not None:
            ok = bool(check())
        tgt = plan.get('target') or plan.get('object')
        _confirm_result(world, world.object_pos(tgt) if tgt else None,
                        frames=frames)
        if retreat:
            _retreat(world, frames=frames, trace=trace,
                     target_name=plan.get('target'))
        return ok
    return False


def _reaim(world, guess, kind=None, frames=None, note='re-aim',
           want_kind=False):
    """추정점 근처를 다시 보고 검출로 정제한다 (카메라만 - 참값 없음).

    가까울수록 물체가 화면에 크게 잡혀 크기 단서(깊이)가 좋아진다.
    kind 를 주면 그 종류의 검출만 후보로 쓴다 (쟁반 등).
    """
    import random as _random
    import sim_perceive as P
    kinds = (kind,) if isinstance(kind, str) else kind
    # 순간이동 금지: center_on 을 바로 부르면 목이 한 번 '튕긴다'.
    world.glide_gaze(guess[1], guess[2], x=max(guess[0], 0.05),
                     max_step_deg=BEHAVIOR['gaze_step_deg'])
    world.center_on(guess)
    dets = P.detect(world, 1.5, 0.0, _random.Random(5))
    px = world.pixel_of(guess)
    if not dets or px is None:
        return (guess, None) if want_kind else guess
    pool = [d for d in dets if kinds is None or d['kind'] in kinds] or dets
    det = min(pool, key=lambda d: (d['u'] - px[0]) ** 2
              + (d['v'] - px[1]) ** 2)
    if frames is not None:
        frames.append(_snap(world, note))
    pt = _det_to_3d(world, det)
    return (pt, det['kind']) if want_kind else pt


GRIP_OPEN = -50.0       # 실물 규약: 음수=벌림, 양수=다묾 (range -50..10)
GRIP_SHUT = 10.0

# 관찰자용 사건 기록부. 에피소드 시작 때 비우고, 절차가 이상 징후를
# 남기면 여기 쌓인다 - 개선 루프가 지적 원장으로 자동 승격한다.
INCIDENTS = []
GRASP_REPORT = {}       # 마지막 파지 시도의 계측 (원인 판정용)
CONTACT_OK = False      # 의도된 접촉 구간(탐침~닫기) - 대상 관통 검사 면제


def _close_on(world, bind, half, frames=None, trace=None):
    """조임 공용 절차. (잡힘, 사유) 반환. 안전 규칙:

    1. 빈손 완폐 금지: 0도(중립)까지 접촉이 없으면 멈추고 실패 -
       실물 force gripper 를 허공에서 끝까지 조이면 모터가 상한다.
    2. 감쌈 검사: 물체 '중심' 이 조 스팬 안(수평 2.2cm, 높이 창)이어야
       잡힘이다 - 조 끝이 모서리를 스친 접촉은 잡힘이 아니다 (자석 방지).
       물리(무엇이 실제로 잡히나)의 대리 판정이라 참값 사용이 정당하다.
    3. 힘-완화: 파고들었으면 접촉 수준까지 되푼다 (과조임 = 고장).
    """
    start_op = world.object_pos(bind)      # 밀어냄 검사 기준
    g = GRIP_OPEN
    for _ in range(24):
        # 실물 force gripper 처럼: '양쪽이 닿을 때까지' 조인다. 움직 조가
        # 물체를 밀어 고정 조에 붙이는 과정이 파지다 - 한쪽 접촉에서
        # 멈추면 물체가 슬롯 안에서 헛돈다 (실측: 고정 쪽 1.1cm 빈 채
        # 맞물림 실패). 물리 정착 스텝이 미끄러짐을 처리한다.
        c = world.jaw_clearance(bind, jaw='moving')
        c_f = world.jaw_clearance(bind, jaw='fixed')
        if c <= -0.05 and c_f <= 0.2:
            break
        if c <= -0.35:
            break                            # 한쪽만 깊이 파고듦 - 중단
        if g >= 0.0 and c > 0.6:
            INCIDENTS.append('허공 조임: 중립(0도)까지 접촉 없음 - 중단')
            if frames is not None:
                frames.append(_snap(world, '허공 조임 중단'))
            return False, '허공 조임 중단'
        if g >= GRIP_SHUT:
            break
        g = min(GRIP_SHUT, g + (3.0 if c < 1.0 else 6.0))
        world.set_arm({'right_arm.hand.gripper': g})
        world.step_physics(10)               # 물체가 밀려 자리잡을 시간
        if frames is not None:
            frames.append(_snap(world, '그리퍼 조임'))
    for _ in range(8):
        if world.jaw_clearance(bind, jaw='moving') >= -0.30 or g <= GRIP_OPEN:
            break                            # 과조임(-0.3cm 초과)만 되풀기
        g -= 1.0
        world.set_arm({'right_arm.hand.gripper': g})
        if frames is not None:
            frames.append(_snap(world, '조임 완화'))
    slot = world.grip_slot()
    op = world.object_pos(bind)
    enclosed = (math.hypot(slot[0] - op[0], slot[1] - op[1]) <= 0.025
                and -0.015 <= slot[2] - (op[2] + half) <= 0.055)
    pinched = (g < GRIP_SHUT
               and world.jaw_clearance(bind, jaw='moving') <= 0.05
               and world.jaw_clearance(bind, jaw='fixed') <= 0.30)
    if pinched and not enclosed:
        INCIDENTS.append('모서리 접촉을 잡힘으로 오인할 뻔 - 감쌈 검사가 거부')
    grabbed = enclosed and pinched
    # 파지 시도가 물체를 밀어냈으면 관찰자 신호 + 실패 시 손을 빼서
    # 더 끌고 다니지 않는다 (밀림이 '잡혀 오는' 것처럼 보인다).
    moved = me._dist(world.object_pos(bind), start_op) * 100
    if moved > 2.0:
        INCIDENTS.append('파지 시도가 물체를 %.0fcm 밀어냄' % moved)
    if not grabbed:
        _nudge_hand(world, (0.0, 0.0, 0.06), target_name=bind,
                    frames=frames, trace=trace, note='손 빼기')
    if trace is not None:
        trace.append(_trace_entry(world, None, bind))
    if frames is not None:
        frames.append(_snap(world, '잡기 %s' % ('성공' if grabbed else '실패')))
    why = None if grabbed else ('감쌈 안 됨' if pinched else '맞물림 안 됨')
    GRASP_REPORT.update(closed=g > GRIP_OPEN + 1.0, grabbed=grabbed,
                        why=why,
                        squeeze_cm=round(-min(0.0, world.jaw_clearance(bind)),
                                         2))
    return grabbed, why


def _set_gripper(world, deg, frames=None, note=None, steps=3):
    """그리퍼 각도(도)를 몇 스텝에 나눠 - 영상에 여닫힘이 보이게."""
    cur = world.pose.get('right_arm.hand.gripper', 0.0)
    for k in range(1, steps + 1):
        g = cur + (deg - cur) * k / steps
        world.set_arm({'right_arm.hand.gripper': g})
        if frames is not None and note:
            frames.append(_snap(world, note))


ORIENT_JOINTS = ('right_arm.hand.forearm_yaw',
                 'right_arm.hand.wrist_pitch', 'right_arm.arm_yaw')


def _nudge_hand(world, dxyz, target_name=None, substeps=5, frames=None,
                trace=None, note='미세이동'):
    """고유수용감각 기반 미세 이동: 손을 정확히 dxyz(m) 만큼 옮긴다.

    화면-오차 서보는 가드·정체감시에 걸려 cm 급 잔차가 남는다. 자기 손
    위치는 FK 로 정확하니, 위치 자코비안(근위 4관절 수치미분) 최소제곱
    으로 관절 변화를 풀어 잘게 나눠 적용한다. 테이블 가드는 유지.
    """
    import numpy as np
    for k in range(substeps):
        cur = me.forward_kinematics(me.CHAINS['right_arm'], world.pose)
        J = []
        for j in sv.SERVO_JOINTS:
            trial = dict(world.pose); trial[j] = world.pose.get(j, 0.0) + 1.0
            p2 = me.forward_kinematics(me.CHAINS['right_arm'], trial)
            J.append([p2[i] - cur[i] for i in range(3)])
        J = np.array(J).T                       # 3 x 4 (m per deg)
        want = np.array(dxyz) / substeps
        # 최소노름 해 - 감쇠 0.02는 J^2(~1e-5, m/도)보다 커서 해를 0으로
        # 누른다 (실측: 요청 9cm 이동 0cm). 클램프가 안전을 맡는다.
        dq, *_ = np.linalg.lstsq(J, want, rcond=None)
        dq = np.clip(dq, -3.0, 3.0)
        pose = dict(world.pose)
        for j, dv in zip(sv.SERVO_JOINTS, dq):
            lo, hi = me.ALL_LIMITS[j]
            pose[j] = min(hi, max(lo, pose.get(j, 0.0) + float(dv)))
        if not sv.in_bounds(pose):
            return False
        saved = dict(world.pose)
        world.set_arm(pose)
        # 조 제외 - 파지 정렬 중 조 끝이 테이블에 붙는 건 정상이다.
        # (조 포함으로 재면 낮은 물체에서 모든 하위스텝이 되돌려져
        # IK 가 무동작이 된다 - 정렬 잔차 6~17cm 의 주범.)
        if world.clearance_of('table', exclude_jaws=True) < 0.15:
            world.set_arm(saved)
            return False
        if trace is not None:
            trace.append(_trace_entry(world, None, target_name))
        if frames is not None:
            frames.append(_snap(world, note))
    return True


def _level_pinch(world, target, frames=None, max_iter=12):
    """파지 자세 추정: 집게면이 수평이 되는 손목 방향을 찾는다.

    물체 자세(테이블 위에 서 있음)가 요구하는 그리퍼 방향은 '수평
    집게' 다. wrist_roll 이 없는 이 팔에서도 forearm_yaw·wrist_pitch·
    arm_yaw(전부 실기 존재)로 손목 방향을 돌릴 수 있다 - 위치 서보는
    근위 4관절만 쓰므로 방향(원위)과 위치(근위)가 분리 제어된다.
    유령 프로브(물리 교란 없음) 좌표 하강.
    """
    def cost(delta):
        tilt, slot = world.ghost_eval(delta)
        drift = math.hypot(slot[0] - target[0], slot[1] - target[1])
        # 이동 벌점을 세게 - 120 이면 기울기 몇 도와 슬롯 10cm 후퇴를
        # 맞바꿔, 만회 불가능한(도달 한계·금지영역) 자세를 고른다 (실측:
        # book 슬롯 9.8cm 후퇴 -> IK 첫 스텝부터 거부).
        return tilt + drift * 350.0, tilt

    cur = {j: world.pose.get(j, 0.0) for j in ORIENT_JOINTS}
    best_c, best_t = cost({})
    for it in range(max_iter):
        step = (6.0, 3.0, 1.5)[min(it // 4, 2)]
        improved = False
        for j in ORIENT_JOINTS:
            lo, hi = me.ALL_LIMITS[j]
            for d in (step, -step):
                trial = dict(cur)
                trial[j] = max(lo, min(hi, cur[j] + d))
                c, t = cost(trial)
                if c < best_c - 0.05:
                    best_c, best_t, cur = c, t, trial
                    improved = True
        if not improved or best_t < 10.0:
            break
    world.set_arm(cur)
    if frames is not None:
        frames.append(_snap(world, '집게 수평화 %.0f도' % world.pinch_tilt()))
    return world.pinch_tilt()


def _probe_band(world, bind, frames=None, trace=None, max_steps=4):
    """닫기 전 접촉대 탐침: 조 틈이 물체를 실제로 감쌀 때까지 미세 하강.

    정렬 게이트는 xy·z 창만 보는데, z 창 상단에서 닫으면 조 끝이 물체
    윗면을 스치기만 한다 (맞물림 실패의 한 갈래). 조-물체 틈이 좁아질
    때까지 0.6cm 씩 내려가 접촉대 안에서 닫게 한다.
    """
    for _ in range(max_steps):
        gap = world.jaw_clearance(bind)     # cm 단위다 (m 아님!)
        if gap <= 0.6:
            break                           # 접촉대 도달
        before = world.grip_slot()[2]
        _nudge_hand(world, (0.0, 0.0, -0.006), target_name=bind,
                    frames=frames, trace=trace, note='접촉대 탐침')
        if world.jaw_clearance(bind) < -0.2:
            # 지나쳤다 - 한 스텝 되올리고 끝 (관통 충돌 방지)
            _nudge_hand(world, (0.0, 0.0, 0.006), target_name=bind,
                        frames=frames, trace=trace, note='탐침 되올림')
            break
        if abs(world.grip_slot()[2] - before) < 0.001:
            break                           # 가드/한계로 더 못 내려감
    GRASP_REPORT['probe_gap_cm'] = round(world.jaw_clearance(bind), 2)


def _grasp(world, obj=None, point=None, frames=None, trace=None):
    """접근 -> 재조준 -> 저속 접촉 잡기. (bind, half, 성공여부) 반환.

    잡기는 pick 과 lift 가 공유한다. 파지 지점은 물체 윗면.
    """
    cup = world.object_pos(obj) if obj else tuple(point)
    bind = obj

    def _half_of(name):
        for o in world.objects:
            if o['name'] == name:
                if o['type'] == 'cylinder':
                    return o['size'][1]
                if o['type'] == 'sphere':
                    return o['size'][0]
                return o['size'][-1]
        return 0.03

    # 예비점은 '윗면' 기준이어야 한다. 중심+10cm 고정이면 병처럼 키 큰
    # 물체는 예비점이 꼭대기 바로 위라 손 캡슐이 이미 관통한다.
    half = _half_of(bind) if bind else 0.06
    # 그리퍼는 접근 '전에' 벌린다 - 닫힌 손가락으로 내려가면 손끝이 이미
    # 물체 실루엣 안에 든 채 벌어지며 옆면을 관통한다 (병에서 실측 -1.9).
    _set_gripper(world, GRIP_OPEN, frames=frames, note='그리퍼 벌림')
    pre = (cup[0], cup[1], cup[2] + half + 0.075)
    _servo(world, pre, 5.0, obj_stop=1.5, steps=25, target_stop=-0.5,
           grasping=True, frames=frames, note='pre-top', trace=trace,
           target_name=bind, seed=1)
    _level_pinch(world, cup, frames=frames)   # 파지 자세: 집게면 수평화
    if bind is None:
        # 브로커 경로: 접지 추정은 몇 cm 틀릴 수 있다 - 가까이서 다시
        # 보고 정제한다 (참값이 아니라 카메라 재검출).
        cup = _reaim(world, cup, frames=frames)
        bind = min(world.movable_objects(),
                   key=lambda n: me._dist(world.object_pos(n), cup))
        half = _half_of(bind)
    # 그리퍼를 벌리고 내려간다 - 손가락 사이에 물체가 들어오게 손바닥이
    # 물체 윗면 근처까지. 그 다음 손가락을 단계적으로 조여 접촉에서 멈춘다
    # (실물 force gripper 가 힘 센서로 멈추는 것의 기하 근사).
    # 하강 깊이는 손바닥 위치로 정한다 (자기 손 = 고유수용감각).
    # target_stop(대상 여유)으로 멈추면 굵은 물체는 손가락이 옆을 스치는
    # 순간 일찍 멈춰, 손가락이 물체 '위' 허공에서 닫힌다 (캔·블록 실패).
    # target_stop=-0.5: 기울어진 손가락이 옆면을 긁기 시작하면(경미한
    # 관통) 즉시 멈춘다 - 판정 한도(-1cm) 안에서 끊는 안전 정지.
    # 주의: 키 큰 물체(병)는 이 팔 기하에서 전완이 기대는 경우가 있다
    # (-1.9cm, 깊이와 무관 실측) - 판정이 충돌로 벌점하고, 회피 자세는
    # 경유 학습의 몫이다. 파지 깊이를 얕추면 작은 물체를 놓친다.
    def _descend(pt):
        # 정렬 기준은 '조 슬롯 중심' 이다 - 손목점을 물체에 맞추면 슬롯이
        # 비껴가 고정 조만 닿는다 (실측). 슬롯-손목 오프셋을 보정해 슬롯이
        # 물체 위에 오도록 서보 목표를 잡는다.
        h = world.hand(); slot = world.grip_slot()
        off = (h[0] - slot[0], h[1] - slot[1], h[2] - slot[2])
        _servo(world, (pt[0] + off[0], pt[1] + off[1],
                       pt[2] + half * 0.4 + off[2]), 1.2, steps=30,
               obj_stop=1.0, ignore_objs=(bind,), target_stop=-0.5,
               step_cap=2.5, stall_slack=0.6, grasping=True, frames=frames,
               note='descend', trace=trace, target_name=bind, seed=2)

    def _aligned(pt):
        slot = world.grip_slot()
        return (math.hypot(slot[0] - pt[0], slot[1] - pt[1]) <= 0.014
                and pt[2] + half - 0.012 <= slot[2] <= pt[2] + half + 0.025)

    # 하강부터는 의도된 접촉 구간이다: 짧은 조로 깊게 감싸는 자세에서
    # 손 캡슐·손바닥이 물체 윗면에 ~1cm 겹치는 건 기하학적 필연 (실측
    # -1.0~-1.9cm 로 전부 '대상 관통' 오판). 이송·복귀 들이받기는 그대로.
    global CONTACT_OK
    CONTACT_OK = True
    _descend(cup)
    # 슬롯 잔차를 IK 미세이동으로 소거 + 고전 파지 트릭: 목표를 고정 조
    # 쪽으로 0.8cm 치우친다 - 움직 조가 눌렀을 때 물체가 고정 조에
    # 갇히게 (정중앙이면 기울어진 집게에서 미끄러져 도망간다, 실측 2cm).
    def _slot_goal():
        mjm = world._gid['right_arm_jaw_moving']
        mjf = world._gid['right_arm_jaw_fixed']
        d = world.data.geom_xpos[mjf] - world.data.geom_xpos[mjm]
        n = math.hypot(float(d[0]), float(d[1])) or 1.0
        return (cup[0] + 0.008 * float(d[0]) / n,
                cup[1] + 0.008 * float(d[1]) / n)
    for _round in range(2):
        for _k in range(3):
            gx, gy = _slot_goal()
            gz = cup[2] + half + 0.005      # 슬롯을 윗면 바로 위로 (실측 접촉대)
            slot = world.grip_slot()
            ex, ey, ez = gx - slot[0], gy - slot[1], gz - slot[2]
            if math.hypot(ex, ey) <= 0.006 and abs(ez) <= 0.012:
                break
            _nudge_hand(world, (ex, ey, ez), target_name=bind,
                        frames=frames, trace=trace, note='슬롯 교정')
        if world.pinch_tilt() <= 12.0:
            break
        _level_pinch(world, cup, frames=frames)   # 기울면 다시 수평화
    # 집게면 기울기: 조 분리 방향이 수평에서 벗어난 각도. 이 팔은
    # wrist_roll 이 없어 자세에 따라 집게가 대각으로 기운다 - 기울면
    # 세워진 물체를 물리적으로 못 문다 (실측 ~60도에서 전패). 학습이
    # '수평 집게가 되는 자세' 를 찾도록 신호로 남긴다.
    mjm = world._gid.get('right_arm_jaw_moving')
    mjf = world._gid.get('right_arm_jaw_fixed')
    if mjm is not None and mjf is not None:
        v = world.data.geom_xpos[mjm] - world.data.geom_xpos[mjf]
        import numpy as _np
        n = _np.linalg.norm(v) or 1.0
        tilt = abs(math.degrees(math.asin(abs(float(v[2])) / n)))
        if tilt > 35.0:
            INCIDENTS.append('집게면 기울어짐 %.0f도 - 이 자세로는 수평 파지 불가'
                             % tilt)

    if not _aligned(cup):
        # 위치가 안 잡혔으면 닫지 않는다 (사용자: 다 도달해서 위치를
        # 잡고 닫아라). 한 번 재조준·재하강하고, 그래도 아니면 실패 -
        # 엉뚱한 데서 닫는 것보다 정직한 실패가 낫다.
        if obj is None:
            cup = _reaim(world, cup, frames=frames, note='재조준')
        _descend(cup)
    if not _aligned(cup):
        slot = world.grip_slot()
        GRASP_REPORT.update(closed=False, grabbed=False,
                            align_cm=round(math.hypot(
                                slot[0] - cup[0], slot[1] - cup[1]) * 100, 1),
                            tilt_deg=round(world.pinch_tilt(), 0))
        if frames is not None:
            frames.append(_snap(world, '정렬 실패 - 닫지 않음'))
        CONTACT_OK = False
        return bind, half, False, world.object_pos(bind)

    orig = world.object_pos(bind)          # 원위치 (되돌려 놓기용)
    slot = world.grip_slot()
    GRASP_REPORT.update(align_cm=round(math.hypot(
        slot[0] - cup[0], slot[1] - cup[1]) * 100, 1),
        tilt_deg=round(world.pinch_tilt(), 0))
    try:
        grabbed, why = _grasp_contact(world, bind, half, cup,
                                      _aligned, frames=frames, trace=trace)
    finally:
        CONTACT_OK = False
    return bind, half, grabbed, orig


def _grasp_contact(world, bind, half, cup, _aligned,
                   frames=None, trace=None):
    """의도된 접촉 구간: 탐침 -> 닫기 -> (실패 시) 재파지 1회."""
    _probe_band(world, bind, frames=frames, trace=trace)
    grabbed, why = _close_on(world, bind, half, frames=frames, trace=trace)
    if not grabbed and why == '맞물림 안 됨':
        # 재파지 1회: 정렬은 맞았는데 맞물림만 실패 - 다시 벌리고 새
        # 물체 위치로 재정렬해 탐침부터 다시. (한 번만 - 시간 예산)
        _set_gripper(world, GRIP_OPEN, frames, '재파지 - 다시 벌림')
        cup = world.object_pos(bind)
        for _k in range(2):
            # 목표는 '지금' 물체 위치 + 고정 조 쪽 0.8cm 치우침
            mjm = world._gid['right_arm_jaw_moving']
            mjf = world._gid['right_arm_jaw_fixed']
            d = world.data.geom_xpos[mjf] - world.data.geom_xpos[mjm]
            n = math.hypot(float(d[0]), float(d[1])) or 1.0
            gx = cup[0] + 0.008 * float(d[0]) / n
            gy = cup[1] + 0.008 * float(d[1]) / n
            slot = world.grip_slot()
            _nudge_hand(world, (gx - slot[0], gy - slot[1],
                                cup[2] + half + 0.005 - slot[2]),
                        target_name=bind, frames=frames, trace=trace,
                        note='재파지 정렬')
        if _aligned(world.object_pos(bind)):
            _probe_band(world, bind, frames=frames, trace=trace)
            grabbed, why = _close_on(world, bind, half,
                                     frames=frames, trace=trace)
            GRASP_REPORT['regrasp'] = True
    return grabbed, why


def _drop_anim(world, bind, to_pos, frames=None, steps=3):
    """놓기/드롭을 가속 낙하로 - 순간이동하면 자석처럼 보인다."""
    p0 = world.object_pos(bind)
    for k in range(1, steps + 1):
        u = (k / steps) ** 2                    # 중력 흉내(가속)
        world.move_object(bind, tuple(p0[i] + (to_pos[i] - p0[i]) * u
                                      for i in range(3)))
        if frames is not None:
            frames.append(_snap(world, '낙하'))


def _hold_offset(world, bind):
    """잡은 뒤 손-물체 상대 오프셋과 따라오기 콜백."""
    h0 = world.hand()
    o0 = world.object_pos(bind)
    off = tuple(o0[i] - h0[i] for i in range(3))

    def follow():
        h = world.hand()
        world.move_object(bind, tuple(h[i] + off[i] for i in range(3)))
    return follow


def _pick(world, obj=None, tray=None, point=None, tray_point=None,
          frames=None, trace=None):
    """집어 목적지로: 잡기 -> 나르기 -> 놓기(쟁반) / 넣기(바구니)."""
    bind, half, grabbed, _orig = _grasp(world, obj=obj, point=point,
                                        frames=frames, trace=trace)
    if not grabbed:
        return False
    hand0 = world.hand()

    # 3) 들어서 나른다: 위로 뽑고 -> 목적지 위 -> 내려놓기.
    dest = world.object_pos(tray) if tray else (tuple(tray_point)
                                                if tray_point else None)
    if dest is None:
        return False
    # 목적지 종류: 바구니는 벽이 있어 테두리 위에서 넣어야 한다.
    dkind = None
    if tray:
        dkind = next((o['kind'] for o in world.objects
                      if o['name'] == tray), None)
    # 나를 때는 높이 든다 - 다른 물체 위를 지나가는 게 안전하다. 스텝도
    # 상한을 둬 obj_stop 을 한 걸음에 뚫고 지나가지 않게 한다.
    # 물리 운반: 부착 없음 - 조임 마찰이 지탱 못 하면 떨어진다 (정직).
    up = (hand0[0], hand0[1], hand0[2] + 0.14)
    _servo(world, up, 4.0, steps=20, ignore_objs=(bind,), carried=True,
           step_cap=4.0, frames=frames, note='lift-carry',
           trace=trace, target_name=bind, seed=3)
    if me._dist(world.hand(), world.object_pos(bind)) > 0.18:
        INCIDENTS.append('운반 중 낙하 (마찰 부족)')
        return False
    _servo(world, (dest[0], dest[1], dest[2] + 0.16), 4.0, steps=35,
           ignore_objs=(bind,), obj_stop=1.0, carried=True,
           step_cap=4.0, frames=frames, note='carry', trace=trace,
           target_name=bind, seed=4)
    if tray is None:
        # 목적지 접지도 오차가 크다(납작해서 크기 단서가 나쁨) - 위에서
        # 다시 보고 정제한 곳에 놓는다. 종류는 재검출이 알려준다.
        dest, dkind = _reaim(world, dest, kind=('tray', 'basket'),
                             frames=frames, note='re-aim dest',
                             want_kind=True)
        _servo(world, (dest[0], dest[1], dest[2] + 0.10), 3.0, steps=15,
               ignore_objs=(bind,), carried=True,
               frames=frames, note='place', trace=trace, target_name=bind,
               seed=5)

    # 4) 놓기. 바구니는 테두리(벽 높이) 위에서 안으로 떨어뜨린다 -
    # 옆에서 밀면 벽에 걸린다. 쟁반은 면 위에.
    if dkind == 'basket':
        hz = W.OBJECT_LIBRARY['basket']['size'][2]
        rim = dest[2] + hz * 2
        _servo(world, (dest[0], dest[1], rim + 0.10), 3.0, steps=15,
               ignore_objs=(bind,), carried=True,
               frames=frames, note='over-rim', trace=trace,
               target_name=bind, seed=6)
    # 놓기 = 벌리면 중력이 한다. 순간이동/가짜 낙하 없음.
    _set_gripper(world, GRIP_OPEN, frames=frames, note='그리퍼 벌림')
    world.step_physics(100)
    if frames is not None:
        frames.append(_snap(world, 'drop'))
    if frames is not None:
        frames.append(_snap(world, 'placed'))
    return True


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
