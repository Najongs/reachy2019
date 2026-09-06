"""opus 조종사(pilot) - 에피소드 '안에서' 보고-생각하고-명령하는 루프.

사용자가 짚은 단순한 구조 그대로다:
  1) 카메라로 물체를 잘 찾는 답변을 만들고     (눈 화면 + 검출 오버레이)
  2) 팔을 어떻게 움직일지 생각해 명령값을 주면  (의미 수준 명령 JSON)
  3) 결정론 프리미티브가 동작한다              (서보/조임/운반 - 기하와 안전)

VLA 가 아니다: opus 는 에피소드당 5~10번, 단계 결정만 한다. 좌표 계산과
충돌 가드는 전부 결정론이다. 파라미터 진화(sim_improve)와 달리 여기서는
관찰->교정이 '지금 즉시' 일어난다 - 정렬이 거부되면 그 사실이 다음
프롬프트로 들어가 opus 가 adjust 를 내린다.

opus 가 받는 것(인지 방화벽 유지): 눈 화면 + 검출 오버레이, 자기 손 위치
(FK = 고유수용감각), 그리퍼 각도, 직전 명령의 결과. 참값은 없다.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_agent as A                              # noqa: E402
import sim_perceive as P                           # noqa: E402
import motion_exec as me                           # noqa: E402

MAX_CMDS = 14

PILOT_PROMPT = """로봇 팔 조종사다. 로봇 눈 화면(번호 박스 = 검출)과 상태를 보고 다음 명령 '하나' 를 내린다.

임무: {instruction}

상태:
{state}

명령 (JSON 한 줄만):
{{"cmd": "...", "why": "짧게"}}
- "gaze"               대상이 화면에 없다 - 시선을 훑어 다시 찾는다
- "ground", "box": n   화면의 n번 박스가 대상이다 (처음/재탐색 후 반드시)
- "approach"           대상 위 예비 위치로 (들어올리기 포함)
- "descend"            내려가 손가락 사이에 담는다
- "adjust", "dx": cm, "dy": cm   손을 옆으로 미세 이동 (앞+/뒤-, 왼+/오-)
- "close"              그리퍼를 닫는다 (정렬됐을 때만 - 거부되면 상태에 사유가 온다)
- "lift"               잡은 채 들어올린다
- "carry", "box": n    n번 박스(쟁반/바구니) 위로 나른다
- "release"            내려놓는다/떨어뜨린다
- "done"               임무 완료 판단
- "abort"              불가능 판단

규칙: 안 보이면 gaze, 보이면 ground 부터. 정렬 거부가 오면 close 를 반복하지 말고 adjust 나 descend 로 고쳐라. JSON 밖 텍스트 금지."""


def _jpeg(img):
    import base64
    import io as _io
    from PIL import Image
    im = Image.fromarray(img).convert('RGB')
    if im.width > 768:
        im = im.resize((768, int(im.height * 768 / im.width)))
    buf = _io.BytesIO()
    im.save(buf, format='JPEG', quality=75)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def run_episode(world, task, client, frames=None, trace=None, noise_px=2.0,
                seed=0, check=None):
    """조종사 루프. (성공여부, 명령기록) 반환.

    check 는 done 시점(복귀 전)에 잰다 - lift 처럼 '들고 있는 순간' 이
    성공인 임무는 복귀 후 재면 무조건 미달이다.
    """
    import random
    rng = random.Random(seed)
    tname = task['target']                 # 계측(대상/비대상 분리)용 - 계획엔 안 씀
    st = {'grounded': None, 'dest': None, 'phase': '시작', 'last': '없음',
          'bind': None, 'half': 0.04, 'holding': False, 'orig': None}
    log = []

    def snap(note):
        if frames is not None:
            frames.append(A._snap(world, note))

    def state_text(dets):
        h = world.hand()
        lines = ['단계: %s' % st['phase'],
                 '직전 결과: %s' % st['last'],
                 '손 위치(자기 감각): x %.2f y %.2f z %.2f' % tuple(h),
                 '그리퍼: %.0f도 (음수=벌림)' % world.pose.get(
                     'right_arm.hand.gripper', 0.0),
                 '검출: %s' % (', '.join('[%d]%s' % (i, d['kind'])
                                        for i, d in enumerate(dets))
                               or '없음')]
        if st['grounded'] is not None:
            g = st['grounded']
            xy = math.hypot(h[0] - g[0], h[1] - g[1]) * 100
            lines.append('접지된 대상 추정: x %.2f y %.2f z %.2f (손과 %.1fcm)'
                         % (g[0], g[1], g[2], me._dist(h, g) * 100))
            lines.append('정렬(수평): %.1fcm - 1.4 이내여야 close 가능. '
                         '하강했고 정렬됐으면 바로 close 하라. adjust 는 '
                         '2회 이상 반복하지 마라.' % xy)
        return '\n'.join(lines)

    for step in range(MAX_CMDS):
        dets = P.detect(world, noise_px, 0.03, rng)
        img = P.overlay(world, dets)
        raw = client.ask_vision(
            PILOT_PROMPT.format(instruction=task['instruction'],
                                state=state_text(dets)),
            image=_jpeg(img))
        try:
            cmd = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
        except Exception:
            st['last'] = '명령 해석 불가'
            continue
        c = cmd.get('cmd')
        log.append({'cmd': c, 'why': cmd.get('why'), 'step': step})
        snap('조종: %s' % c)

        if c == 'gaze':
            mem = P.SceneMemory()
            P.sweep(world, mem, noise_px=noise_px, dropout=0.03, rng=rng)
            # 훑고 나서 가장 확실한 기억 쪽으로 시선을 되돌린다 - 안
            # 그러면 다음 프레임도 '검출 없음' 이라 조종사가 포기한다.
            best = max(mem.items, key=lambda it: it['conf'], default=None)
            if best is not None:
                P.gaze_toward(world, best)
            st['last'] = '훑기 완료. 기억: %s' % (mem.summary() or '없음')
            st['phase'] = '탐색'
        elif c == 'ground':
            b = cmd.get('box')
            if b is not None and 0 <= int(b) < len(dets):
                st['grounded'] = A._det_to_3d(world, dets[int(b)])
                st['last'] = '%d번 박스 접지' % int(b)
                st['phase'] = '접지됨'
            else:
                st['last'] = 'ground 실패: 박스 번호 없음/범위 밖'
        elif c == 'approach' and st['grounded']:
            g = st['grounded']
            A._lift_ready(world, frames=frames, trace=trace, watch=g,
                          target_name=tname)
            A._set_gripper(world, A.GRIP_OPEN, frames=frames, note='벌림')
            A._servo(world, (g[0], g[1], g[2] + st['half'] + 0.075), 5.0,
                     obj_stop=1.5, steps=25, target_stop=-0.5, frames=frames,
                     note='approach', trace=trace, target_name=tname)
            st['last'] = 'approach 완료'
            st['phase'] = '예비 위치'
        elif c == 'descend' and st['grounded']:
            g = st['grounded']
            A._servo(world, (g[0], g[1], g[2] + st['half'] + 0.030), 1.2,
                     steps=30, obj_stop=1.0, target_stop=-0.5, step_cap=2.5,
                     stall_slack=0.6, frames=frames, note='descend',
                     trace=trace, target_name=tname)
            h = world.hand()
            st['last'] = ('descend 완료 (추정 대상까지 수평 %.1fcm)'
                          % (math.hypot(h[0] - g[0], h[1] - g[1]) * 100))
            st['phase'] = '하강함'
        elif c == 'adjust' and st['grounded']:
            dx = max(-3.0, min(3.0, float(cmd.get('dx') or 0))) / 100.0
            dy = max(-3.0, min(3.0, float(cmd.get('dy') or 0))) / 100.0
            g = st['grounded']
            st['grounded'] = (g[0] + dx, g[1] + dy, g[2])
            h = world.hand()
            A._servo(world, (h[0] + dx, h[1] + dy, h[2]), 1.0, steps=8,
                     step_cap=2.0, frames=frames, note='adjust',
                     trace=trace, target_name=tname)
            st['last'] = 'adjust (%.1f, %.1f)cm' % (dx * 100, dy * 100)
        elif c == 'close':
            # 결정론 정렬 게이트가 심판한다 - 거부 사유가 다음 상태로 간다.
            bind = min(world.movable_objects(),
                       key=lambda n: me._dist(world.object_pos(n),
                                              st['grounded'] or world.hand()))
            half = 0.04
            for o in world.objects:
                if o['name'] == bind:
                    half = {'cylinder': o['size'][1], 'sphere': o['size'][0],
                            }.get(o['type'], o['size'][-1])
            g = st['grounded'] or world.object_pos(bind)
            h = world.hand()
            xy = math.hypot(h[0] - g[0], h[1] - g[1])
            if xy > 0.014:
                st['last'] = 'close 거부: 정렬 미달 (수평 %.1fcm > 1.4)' % (xy * 100)
                continue
            ok, why = A._close_on(world, bind, half, frames=frames,
                                  trace=trace)
            if ok:
                st.update(bind=bind, half=half, holding=True,
                          orig=world.object_pos(bind))
                st['last'] = 'close 성공 (잡음)'
                st['phase'] = '잡음'
            else:
                st['last'] = 'close 실패: %s' % why
        elif c == 'lift' and st['holding']:
            h = world.hand()
            A._servo(world, (h[0], h[1], h[2] + 0.15), 3.0, steps=20,
                     ignore_objs=(st['bind'],), carried=True,
                     step_cap=4.0, frames=frames, note='lift', trace=trace,
                     target_name=st['bind'])
            st['last'] = 'lift 완료'
            st['phase'] = '들었음'
        elif c == 'carry' and st['holding']:
            b = cmd.get('box')
            dest = None
            if b is not None and 0 <= int(b) < len(dets):
                dest = A._det_to_3d(world, dets[int(b)])
            if dest is None:
                st['last'] = 'carry 실패: 목적지 박스 없음'
                continue
            A._servo(world, (dest[0], dest[1], dest[2] + 0.16), 4.0, steps=35,
                     ignore_objs=(st['bind'],), obj_stop=1.0,
                     carried=True, step_cap=4.0, frames=frames, note='carry',
                     trace=trace, target_name=st['bind'])
            st['dest'] = dest
            st['last'] = 'carry 완료'
            st['phase'] = '목적지 위'
        elif c == 'release' and st['holding']:
            A._set_gripper(world, A.GRIP_OPEN, frames=frames, note='벌림')
            world.step_physics(100)          # 중력이 떨어뜨린다
            st['holding'] = False
            st['last'] = 'release 완료'
            st['phase'] = '놓음'
        elif c in ('done', 'abort'):
            break
        else:
            st['last'] = '%s 불가 (전제 미충족: %s)' % (c, st['phase'])

    ok = bool(check()) if check is not None else None
    if st['holding']:
        # 잡은 채 끝났으면 제자리에 내려놓는다 - 안 그러면 공중에 남는다.
        A._set_gripper(world, A.GRIP_OPEN, frames=frames, note='벌림')
        A._drop_anim(world, st['bind'], st['orig'], frames=frames, steps=2)
    A._retreat(world, frames=frames, trace=trace, target_name=tname)
    return ok, log
