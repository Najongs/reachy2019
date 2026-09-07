"""복합 평가 - 성공/실패만이 아니라 '얼마나 잘' 했는지를 잰다.

두 층이다:

  quality(trace)      결정론 품질 지표. 궤적 기록에서 계산:
                        efficiency  손이 간 길이 / 직선 거리 (1.0 이 최선)
                        smoothness  관절 속도의 변화(저크)가 작을수록 높음
                        margin      테이블·물체 여유의 평균 (여유 있게 돌았나)
                        steps       걸린 스텝 수
                      -> score 0~100. 같은 '성공' 이라도 거칠고 위험하게
                      스치며 간 동작과 부드럽게 돌아간 동작을 가른다.

  critique(client, frames, ...)   opus 가 프레임 3장(들어올림/중간/도달)과
                      지표를 보고 자연스러움을 비평한다. 결과는 JSON:
                      {naturalness 1~10, issues[], advice{파라미터: 방향}}.
                      advice 는 BEHAVIOR 손잡이에 대한 제안이고, 개선 루프가
                      객관 성공을 문지기로 세운 채 시험해 본다.

원칙: 성공/안전 판정은 여전히 결정론이다. 언어모델은 '품질' 만 말한다 -
모델이 성공을 판정하기 시작하면 자신 있게 틀린다(이 세션에서 반복 확인).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def quality(trace):
    """궤적 기록 -> 품질 지표 + 종합 점수(0~100)."""
    if len(trace) < 3:
        return {'score': 0, 'note': '궤적이 너무 짧음'}

    # 복귀(ret) 구간은 품질 계산에서 뺀다 - 임무 후 팔을 거두는 길이를
    # 효율/스텝으로 벌점하면 복귀 자체를 못 하게 된다. 충돌·여유 검사는
    # 전체 구간(복귀 포함)으로 한다.
    core = [t for t in trace if not t.get('ret')]
    hands = [t['hand'] for t in core if 'hand' in t]
    if len(hands) < 3:
        return {'score': 0, 'note': '손 궤적 없음'}

    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    path = sum(dist(a, b) for a, b in zip(hands[:-1], hands[1:]))
    straight = dist(hands[0], hands[-1])
    efficiency = straight / path if path > 1e-6 else 0.0

    # 저크: 관절 속도(스텝 간 변화)의 변화량 평균. 작을수록 부드럽다.
    joints = [t['joints'] for t in core if 'joints' in t]
    jerk = 0.0
    if len(joints) >= 3:
        keys = joints[0].keys()
        vels = [{k: b[k] - a[k] for k in keys}
                for a, b in zip(joints[:-1], joints[1:])]
        jerks = [sum(abs(b[k] - a[k]) for k in keys)
                 for a, b in zip(vels[:-1], vels[1:])]
        jerk = sum(jerks) / len(jerks)

    tables = [t['table'] for t in trace]
    objs = [t['obj'] for t in trace]
    margin = min(min(tables), min(objs))
    avg_margin = sum(min(a, b) for a, b in zip(tables, objs)) / len(trace)

    # 시야 유지: 목표가 눈 화면에 있던 스텝 비율. 실물은 눈 뷰만 보고
    # 동작하므로 목표를 잃는 목 움직임은 그 자체로 나쁜 동작이다.
    views = [t['view'] for t in core if 'view' in t]
    view_ratio = (sum(views) / len(views)) if views else None

    # 과신전: 어깨에서 손까지 최대 신장. 팔 전장 ~60cm 에서 50cm 넘게
    # 뻗으며 올라오면 사람 눈에 '너무 뻗는' 동작이다 - 비평가가 지표로
    # 보고 via 자세 조언에 쓴다 (배점은 효율이 이미 우회를 벌점한다).
    shoulder = (0.0, -0.19, 0.0)
    stretch = max(dist(h, shoulder) for h in hands) * 100

    metrics = {
        'stretch_max_cm': round(stretch, 1),
        'efficiency': round(efficiency, 3),
        'jerk_deg': round(jerk, 2),
        'min_margin_cm': round(margin, 1),
        'avg_margin_cm': round(avg_margin, 1),
        'steps': len(core),
        'path_cm': round(path * 100, 1),
        'view_ratio': round(view_ratio, 2) if view_ratio is not None else None,
    }

    # 종합: 효율 35 + 부드러움 25 + 여유 20 + 간결함 10 + 시야 10
    score = 0.0
    score += 35 * min(1.0, efficiency / 0.7)          # 0.7 이면 만점 (우회 포함)
    score += 25 * max(0.0, 1.0 - jerk / 6.0)          # 저크 6도/스텝이면 0점
    # 손가락 파지에서는 손끝이 테이블 ~1cm 까지 가는 게 '올바른' 동작이다.
    # 만점 기준을 3.5cm 로 두면 파지 자체가 벌점된다 (무동작 기준선 원칙의
    # 역방향 검사: 올바른 동작이 몇 점인지도 봐야 한다).
    score += 20 * max(0.0, min(1.0, (margin - (-0.5)) / 1.5))   # 여유 +1cm 만점
    score += 5 * max(0.0, 1.0 - max(0, len(core) - 20) / 40.0)
    score += 10 * (view_ratio if view_ratio is not None else 1.0)

    # 접촉 부드러움: 대상에 가장 가까웠던 순간의 손 속도. 접촉(집기)은
    # 허용이지만 '쿵' 은 안 된다 - 사람도 잡기 직전에 손을 늦춘다.
    tgts = [t.get('tgt') for t in core]
    touch_speed = None
    if any(v is not None for v in tgts):
        i = min((v, k) for k, v in enumerate(tgts) if v is not None)[1]
        if 0 < i < len(hands):
            touch_speed = dist(hands[i], hands[i - 1]) * 100
    metrics['touch_speed_cm'] = (round(touch_speed, 1)
                                 if touch_speed is not None else None)
    # 조임 깊이: 대상 최소 여유. 실물 그리퍼는 과조임이 고장 1순위라
    # 부드러움 점수가 속도와 깊이 중 나쁜 쪽을 따른다.
    tgt_min = min((v for v in tgts if v is not None), default=None)
    metrics['squeeze_cm'] = (round(-min(0.0, tgt_min), 2)
                             if tgt_min is not None else None)
    sp = (1.0 if touch_speed is None
          else max(0.0, 1.0 - max(0.0, touch_speed - 1.5) / 3.0))
    sq = (1.0 if tgt_min is None
          else max(0.0, 1.0 - max(0.0, -tgt_min - 0.15) / 0.85))
    score += 5 * min(sp, sq)
    metrics['score'] = int(round(score))
    return metrics


# ── 3단계 판정 ───────────────────────────────────────────────────────────
#
# 평가는 순서가 있다 (사용자 정의):
#   1) 충돌 없이 동작을 수행할 것      - 경로 어디서든 충돌이면 그걸로 끝
#   2) 목표 동작을 자연스럽게 움직일 것 - 품질(효율·저크·여유)이 문턱 미달이면
#                                        도달했어도 성공이 아니다
#   3) 이후 성공 판단                   - 1·2 를 통과한 것만 성공을 논한다
#
# 성공의 정의가 조여진다: 거칠게 스치며 닿은 동작은 '부자연' 이지 성공이
# 아니고, 우아하지만 못 닿은 동작도 '미도달' 이지 성공이 아니다. 둘 다
# 만족해야 한다. natural_min 은 행동이 좋아질수록 올라간다(평가도 함께 개선).

STAGES = ('collision', 'unnatural', 'missed', 'success')
STAGE_KO = {'collision': '충돌', 'unnatural': '부자연',
            'missed': '미도달', 'success': '성공'}


TARGET_PEN_CM = -1.0    # 대상 물체 허용 관통 - 접촉(잡기)은 목표지 충돌이 아니다


def stage_verdict(path_tab, path_obj, path_tgt, reached, quality_score,
                  natural_min=55):
    """(stage, why). 단계 순서대로 걸리는 첫 항목이 판정이다.

    충돌의 정의가 대상/비대상으로 갈린다: 테이블·다른 물체는 스치는 것도
    금지(-0.5cm)지만, **대상 물체는 닿는 게 목표**라 접촉은 허용하고
    관통(-1cm 초과)만 충돌로 본다. '충돌하지 마' 를 대상까지 걸면
    집기가 원천적으로 불가능해진다.
    """
    if path_tab < -0.5:
        return 'collision', '경로에서 테이블 충돌 (%.1fcm)' % path_tab
    if path_obj < -0.5:
        return 'collision', '경로에서 다른 물체 충돌 (%.1fcm)' % path_obj
    if path_tgt < TARGET_PEN_CM:
        return 'collision', '대상 관통 (%.1fcm - 접촉 한도 %.1f)' % (
            path_tgt, TARGET_PEN_CM)
    if quality_score < natural_min:
        return 'unnatural', '품질 %d < 문턱 %d (거칠거나 우회)' % (
            quality_score, natural_min)
    if not reached:
        return 'missed', '충돌 없고 자연스러우나 목표 미달'
    return 'success', None


FILM_N = 8              # 필름 스트립 프레임 수 - 전체 과정을 한 장에 담는다


def strip_image(frames, n=FILM_N, cols=4):
    """프레임들 -> 시간순 격자 한 장 (제3자 시점만, 번호 라벨).

    프레임 2~3장만 보면 '중간에 뭘 했는지' 가 빠진다. 전 과정을 균등하게
    n 장 뽑아 왼쪽 위 -> 오른쪽 아래 시간 순 격자로 만든다. _snap 프레임은
    [눈|제3자] 가로 결합인데, 자연스러움은 몸 전체가 보이는 제3자 쪽만 쓴다.
    투과 순간의 붉은 테두리는 그대로 살아 있다.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    picked = pick_frames(frames, n)
    if not picked:
        return None

    def _row(side):
        tiles = []
        for i, f in enumerate(picked):
            half = f.shape[1] // 2
            part = f[:, :half] if side == 'eye' else f[:, half:]
            im = Image.fromarray(part).resize((320, 240))
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 30, 20), fill=(0, 0, 0))
            d.text((8, 4), str(i + 1), fill=(255, 255, 90))
            tiles.append(np.asarray(im))
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        return [np.concatenate(tiles[r:r + cols], axis=1)
                for r in range(0, len(tiles), cols)]

    # 로봇 눈 시점을 함께 본다 - '지금 안 보이는데 움직인다' 를 비평이
    # 짚을 수 있어야 한다 (사용자 지적). 위 = 눈, 아래 = 제3자.
    return np.concatenate(_row('eye') + _row('third'), axis=0)


def _encode(img, width=896):
    """ndarray -> JPEG base64 (폭 제한으로 토큰 절약)."""
    import base64
    import io as _io

    from PIL import Image

    im = Image.fromarray(img).convert('RGB')
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)))
    buf = _io.BytesIO()
    im.save(buf, format='JPEG', quality=72)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _parse_json(raw):
    if not raw:
        return None
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        out = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


CAUSES = ('경로 충돌', '파지 시도 없음', '정렬 미달(닫힘 미발행)',
          '맞물림 실패', '들기/운반 중 놓침', '품질 미달')


def diagnose(q):
    """이 에피소드가 '왜' 실패했는지 계측으로 판정한다.

    평가의 일은 점수만이 아니라 원인이다 (사용자) - 방향 결정은 최다
    원인을 겨냥해야 한다. 규칙은 파지 리포트(닫힘/맞물림/정렬)와 단계
    판정을 잇는다.
    """
    if q.get('stage') == 'success':
        return None
    if q.get('stage') == 'collision':
        return '경로 충돌'
    g = q.get('grasp') or {}
    if not g:
        return '파지 시도 없음'
    if not g.get('closed'):
        return '정렬 미달(닫힘 미발행)'
    if not g.get('grabbed'):
        return '맞물림 실패'
    if q.get('stage') == 'missed':
        return '들기/운반 중 놓침'
    return '품질 미달'


CRITIQUE_PROMPT = """로봇 팔 동작 품질 비평가다. 이미지는 한 동작의 필름 스트립이다: 위쪽 절반 = 로봇 눈 시점, 아래쪽 절반 = 제3자 시점 (같은 순간, 번호 1이 시작). 붉은 테두리 프레임은 그 순간 투과가 있었던 것. 눈 시점에서 대상이 안 보이는데 팔이 움직이는 구간이 있으면 반드시 지적하라.

지시: {instruction}
계측 지표: {metrics}

전체 흐름을 보고 사람 눈에 자연스러운지 비평하라. 성공 여부는 판정하지 마라(그건 계측이 한다). 오직 '어떻게 움직였는가' 만.

JSON 한 줄로만 답하라:
{{"naturalness": 1~10, "rubric": {{"smooth": 1~10, "direct": 1~10, "posture": 1~10, "tempo": 1~10}}, "worst_frame": 번호, "failure_cause": "실패의 시각적 원인 한 구절 (성공이면 null)", "fix_hint": "그 원인을 고칠 구체적 힌트 한 문장", "issues": ["짧은 지적", ...], "advice": {{"파라미터": "up"|"down"}}}}

rubric 뜻: smooth=급격한 방향전환·떨림 없음, direct=군더더기 없는 경로, posture=중간 자세가 사람 팔처럼 자연스러운가, tempo=속도가 일정한가. worst_frame=가장 어색한 프레임 번호.

advice 에 쓸 수 있는 파라미터와 뜻:
- step_deg: 스텝 크기 (down=더 잘게 움직여 부드럽게, up=시원시원하게)
- gain: 이득 (down=신중하게, up=민첩하게)
- damping: 감쇠 (up=떨림 억제)
- table_min: 테이블 여유 (up=더 높이 돌아 안전하게)
- lift_steps: 들어올리기 보간 수 (up=더 부드러운 들어올림)
- via1_/via2_/via3_ 접두사 + pitch/roll/yaw/elbow: 들어올리기 경유 자세 3개 (예: via2_elbow) - 경로 모양 자체가 어색하면 이걸 지목하라

문제가 없으면 issues 를 빈 배열로, advice 를 빈 객체로. JSON 밖 텍스트 금지."""


def critique(client, frames, instruction, metrics, feedback=None):
    """opus 전과정 비평. {'naturalness','rubric','worst_frame','issues','advice'}.

    frames: 에피소드의 프레임 목록 전체. 내부에서 필름 스트립을 만든다.
    feedback: 운영자/오케스트라의 미결 지적 - 비평이 이걸 우선 반영한다.
    """
    if client is None or not frames:
        return None
    strip = strip_image(frames)
    if strip is None:
        return None
    prompt = CRITIQUE_PROMPT.format(
        instruction=instruction,
        metrics=json.dumps(metrics, ensure_ascii=False))
    if feedback:
        prompt += ('\n\n운영자 지적(미결 - 조언에 우선 반영하라):\n- '
                   + '\n- '.join(feedback[-5:]))
    out = _parse_json(client.ask_vision(prompt, image=_encode(strip)))
    return out


ORCHESTRA_PROMPT = """로봇 학습 파이프라인의 오케스트라(총감독)다. 필름 스트립(최근 동작)과 아래 현황을 보고, 운영자가 하듯 판단하라.

현황:
{state}

네가 할 수 있는 일:
1. experiment - 환경을 바꿔 새로운 상황을 시험한다. 조절 가능:
   table [낮은높이, 높은높이] (m, -0.40~-0.24), objects [최소, 최대] (1~4), min_gap 물체간격 (0.05~0.14)
   성적이 좋으면 어렵게(낮고 다양한 테이블, 물체 많게, 간격 좁게), 나쁘면 쉽게.
2. feedback_add - 동작에서 불필요하거나 빠진 것을 운영자처럼 지적한다 (비평가가 매번 반영한다). 이미 있는 지적과 중복 금지.
3. feedback_done - 해결된 지적의 번호를 종결한다 (스트립에서 더는 안 보이면).

JSON 한 줄로만:
{{"experiment": {{"table": [lo,hi], "objects": [lo,hi], "min_gap": g}} 또는 null, "feedback_add": ["지적", ...], "feedback_done": [번호, ...], "why": "한 문장"}}
바꿀 게 없으면 experiment 는 null, 배열들은 빈 배열. JSON 밖 텍스트 금지."""


def orchestrate(client, frames, state_text):
    """오케스트라 한 번: 실험 제안/지적 관리. dict 또는 None."""
    if client is None:
        return None
    img = None
    if frames:
        st = strip_image(frames)
        if st is not None:
            img = _encode(st)
    out = _parse_json(client.ask_vision(
        ORCHESTRA_PROMPT.format(state=state_text), image=img))
    return out


DUEL_PROMPT = """로봇 팔 동작 비교 심판이다. 이미지 위쪽 절반이 동작 A, 아래쪽 절반이 동작 B다. 둘 다 같은 지시를 수행한 전 과정의 필름 스트립이다(각각 시간 순, 번호 순).

지시: {instruction}

어느 쪽이 사람 눈에 더 자연스러운가? 성공 여부가 아니라 움직임의 질만 비교하라 - 부드러움, 경로의 군더더기, 자세, 템포.

JSON 한 줄로만: {{"winner": "A"|"B", "why": "한 문장"}}"""


def save_strip(frames, path):
    """필름 스트립을 파일로 (개선 과정을 눈으로 확인하는 용도)."""
    from PIL import Image
    st = strip_image(frames)
    if st is None:
        return False
    Image.fromarray(st).save(path, quality=80)
    return True


def duel(client, frames_a, frames_b, instruction, save_path=None):
    """같은 태스크를 수행한 두 동작의 쌍대 비교. {'winner','why'} 또는 None.

    절대 점수는 호출마다 흔들려도 '어느 쪽이 낫나' 는 안정적이다. 결정론
    품질 점수가 박빙일 때 사람 눈 기준의 심판으로 쓴다.
    """
    if client is None or not frames_a or not frames_b:
        return None
    import numpy as np
    from PIL import Image, ImageDraw

    strips = []
    for label, fr in (('A', frames_a), ('B', frames_b)):
        st = strip_image(fr, n=6, cols=6)      # 한 줄 x 6장 -> A/B 위아래 비교
        if st is None:
            return None
        im = Image.fromarray(st)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 44, 30), fill=(20, 20, 120) if label == 'A'
                    else (120, 20, 20))
        d.text((14, 6), label, fill=(255, 255, 255))
        strips.append(np.asarray(im))
    gap = np.full((12, strips[0].shape[1], 3), 255, dtype=strips[0].dtype)
    combo = np.concatenate([strips[0], gap, strips[1]], axis=0)
    if save_path:
        try:
            Image.fromarray(combo).save(save_path, quality=80)
        except Exception:
            pass
    out = _parse_json(client.ask_vision(
        DUEL_PROMPT.format(instruction=instruction), image=_encode(combo, 1152)))
    if out and out.get('winner') in ('A', 'B'):
        return out
    return None


def pick_frames(frames, n=3):
    """프레임 목록에서 대표 n장 (처음/중간/끝)."""
    if not frames:
        return []
    if len(frames) <= n:
        return list(frames)
    idx = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in idx]
