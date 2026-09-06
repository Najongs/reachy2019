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

    hands = [t['hand'] for t in trace if 'hand' in t]
    if len(hands) < 3:
        return {'score': 0, 'note': '손 궤적 없음'}

    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    path = sum(dist(a, b) for a, b in zip(hands[:-1], hands[1:]))
    straight = dist(hands[0], hands[-1])
    efficiency = straight / path if path > 1e-6 else 0.0

    # 저크: 관절 속도(스텝 간 변화)의 변화량 평균. 작을수록 부드럽다.
    joints = [t['joints'] for t in trace if 'joints' in t]
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

    metrics = {
        'efficiency': round(efficiency, 3),
        'jerk_deg': round(jerk, 2),
        'min_margin_cm': round(margin, 1),
        'avg_margin_cm': round(avg_margin, 1),
        'steps': len(trace),
        'path_cm': round(path * 100, 1),
    }

    # 종합: 효율 40 + 부드러움 30 + 여유 20 + 간결함 10
    score = 0.0
    score += 40 * min(1.0, efficiency / 0.7)          # 0.7 이면 만점 (우회 포함)
    score += 30 * max(0.0, 1.0 - jerk / 6.0)          # 저크 6도/스텝이면 0점
    score += 20 * max(0.0, min(1.0, (margin - (-0.5)) / 4.0))   # 여유 3.5cm+ 만점
    score += 10 * max(0.0, 1.0 - max(0, len(trace) - 20) / 40.0)
    metrics['score'] = int(round(score))
    return metrics


CRITIQUE_PROMPT = """로봇 팔 동작 품질 비평가다. 아래는 한 동작의 프레임(들어올림/중간/도달 순)과 계측 지표다.

지시: {instruction}
지표: {metrics}

동작이 사람 눈에 자연스러운지 비평하라. 성공 여부는 판정하지 마라(그건 계측이 한다). 오직 '어떻게 움직였는가' 만.

JSON 한 줄로만 답하라:
{{"naturalness": 1~10, "issues": ["짧은 지적", ...], "advice": {{"파라미터": "up"|"down"}}}}

advice 에 쓸 수 있는 파라미터와 뜻:
- step_deg: 스텝 크기 (down=더 잘게 움직여 부드럽게, up=시원시원하게)
- gain: 이득 (down=신중하게, up=민첩하게)
- damping: 감쇠 (up=떨림 억제)
- table_min: 테이블 여유 (up=더 높이 돌아 안전하게)
- lift_steps: 들어올리기 보간 수 (up=더 부드러운 들어올림)

문제가 없으면 issues 를 빈 배열로, advice 를 빈 객체로. JSON 밖 텍스트 금지."""


def critique(client, frame_images, instruction, metrics):
    """opus 프레임 비평. {'naturalness', 'issues', 'advice'} 또는 None.

    frame_images: ndarray 3장 내외. 세로로 이어붙여 한 장으로 보낸다.
    """
    if client is None or not frame_images:
        return None
    import base64
    import io as _io

    import numpy as np
    from PIL import Image

    strip = np.concatenate(list(frame_images), axis=0)
    im = Image.fromarray(strip).convert('RGB')
    # 3장 세로면 길다 - 폭 512 로 줄여 토큰을 아낀다.
    if im.width > 512:
        im = im.resize((512, int(im.height * 512 / im.width)))
    buf = _io.BytesIO()
    im.save(buf, format='JPEG', quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    prompt = CRITIQUE_PROMPT.format(
        instruction=instruction,
        metrics=json.dumps(metrics, ensure_ascii=False))
    raw = client.ask_vision(prompt, image=b64)
    if not raw:
        return None
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        out = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    if not isinstance(out, dict):
        return None
    out['raw'] = raw
    return out


def pick_frames(frames, n=3):
    """프레임 목록에서 대표 n장 (처음/중간/끝)."""
    if not frames:
        return []
    if len(frames) <= n:
        return list(frames)
    idx = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in idx]
