"""오케스트라 - 배치가 끝나면 '지표 자체' 를 감사하고 다음을 정한다.

이 세션에서 겪은 버그는 전부 한 종류였다: harness 가 현실과 안 맞는 점수를
자신 있게 보고했다(손이 컵을 관통하며 '성공', 카메라가 안 돌았는데 서보 '동작',
휴식 자세가 '위험'). 여러 모델이 서로를 채점하는 파이프라인에서 이걸 안 잡으면
자신 있게 쓰레기로 수렴한다. 그래서 오케스트라의 첫 임무는 시나리오 검토가
아니라 **지표 감사** 다.

구성:
  audit(records)      결정론적 감사 - 거짓 성공/깨진 지표/체계적 실패 플래그.
                      LLM 아님. 여기서 걸리면 사람이 봐야 한다.
  curriculum(summary) 다음 배치 난이도 제안 (결정론적 규칙).
  review(summary, audit, client)  Codex 총평 한 번 - 감사 결과와 요약을 주고
                      사람이 읽을 소견을 받는다. 배치당 1회, 전용 세션.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def audit(records):
    """결정론적 지표 감사. 걸린 것들의 목록(비면 깨끗)."""
    flags = []
    n = len(records)
    if not n:
        return ['레코드가 0건 - 배치 자체가 안 돌았다.']

    # 1) 거짓 성공: 성공인데 물체를 파고들었다.
    for r in records:
        v = r['verdict']
        if v.get('success') and v.get('object_clearance_cm', 9) < -0.5:
            flags.append('거짓 성공 의심: %s 는 성공 판정인데 물체를 %.1fcm '
                         '파고들었다. 성공 기준이 투과를 안 보고 있다.'
                         % (r['task']['id'], v['object_clearance_cm']))

    # 1b) 거짓 성공: 성공인데 '경로' 에서 테이블/물체를 파고들었다.
    #     최종 상태만 보고 13/13 프레임 테이블 투과를 성공으로 통과시킨
    #     실전 사례에서 나온 규칙이다.
    for r in records:
        v = r['verdict']
        ex = r.get('execution') or {}
        for key, label in (('table_min_cm', '테이블'), ('object_min_cm', '물체')):
            if v.get('success') and ex.get(key, 9) < -0.5:
                flags.append('거짓 성공 의심: %s 는 성공인데 경로에서 %s 를 '
                             '%.1fcm 파고들었다. 판정이 경로를 안 보고 있다.'
                             % (r['task']['id'], label, ex[key]))

    # 2) 판정과 실행의 모순: '손을 움직이는' 유형인데 성공 판정이 손과
    #    무관하게 참이 됐다. look 은 시선 태스크라 손 거리를 안 본다 - 첫
    #    실행에서 look 에 이 규칙을 적용해 오탐 3건을 낸 교훈이다.
    for r in records:
        if r['task']['kind'] not in ('reach', 'point', 'pick', 'lift'):
            continue
        v = r['verdict']
        ex = r.get('execution') or {}
        d = ex.get('final_dist_cm')
        if v.get('success') and d is not None and d > 25:
            flags.append('모순: %s 성공인데 손-목표 %.0fcm - 판정이 실행과 '
                         '무관하게 참이 됐다.' % (r['task']['id'], d))

    # 3) 전부 만점/전부 빵점: 지표가 아무것도 구분하지 못하고 있다.
    oks = [bool(r['verdict'].get('success')) for r in records]
    if n >= 6 and all(oks):
        flags.append('전원 성공(%d/%d) - 지표가 포화됐거나 태스크가 너무 쉽다. '
                     '난이도를 올리거나 판정을 조여라.' % (n, n))
    if n >= 6 and not any(oks):
        flags.append('전원 실패(0/%d) - 체계적 문제다. 유형별 실패 원인을 보라.'
                     % n)

    # 4) 인지가 전부 실패: 감지 파이프라인이 죽었는데 배치가 돌았다.
    per = [bool(r.get('perception', {}).get('seen')) for r in records]
    if n >= 4 and not any(per):
        flags.append('인지 전멸 - 검출/시선 훑기가 통째로 죽었는데 배치는 돌았다.')

    infra = [r['task']['id'] for r in records
             if r.get('verdict', {}).get('stage') == 'infra_error']
    if infra:
        flags.append('계획 백엔드 오류 %d건(%s) - 동작 실패 통계와 분리하고 '
                     '배치를 중단하라.' % (len(infra), ', '.join(infra)))

    # 5) 유형별 전멸: 특정 유형만 0이면 기구학/판정의 구조적 문제.
    by_kind = {}
    for r, ok in zip(records, oks):
        by_kind.setdefault(r['task']['kind'], []).append(ok)
    for kind, results in by_kind.items():
        if len(results) >= 2 and not any(results):
            flags.append('%s 유형 전멸(0/%d) - 이 유형의 계획이나 판정, 또는 '
                         '로봇의 물리적 한계를 의심하라(예: pick 은 이 손으로 '
                         '작은 물체를 못 감싼다).' % (kind, len(results)))
    return flags


def curriculum(summary):
    """다음 배치 난이도(결정론). {'objects': n, 'noise_px': f, 'note': str}"""
    rate = summary.get('rate', 0)
    if rate >= 0.9:
        return {'objects': 3, 'noise_px': 4.0,
                'note': '성공률 %.0f%% - 물체 3개·잡음 4px 로 올린다.' % (rate * 100)}
    if rate >= 0.6:
        return {'objects': 2, 'noise_px': 2.0,
                'note': '성공률 %.0f%% - 유지.' % (rate * 100)}
    return {'objects': 1, 'noise_px': 1.0,
            'note': '성공률 %.0f%% - 물체 1개·잡음 1px 로 낮춰 원인을 분리한다.'
            % (rate * 100)}


def review(summary, audit_flags, client=None):
    """Codex 총평 (배치당 1회). client 없으면 결정론 요약만."""
    if client is None:
        return None
    prompt = (
        '로봇 시뮬 파이프라인 실행 리뷰어다. 아래 배치 요약과 자동 감사 결과를 '
        '보고, 사람이 읽을 총평을 한국어 5문장 이내로 써라. 감사에 걸린 항목이 '
        '있으면 그것부터, 없으면 다음에 시도할 개선 1가지를 제안하라. '
        '점수를 매기지 말고 구체적으로. '
        '반드시 {"review":"총평"} JSON 객체로 답하라.\n\n'
        '배치 요약:\n%s\n\n자동 감사:\n%s'
        % (json.dumps(summary, ensure_ascii=False, indent=1),
           '\n'.join('- ' + f for f in audit_flags) or '- 깨끗함'))
    try:
        out = client.ask_evaluation(prompt, kind='review')
        return (out or {}).get('review')
    except Exception:
        return None
