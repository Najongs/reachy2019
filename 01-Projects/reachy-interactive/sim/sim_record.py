"""파이프라인의 모든 산출물이 흐르는 단일 레코드 스키마 + 문서 생성.

지금까지 실행 기록이 sim_results*.json 네 파일로 흩어졌다. 흩어지면 나중에
"이 동작이 왜 실패했지" 를 되짚을 수 없다. 시도 1건 = 레코드 1건으로 통일한다:

  {run_id, task, instruction, answer, perception, plan, execution, verdict,
   findings, artifacts, seed, ts}

- perception:  검출 목록 + 기억 요약 + 오버레이 이미지 경로 (참값 없음)
- plan:        opus 원문 + 해석된 선택
- execution:   서보 스텝 수, 최종 손-목표 거리, 프레임별 최소 clearance
- verdict:     객관 판정 (참값 채점 + 투과) - 모델 판정 아님
- artifacts:   영상/프레임 경로

write() 가 레코드를 sim_data/<run>/ 에 남기고, aggregate() 가 요약하고,
report() 가 docs/eval/<run>.md 를 만든다. 문서는 레코드에서 '생성' 된다 -
손으로 쓰는 문서는 코드와 어긋난다.
"""

import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'sim_data')
DOCS = os.path.join(ROOT, 'docs', 'eval')


def new_run(tag=''):
    run_id = time.strftime('%Y%m%d-%H%M%S') + (('-' + tag) if tag else '')
    os.makedirs(os.path.join(DATA, run_id), exist_ok=True)
    return run_id


def run_dir(run_id):
    return os.path.join(DATA, run_id)


def write(run_id, record):
    record.setdefault('ts', time.strftime('%Y-%m-%d %H:%M:%S'))
    record['run_id'] = run_id
    path = os.path.join(run_dir(run_id), record['task']['id'] + '.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
    return path


def load_all(run_id):
    out = []
    d = run_dir(run_id)
    for f in sorted(os.listdir(d)):
        if f.endswith('.json') and f != 'summary.json':
            with open(os.path.join(d, f), encoding='utf-8') as fh:
                out.append(json.load(fh))
    return out


def aggregate(run_id):
    """성공률/유형별/실패패턴/투과 요약. summary.json 으로도 남긴다."""
    recs = load_all(run_id)
    by_kind = {}
    pierced = []
    failures = []
    for r in recs:
        k = r['task']['kind']
        by_kind.setdefault(k, [0, 0])
        ok = bool(r['verdict'].get('success'))
        by_kind[k][0] += ok
        by_kind[k][1] += 1
        if r['verdict'].get('object_clearance_cm', 9) < -0.5:
            pierced.append(r['task']['id'])
        if not ok:
            failures.append({'id': r['task']['id'],
                             'why': r['verdict'].get('why') or
                             (r.get('findings') or ['?'])[0]})
    n = len(recs)
    ok_n = sum(v[0] for v in by_kind.values())
    stages = {}
    for r in recs:
        st = r['verdict'].get('stage_ko') or (
            '성공' if r['verdict'].get('success') else '실패')
        stages[st] = stages.get(st, 0) + 1
    summary = {
        'run_id': run_id, 'n': n, 'success': ok_n,
        'rate': round(ok_n / n, 3) if n else 0,
        'stages': stages,
        'by_kind': {k: {'ok': v[0], 'n': v[1]} for k, v in by_kind.items()},
        'pierced': pierced, 'failures': failures,
    }
    with open(os.path.join(run_dir(run_id), 'summary.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


def report(run_id, orchestra=None):
    """docs/eval/<run>.md 생성. 표 + 태스크별 상세 + 오케스트라 총평."""
    recs = load_all(run_id)
    summary = aggregate(run_id)
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, run_id + '.md')

    lines = ['# 실행 리포트 %s' % run_id, '',
             '- 성공 **%d/%d** (%.0f%%)' % (summary['success'], summary['n'],
                                            summary['rate'] * 100),
             '- 유형별: ' + ', '.join('%s %d/%d' % (k, v['ok'], v['n'])
                                      for k, v in
                                      sorted(summary['by_kind'].items())),
             '- 단계 분해: ' + ', '.join('%s %d' % (k, v) for k, v in
                                          sorted(summary['stages'].items())),
             '- 물체 투과: %s' % (', '.join(summary['pierced']) or '없음'), '',
             '| 태스크 | 지시 | 계획 | 판정 | 품질 | 물체여유(cm) | 산출물 |',
             '|---|---|---|---|---|---|---|']
    for r in recs:
        t = r['task']
        v = r['verdict']
        art = r.get('artifacts', {})
        vid = art.get('video')
        link = '[영상](%s)' % os.path.relpath(vid, DOCS) if vid else '-'
        q = ((r.get('execution') or {}).get('quality') or {}).get('score', '-')
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            t['id'], t['instruction'][:24],
            (r.get('plan') or {}).get('mode', '-'),
            '✓' if v.get('success') else '✗', q,
            v.get('object_clearance_cm', '-'), link))

    if summary['failures']:
        lines += ['', '## 실패 상세', '']
        for f in summary['failures']:
            lines.append('- **%s**: %s' % (f['id'], f['why']))

    if orchestra:
        lines += ['', '## 오케스트라 총평', '', orchestra]

    lines += ['', '---', '레코드: `sim_data/%s/` (시도별 JSON + 영상)' % run_id]
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path
