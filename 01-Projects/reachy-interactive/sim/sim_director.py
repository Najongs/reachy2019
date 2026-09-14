"""감독(director) - 짧게 돌리고, 평가를 읽고, 방향을 바꿔 다시 돌린다.

사용자 지적에서 나왔다: docs/eval 에 평가가 쌓여도 그걸 읽고 방향을
바꾸는 소비자가 없었다. 이 루프가 그 소비자다:

  사이클마다
    1) 개선 루프를 '짧게' (CYCLE_ITERS 회)
    2) 실전 배치 1개 (브로커) -> docs/eval/<run>.md + 감사 플래그
    3) 증거를 모아 Codex에게: 방향 결정 JSON
       (환경, 태스크 유형, 문턱, 파라미터 리셋, 지적 추가/종결)
    4) 결정을 '적용' 하고 docs/eval/direction-log.md 에 기록
    5) 다음 사이클

적용은 전부 한계 안에서(clamp) - 감독이 물리적으로 말이 안 되는 지시를
내릴 수 없다. 코드 수준의 의심은 로그에 적어 사람에게 넘긴다.

사용:
    python3 sim/sim_director.py --hours 8 --token reachy2019
    python3 sim/sim_director.py --cycles 2 --token reachy2019   # 예행
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_improve as I                            # noqa: E402
import sim_critic as C                             # noqa: E402
import sim_tasks as T                              # noqa: E402

CYCLE_ITERS = 8
RUN_GENERATION = 'codex-v1'
STATE_FILE = os.path.join(HERE, '..', 'config', 'sim_direction.json')
LOG_MD = os.path.join(HERE, '..', 'docs', 'eval', 'direction-log.md')


class EvaluationUnavailable(RuntimeError):
    pass

DIRECTOR_PROMPT = """로봇 학습 파이프라인의 감독이다. 방금 끝난 짧은 학습 사이클의 평가와 실전 배치 결과를 보고, 다음 사이클의 방향을 정한다. 필름 스트립(위=로봇 눈, 아래=제3자)이 오면 함께 본다.

## 이번 사이클 증거
{evidence}

'실패원인 분포' 가 판단의 축이다: 방향은 최다 원인을 겨냥하라 -
정렬 미달이면 서보/자세, 맞물림 실패면 조임/깊이, 놓침이면 마찰/속도,
경로 충돌이면 경유/환경. 원인과 무관한 손잡이를 돌리지 마라.

## 지금 방향 (이전 결정)
{direction}

바꿀 수 있는 것 (전부 선택):
- "env": {{"table": [lo,hi] (-0.40~-0.24), "objects": [lo,hi] (1~4), "min_gap": 0.05~0.14}}
- "kinds": ["pick","lift"] 부분집합 - 어디에 집중할지
- "natural_min": 45~75 - 자연스러움 문턱 시작값 (성공이 안 나오면 낮춰 압박을 줄여라)
- "reset_params": "defaults" | null - 파라미터가 구석에 갇혔다고 보면 리셋
- "feedback_add": ["동작 지적", ...] / "feedback_done": [번호,...] - 해결됐거나 겹치는 지적은 과감히 종결하라
- "code_suspect": "절차/코드 결함 의심이면 한 문장" (사람에게 전달됨)

JSON 한 줄로만:
{{"env": ... 또는 null, "kinds": [...] 또는 null, "natural_min": n 또는 null, "reset_params": ..., "feedback_add": [], "feedback_done": [], "code_suspect": null, "why": "방향 요약 한 문장"}}"""


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def _load_state():
    try:
        return json.load(open(STATE_FILE, encoding='utf-8'))
    except Exception:
        return {'env': None, 'kinds': ['pick', 'lift'], 'natural_min': 55}


def _save_state(st):
    json.dump(st, open(STATE_FILE, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


def _log_direction(cyc, summary, batch_line, decision):
    os.makedirs(os.path.dirname(LOG_MD), exist_ok=True)
    new = not os.path.exists(LOG_MD)
    with open(LOG_MD, 'a', encoding='utf-8') as fh:
        if new:
            fh.write('# 방향 결정 기록 (감독 루프)\n\n'
                     '사이클마다: 평가 -> 방향 결정 -> 적용. '
                     '이 파일이 "평가가 반영되는 증거" 다.\n\n')
        fh.write('## %s 사이클 %d\n' % (time.strftime('%m-%d %H:%M'), cyc))
        fh.write('- 학습: 품질 %.0f->%.0f (문턱 %d), 시험 %s\n'
                 % (summary['quality_first'], summary['quality_last'],
                    summary['natural_min'], summary.get('exam')))
        if summary.get('incidents'):
            fh.write('- 사건: %s\n' % '; '.join(summary['incidents']))
        fh.write('- 실전 배치: %s\n' % batch_line)
        fh.write('- **방향**: %s\n' % (decision.get('why') or '유지'))
        for k in ('env', 'kinds', 'natural_min', 'reset_params'):
            if decision.get(k):
                fh.write('  - %s -> %s\n' % (k, decision[k]))
        if decision.get('code_suspect'):
            fh.write('  - 코드 의심(사람 확인 필요): %s\n'
                     % decision['code_suspect'])
        fh.write('\n')


def run(cycles=None, hours=None, token=None, url='http://127.0.0.1:8080',
        seed=1, offline=False):
    from llm_client import BrokerClient
    import sim_pipeline as PL
    client = None
    if not offline:
        client = BrokerClient(url, token=token, session='sim-director')
        health = client.health() or {}
        components = health.get('components') or {}
        unavailable = [name for name in ('grounding', 'evaluation')
                       if not components.get(name, {}).get('enabled')
                       or not components.get(name, {}).get('ok')]
        if unavailable:
            # 중단하지 않는다 - 2026-09-14 사고: Codex 한도 하나로 밤새
            # 기동-즉사를 반복하며 학습을 통째로 잃었다. 평가기가 없으면
            # 계측 전용(offline) 으로 강등해 CEM 이라도 굴린다.
            print('[감독 강등] 브로커 구성 사전검사 실패(%s) - '
                  'offline 계측 전용으로 계속' % ', '.join(unavailable),
                  flush=True)
            offline = True
            client = None
        else:
            probe = client.ask_evaluation(
                '평가기 연결 확인이다. {"ok": true}만 반환하라.', kind='health')
            if not probe or probe.get('ok') is not True:
                print('[감독 강등] 평가기 사전검사 실패(%s) - '
                      'offline 계측 전용으로 계속'
                      % (client.last_error or probe), flush=True)
                offline = True
                client = None
    else:
        print('[감독] offline metric-only: Codex/Ollama 호출 없이 CEM 계측만 사용',
              flush=True)
    st = _load_state()
    # 시작할 때 지난 런의 산출물을 저장소 보관고(04-Archives)로 내린다 -
    # sim_data 가 수백 개 폴더로 부풀지 않게 (한 번 764M/431개까지 갔다).
    # 보관 위치는 프로젝트 밖: 대화 로그·사람 데이터와 같은 성격 (사용자).
    try:
        import re, shutil
        data = os.path.join(HERE, '..', 'sim_data')
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
        import para  # PARA 기준 경로
        dst = os.path.join(para.SIM_RUNS, RUN_GENERATION,
                           'run-%s' % time.strftime('%Y%m%d'))
        for name in os.listdir(data):
            if name in ('archive', 'director.log', 'improve_10h.log'):
                continue
            if re.match(r'(improve-|\d{8}-)', name):
                os.makedirs(dst, exist_ok=True)
                shutil.move(os.path.join(data, name), dst)
    except Exception as e:
        print('[정리 건너뜀] %s' % e, flush=True)
    deadline = time.time() + hours * 3600 if hours else None
    cyc = 0
    print('감독 루프: %s (사이클=학습 %d회 + 배치 + 방향 결정)'
          % ('%.1f시간' % hours if hours else '%d사이클' % cycles,
             CYCLE_ITERS), flush=True)
    while True:
        cyc += 1
        if deadline is not None and time.time() >= deadline:
            break
        if deadline is None and cyc > (cycles or 1):
            break
        try:
            # 원장 위생은 사이클 시작에 먼저 - 뒤가 예외로 죽어도 부풀지
            # 않게 (실측: 태스크 생성 예외 7연속에 미결 26건까지 부풂).
            fb0 = I.load_feedback()
            open0 = [i for i, f in enumerate(fb0) if not f.get('done')]
            for i in open0[:-12]:
                fb0[i]['done'] = True
                fb0[i]['note'] += ' [자동정리]'
            if len(open0) > 12:
                I.save_feedback(fb0)
            # 1) 짧은 학습
            summary = I.run(iters=CYCLE_ITERS, seed=seed * 1000 + cyc,
                            token=None if offline else token, url=url,
                            env0=st.get('env'),
                            kinds=tuple(st.get('kinds') or ('pick', 'lift')),
                            natural_min0=int(st.get('natural_min') or 55),
                            reset_params=st.pop('reset_params', None))
            # 2) 실전 배치 (평가 리포트 생성). offline 모드에서는
            # 브로커/비전 모델을 건드리지 않고 계측 학습만 지속한다.
            if offline:
                batch_line = 'offline metric-only (실전 배치 생략)'
                dec = {'why': 'Codex 사용량 제한 없는 결정론 CEM 모드 유지'}
                _save_state(st)
                _log_direction(cyc, summary, batch_line, dec)
                print('[감독] offline 사이클 %d: %s' % (cyc, dec['why']),
                      flush=True)
                continue

            # 실전 배치 (평가 리포트 생성)
            try:
                rid, bsum, flags = PL.run_batch(
                    3, tuple(st.get('kinds') or ('pick', 'lift')), 'broker',
                    url, token, seed=cyc * 37, tag='dir%d' % cyc,
                    include_speech=False)
                batch_line = '%s | 감사 %s | docs/eval/%s.md' % (
                    bsum.get('stages'), (flags[0][:60] if flags else '깨끗'),
                    rid)
                if any('계획 백엔드 오류' in f for f in flags):
                    raise EvaluationUnavailable(flags[0])
            except EvaluationUnavailable:
                raise
            except Exception as e:
                batch_line = '배치 실패: %s' % e
            # 3) 증거 -> 방향 결정
            fb = I.load_feedback()
            try:
                sys.path.insert(0, os.path.abspath(
                    os.path.join(HERE, '..', 'ops', 'data')))
                import knowledge_ledger as KL
                knowledge = KL.digest_for_director()
            except Exception as e:
                knowledge = {'오류': str(e)[:80]}
            evidence = json.dumps({
                '지식 요약(실전+누적)': knowledge,
                '학습': {k: summary.get(k) for k in
                         ('quality_first', 'quality_last', 'exam',
                          'natural_min', 'incidents', 'duels', 'picked')},
                '실패원인 분포(이번 사이클)': summary.get('causes'),
                '실전배치': batch_line,
                '미결지적': {i: f['note'] for i, f in enumerate(fb)
                             if not f.get('done')},
                '운영자 답신': st.get('operator_note'),
            }, ensure_ascii=False)
            strip = _latest(os.path.join(HERE, '..', 'sim_data',
                                         'improve-*-media', '*_strip.jpg'))
            img = None
            if strip:
                import base64
                img = base64.b64encode(open(strip, 'rb').read()).decode()
            dec = client.ask_evaluation(
                DIRECTOR_PROMPT.format(
                    evidence=evidence,
                    direction=json.dumps(
                        {k: st.get(k) for k in
                         ('env', 'kinds', 'natural_min')},
                        ensure_ascii=False)),
                kind='director', image=img)
            if dec is None:
                raise EvaluationUnavailable(
                    client.last_error or '감독 평가 응답 없음')
            # 4) 적용 (한계 안에서)
            if dec.get('env'):
                st['env'] = {k: list(v) if isinstance(v, tuple) else v
                             for k, v in T.clamp_env(dec['env']).items()}
            if dec.get('kinds'):
                ks = [k for k in dec['kinds'] if k in ('pick', 'lift',
                                                       'reach', 'look')]
                if ks:
                    st['kinds'] = ks
            if dec.get('natural_min'):
                st['natural_min'] = max(45, min(75, int(dec['natural_min'])))
            if dec.get('reset_params') == 'defaults':
                st['reset_params'] = 'defaults'
            for note in (dec.get('feedback_add') or [])[:3]:
                if isinstance(note, str) and note.strip() and not any(
                        f['note'] == note.strip() for f in fb):
                    fb.append({'note': note.strip(), 'done': False})
            for idx in (dec.get('feedback_done') or []):
                if isinstance(idx, int) and 0 <= idx < len(fb):
                    fb[idx]['done'] = True
            # 원장 위생: 미결이 12건을 넘으면 오래된 것부터 자동 정리 -
            # 부풀면 비평가는 최근 5건만 보게 되어 나머지가 죽는다.
            open_idx = [i for i, f in enumerate(fb) if not f.get('done')]
            for i in open_idx[:-12]:
                fb[i]['done'] = True
                fb[i]['note'] += ' [자동정리]'
            I.save_feedback(fb)
            _save_state(st)
            _log_direction(cyc, summary, batch_line, dec)
            print('[감독] 사이클 %d: %s' % (cyc, dec.get('why')), flush=True)
        except KeyboardInterrupt:
            break
        except EvaluationUnavailable as e:
            print('[감독 중단] 평가기/접지기 실패: %s' % e, flush=True)
            break
        except Exception as e:
            print('[감독] 사이클 %d 예외(계속): %s: %s'
                  % (cyc, type(e).__name__, e), flush=True)
            time.sleep(5)
    print('감독 종료: %d사이클. 방향 기록: docs/eval/direction-log.md'
          % (cyc - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cycles', type=int)
    ap.add_argument('--hours', type=float)
    ap.add_argument('--token')
    ap.add_argument('--offline', action='store_true',
                    help='브로커 없이 결정론 계측/CEM만 실행 (사용량 제한 없음)')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    run(cycles=args.cycles, hours=args.hours, token=args.token,
        url=args.url, seed=args.seed, offline=args.offline)


if __name__ == '__main__':
    main()
