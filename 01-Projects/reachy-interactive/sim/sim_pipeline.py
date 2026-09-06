"""7-역할 파이프라인의 최상위 러너 - 한 배치를 끝까지 돌린다.

  1 명령    sim_tasks (템플릿) / 지시문
  2 환경    sim_tasks.generate_feasible - 실현가능성 필터 통과한 장면만
  3 답변    ollama /reply 한 줄 ("네, 컵으로 손을 가져갈게요")
  4 동작    opus /vision 접지 -> 결정론 서보 (sim_agent.BrokerPlanner)
  5 피드백  객관 검증 실패 시 지적을 만들어 1회 재시도 (lessons 되먹임)
  6 오케스트라  배치 끝에 지표 감사 + 커리큘럼 + opus 총평 1회
  7 데이터  시도별 레코드 -> sim_data/<run>/ -> docs/eval/<run>.md

사용:
    python3 sim/sim_pipeline.py -n 6 --planner oracle          # harness 검증
    python3 sim/sim_pipeline.py -n 6 --planner broker --token reachy2019
"""

import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')   # GPU0 만 쓴다

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                           # noqa: E402
import sim_agent as A                              # noqa: E402
import sim_critic as C                             # noqa: E402
import sim_orchestra as O                          # noqa: E402
import sim_record as R                             # noqa: E402
import sim_tasks as T                              # noqa: E402


def _answer_line(client, instruction):
    """역할 3: ollama 가 지시에 짧게 응답 (로봇이 말할 한 줄)."""
    if client is None:
        return None
    out = client.ask('다음 지시에 로봇으로서 한 문장으로만 응답해: ' + instruction)
    return (out or '').strip() or None


def run_episode(task, planner, run_id, answer_client=None, retry=True):
    """한 에피소드: 인지->계획->행동->검증(->재시도)->레코드."""
    world = T.build_world(task)
    frames = []
    rec = {'task': task, 'instruction': task['instruction'], 'seed': task.get('seed')}

    try:
        rec['answer'] = _answer_line(answer_client, task['instruction'])

        view = planner.perceive(world, task)
        dets = view.get('dets') or []
        mem = view.get('memory')
        rec['perception'] = {
            'seen': bool(view.get('seen')),
            'dets': [{k: round(v, 1) if isinstance(v, float) else v
                      for k, v in d.items() if k != 'name'} for d in dets],
            'memory': mem.summary() if mem is not None else None,
        }

        plan = planner.plan(world, task, view)
        rec['plan'] = {'mode': plan.get('mode'), 'raw': plan.get('raw'),
                       'why': plan.get('why')}

        trace = []
        A.execute(world, plan, frames=frames, trace=trace)
        verdict = _verdict(world, task, trace)

        # 역할 5: 실패면 지적을 만들어 1회 재시도. 지적은 결정론(검증 결과)에서.
        if retry and not verdict['success'] and plan.get('mode') != 'noop':
            finding = verdict.get('why') or '목표에 닿지 못함'
            if hasattr(planner, 'lessons'):
                planner.lessons.append(finding)
            view2 = planner.perceive(world, task)
            plan2 = planner.plan(world, task, view2)
            A.execute(world, plan2, frames=frames, trace=trace)
            v2 = _verdict(world, task, trace)
            if v2['success']:
                verdict = v2
                rec['plan']['retry'] = plan2.get('mode')
            verdict['retried'] = True

        rec['verdict'] = verdict
        rec['execution'] = {
            'final_dist_cm': round(me._dist(
                world.hand(), world.object_pos(task['target'])) * 100, 1),
            'frames': len(frames),
            # 경로 전체의 최소 여유 - 최종 상태만 보면 중간 투과를 놓친다.
            'table_min_cm': round(min((t['table'] for t in trace), default=99), 1),
            'object_min_cm': round(min((t['obj'] for t in trace), default=99), 1),
            # 품질(효율·저크·여유) - 성공만으로는 거칠게 스친 동작과 부드럽게
            # 돌아간 동작을 못 가른다. sim_improve 가 이 값을 보고 행동을 조인다.
            'quality': C.quality(trace) if trace else None,
        }
        rec['findings'] = ([] if verdict['success']
                           else [verdict.get('why') or '실패'])
    except Exception as e:
        rec['verdict'] = {'success': False, 'why': '%s: %s'
                          % (type(e).__name__, e)}
        rec['findings'] = [rec['verdict']['why']]
    finally:
        art = {}
        if frames:
            vid = os.path.join(R.run_dir(run_id), task['id'] + '.mp4')
            try:
                A._write_video(frames, vid)
                art['video'] = vid
            except Exception:
                pass
        rec['artifacts'] = art
        world.close()
    R.write(run_id, rec)
    return rec


NATURAL_MIN = 55        # 자연스러움 문턱 (sim_improve 의 래칫과 같은 시작값)


def _verdict(world, task, trace=None):
    """3단계 판정: 1 충돌 없이 -> 2 자연스럽게 -> 3 성공.

    순서가 정의다. 경로 어디서든 충돌이면 그걸로 끝이고, 도달했어도 품질이
    문턱 미달이면 '부자연' 이지 성공이 아니다. look 처럼 팔을 안 쓰는
    태스크는 품질 문턱을 건너뛴다(움직임이 없으니 잴 것도 없다).
    """
    reached = bool(T.success_fn(task)(world))
    obj_gap, obj_pair = world.clearance(ignore=('table',))
    path_min = min(min((t['table'] for t in (trace or [])), default=99),
                   min((t['obj'] for t in (trace or [])), default=99))
    q = C.quality(trace) if trace else None
    uses_arm = task['kind'] in ('reach', 'point', 'pick')
    qscore = (q or {}).get('score', 100 if not uses_arm else 0)

    stage, why = C.stage_verdict(path_min, reached,
                                 qscore if uses_arm else 100, NATURAL_MIN)
    out = {'stage': stage, 'stage_ko': C.STAGE_KO[stage],
           'success': stage == 'success',
           'object_clearance_cm': round(obj_gap, 1),
           'path_min_cm': round(path_min, 1)}
    if why:
        out['why'] = why
    elif stage != 'success' and not reached:
        d = me._dist(world.hand(), world.object_pos(task['target'])) * 100
        out['why'] = '판정 미달 (손-대상 %.0fcm)' % d
    return out


def run_batch(n=6, kinds=('look', 'reach'), planner_name='oracle',
              url='http://127.0.0.1:8080', token=None, seed=0, tag=''):
    run_id = R.new_run(tag or planner_name)
    print('run %s | 계획자 %s | 태스크 %d개 생성(실현가능성 필터)...'
          % (run_id, planner_name, n))
    tasks = T.generate_feasible(n, seed=seed, kinds=kinds)

    answer_client = None
    if planner_name == 'broker':
        from llm_client import BrokerClient
        planner = A.BrokerPlanner(url, token=token, seed=seed)
        answer_client = BrokerClient(url, token=token, session='sim-answer')
    else:
        planner = A.OraclePlanner()

    records = []
    for t in tasks:
        rec = run_episode(t, planner, run_id, answer_client=answer_client)
        v = rec['verdict']
        ans = (' | "%s"' % rec['answer'][:30]) if rec.get('answer') else ''
        print('  %s [%-5s] %-34s [%s] %s%s' % (
            '✓' if v['success'] else '✗', t['kind'], t['instruction'][:34],
            v.get('stage_ko', '?'), v.get('why', '')[:36], ans))
        records.append(rec)

    # 역할 6+7: 감사 -> 커리큘럼 -> 총평 -> 리포트
    summary = R.aggregate(run_id)
    flags = O.audit(records)
    cur = O.curriculum(summary)
    review = O.review(summary, flags,
                      client=answer_client if planner_name == 'broker' else None)

    orchestra_md = '\n'.join(
        ['**지표 감사**: ' + ('깨끗함' if not flags else '')] +
        ['- ⚠ ' + f for f in flags] +
        ['', '**커리큘럼**: ' + cur['note']] +
        (['', '**opus 총평**: ' + review] if review else []))
    doc = R.report(run_id, orchestra=orchestra_md)

    print('\n성공 %d/%d | 감사 %s | %s'
          % (summary['success'], summary['n'],
             '깨끗' if not flags else '⚠ %d건' % len(flags), cur['note']))
    for f in flags:
        print('  ⚠ %s' % f)
    print('리포트: %s' % os.path.relpath(doc))
    return run_id, summary, flags


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-n', type=int, default=6)
    ap.add_argument('--kinds', default='look,reach')
    ap.add_argument('--planner', choices=['oracle', 'broker'], default='oracle')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--token')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    run_batch(args.n, tuple(k.strip() for k in args.kinds.split(',')),
              args.planner, args.url, args.token, args.seed, args.tag)


if __name__ == '__main__':
    main()
