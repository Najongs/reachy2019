"""행동을 반복 개선하는 루프 - 성공은 문지기, 품질은 목표.

    평가(결정론 지표 + opus 비평) -> 파라미터 후보 생성 -> 같은 태스크로
    재평가 -> 객관 성공이 안 깨진 후보 중 품질 최고를 채택 -> 반복

강화학습의 실용판이다: 보상(품질 점수 + 자연스러움 비평)을 보고 정책
(BEHAVIOR 손잡이)을 갱신한다. 정확히 지키는 것 하나 - **객관 성공/안전은
언어모델이 아니라 계측이 판정하고, 그걸 깨는 후보는 품질이 아무리 좋아도
버린다.** 언어모델은 '어떻게 움직였는가' 만 비평한다.

후보 = 현재 + opus 조언 반영 + 무작위 흔들기. 같은 태스크 묶음으로 비교해야
운이 아니라 파라미터 차이를 잰다.

사용:
    python3 sim/sim_improve.py --iters 3 --token reachy2019
    python3 sim/sim_improve.py --iters 2                # opus 비평 없이 지표만
"""

import copy
import json
import os
import random
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_agent as A
import motion_exec as me                              # noqa: E402
import sim_critic as C                             # noqa: E402
import sim_tasks as T                              # noqa: E402

# 파라미터 허용 범위 - 이 밖은 물리적으로 무의미하거나 위험.
BOUNDS = {'step_deg': (1.5, 8.0), 'gain': (0.3, 1.4), 'damping': (1.0, 10.0),
          'table_min': (0.2, 3.0), 'lift_steps': (3, 14),
          # 경로 구조 - 경유 자세 3개를 통째로 탐색. 투과는 문지기가 거른다.
          'via1_pitch': (-40.0, 15.0), 'via1_roll': (-110.0, -25.0),
          'via1_yaw': (0.0, 60.0), 'via1_elbow': (-125.0, 5.0),
          'via2_pitch': (-40.0, 15.0), 'via2_roll': (-110.0, -25.0),
          'via2_yaw': (0.0, 60.0), 'via2_elbow': (-125.0, 5.0),
          'via3_pitch': (-40.0, 15.0), 'via3_roll': (-110.0, -25.0),
          'via3_yaw': (0.0, 60.0), 'via3_elbow': (-125.0, -60.0),
          'gaze_step_deg': (2.0, 12.0)}
INT_PARAMS = ('lift_steps',)


def _clamp(params):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = max(lo, min(hi, params.get(k, A.BEHAVIOR[k])))
        out[k] = int(round(v)) if k in INT_PARAMS else round(float(v), 2)
    return out


def _jitter(current, rng, scale=1.0):
    """현재 주변의 무작위 후보. 각도형 파라미터는 덧셈, 배율형은 곱셈."""
    j = dict(current)
    for k in BOUNDS:
        if rng.random() < 0.5:
            continue
        if k.startswith('via'):
            span = (BOUNDS[k][1] - BOUNDS[k][0]) * 0.12 * scale
            j[k] = current[k] + rng.uniform(-span, span)
        else:
            j[k] = current[k] * rng.uniform(1 - 0.25 * scale, 1 + 0.3 * scale)
    return j


def episode(task, keep_frames=False, natural_min=55):
    """한 에피소드 -> 3단계 판정. (stage, 품질 dict, frames).

    stage: collision / unnatural / missed / success (sim_critic.stage_verdict).
    충돌이면 그걸로 끝, 자연스러움 문턱 미달이면 도달했어도 성공이 아니다.
    """
    world = T.build_world(task)
    frames = [] if keep_frames else None
    trace = []
    try:
        planner = A.OraclePlanner()
        view = planner.perceive(world, task)
        plan = planner.plan(world, task, view)
        A.execute(world, plan, frames=frames, trace=trace)
        reached = bool(T.success_fn(task)(world))
        final_cm = me._dist(world.hand(),
                            world.object_pos(task['target'])) * 100
        path_tab = min((t['table'] for t in trace), default=99)
        path_obj = min((t['obj'] for t in trace), default=99)
        path_tgt = min((t.get('tgt', 99) for t in trace), default=99)
        q = C.quality(trace) if trace else {'score': 0}
        # look 처럼 팔을 안 쓰는 태스크는 궤적이 없다 - 품질 문턱을 건너뛴다
        # (sim_pipeline._verdict 와 같은 규칙).
        uses_arm = task['kind'] in ('reach', 'point', 'pick')
        stage, why = C.stage_verdict(
            path_tab, path_obj, path_tgt, reached,
            q.get('score', 0) if uses_arm else 100, natural_min)
        q['stage'], q['why'] = stage, why
        q['final_cm'] = round(final_cm, 1)
        q['reached'] = reached
        return stage, q, frames
    finally:
        world.close()


def evaluate(params, tasks, natural_min=55):
    """태스크 묶음 -> {'collision','unnatural','missed','success', 'quality'}."""
    A.BEHAVIOR.update(_clamp(params))
    counts = {k: 0 for k in C.STAGES}
    scores, dists = [], []
    for t in tasks:
        stage, q, _ = episode(t, natural_min=natural_min)
        counts[stage] += 1
        scores.append(q.get('score', 0))
        # 거리는 '부족분' 만 센다. 도달했으면 0 - 도달한 것끼리 0.1cm 를
        # 다투게 두면 거리가 품질(자연스러움)을 영원히 눌러, 비평이
        # 짚어준 과신전·우회가 순위에 반영될 기회를 잃는다 (실제로 그랬다).
        if q.get('reached'):
            dists.append(0.0)
        else:
            dists.append(max(0.0, q.get('final_cm', 99.0) - 14.0))
    counts['quality'] = sum(scores) / max(len(scores), 1)
    counts['miss_cm'] = sum(dists) / max(len(dists), 1)
    return counts


def hold_duel(client, task, params_a, params_b, natural_min, save_to=None):
    """같은 태스크로 A(현재)/B(도전자) 를 돌려 opus 쌍대 심판. 'A'/'B'/None.

    결정론 품질 점수가 DUEL_MARGIN 이내로 갈리는 선택은 점수 잡음 밴드
    안이다 - 그 안에서는 사람 눈(opus 비교)이 가르게 한다.
    """
    saved = dict(A.BEHAVIOR)
    try:
        A.BEHAVIOR.update(_clamp(dict(params_a)))
        _, qa, fa = episode(task, keep_frames=True, natural_min=natural_min)
        A.BEHAVIOR.update(_clamp(dict(params_b)))
        _, qb, fb = episode(task, keep_frames=True, natural_min=natural_min)
    finally:
        A.BEHAVIOR.update(saved)
    r = C.duel(client, fa, fb, task['instruction'], save_path=save_to)
    keep = ('efficiency', 'jerk_deg', 'min_margin_cm', 'view_ratio',
            'steps', 'score')
    detail = {'a': {k: qa.get(k) for k in keep},
              'b': {k: qb.get(k) for k in keep}}
    return (r or {}).get('winner'), detail


def cem_sample(center, sigma, rng, n, scale=1.0):
    """CEM: 분포 N(center, sigma*scale) 에서 후보 n 개.

    한 점 주변 무작위 흔들기(_jitter)는 구석에 갇히면 못 나온다 - 1099회
    정체의 한 원인. CEM 은 상위 후보들의 분포로 다음 세대를 만들어
    탐색 방향 자체가 학습된다. sim_opt 의 키프레임 CEM 과 같은 원리.
    """
    out = []
    for i in range(n):
        c = {k: rng.gauss(center[k], sigma[k] * scale) for k in BOUNDS}
        out.append(('cem%d' % i, _clamp(c)))
    return out


def propose(current, advice, rng, n_random=3, scale=1.0):
    """후보들: 현재 / 조언 반영 / 무작위 흔들기(scale 로 폭 조절)."""
    cands = [('current', dict(current))]
    if advice:
        adv = dict(current)
        for k, direction in advice.items():
            if k not in BOUNDS:
                continue
            if k.startswith('via'):
                span = (BOUNDS[k][1] - BOUNDS[k][0]) * 0.15
                adv[k] = current[k] + (span if direction == 'up' else -span)
            else:
                adv[k] = current[k] * (1.25 if direction == 'up' else 0.8)
        cands.append(('advice', adv))
    for i in range(n_random):
        cands.append(('jitter%d' % i, _jitter(current, rng, scale)))
    return [(name, _clamp(c)) for name, c in cands]


REFRESH_EVERY = 6       # 이 횟수마다 태스크 묶음을 새로 뽑는다 (과적합 방지)
VALIDATE_EVERY = 60     # end-to-end 검증 주기
CRITIQUE_EVERY = 3      # opus 비평 주기 (10시간 기준 시간당 ~20콜 수준)
DUEL_MARGIN = 4.0       # 이 품질 차 이내면 '박빙' - opus 쌍대 결투로 심판


def run(iters=3, n_tasks=4, seed=0, token=None, url='http://127.0.0.1:8080',
        hours=None):
    """개선 루프. hours 를 주면 그 시간 동안 계속 돈다 (iters 무시)."""
    rng = random.Random(seed)
    deadline = time.time() + hours * 3600 if hours else None
    label = '%.1f시간' % hours if hours else '%d회' % iters
    print('개선 루프: %s, 태스크 %d개 (%d회마다 교체)'
          % (label, n_tasks, REFRESH_EVERY), flush=True)

    client = None
    if token:
        from llm_client import BrokerClient
        client = BrokerClient(url, token=token, session='sim-critic')

    def new_tasks(k, fallback=None):
        """실현 가능한 태스크 n_tasks 개. 모자라면 시드를 바꿔 재시도.

        빈 목록이 그대로 흘러가면 '성공 0/0' 이 전원 성공(공허참)으로 통해
        래칫이 폭주한다 - 실제로 겪었다. 끝내 못 만들면 이전 것을 유지.
        """
        for tryn in range(6):
            ts = T.generate_feasible(
                n_tasks, seed=seed * 977 + k + tryn * 131071,
                kinds=('reach',))
            if len(ts) >= n_tasks:
                return ts
        if fallback:
            print('  태스크 생성 부족(%d) - 기존 유지' % len(ts), flush=True)
            return fallback
        if not ts:
            raise RuntimeError('실현 가능한 태스크를 만들 수 없음')
        return ts

    # 시험지(고정)와 훈련셋(회전)을 가른다. 태스크 교체 충격으로 점수가
    # 널뛰고 성공 0 붕괴의 방아쇠가 됐던 것 - 시험지는 절대 안 바뀌므로
    # 추이가 연속이고, 훈련셋 회전은 과적합만 막는다.
    exam = new_tasks(-7)                    # 고정 시험지
    tasks = new_tasks(0)                    # 회전 훈련셋
    history = []
    current = _clamp(dict(A.BEHAVIOR))
    # CEM 분포: 중심=엘리트 평균, 폭=엘리트 분산 (하한/상한 있음)
    span = {k: BOUNDS[k][1] - BOUNDS[k][0] for k in BOUNDS}
    center = dict(current)
    sigma = {k: 0.10 * span[k] for k in BOUNDS}
    # 자연스러움 문턱 - 전원 성공이 이어지면 올라간다(평가도 함께 개선).
    natural_min = 55

    def rank(counts):
        """충돌 적게 > 성공 많이 > 목표에 가깝게 > 품질 높게.

        거리는 **부족분**(도달 못한 만큼)이다. 도달하면 0 이라 전원
        성공이면 거리 동률 -> 품질이 결정한다. 실패 중에는 '더 가까이'
        압력이 살아 있다. 날거리로 두면 성공끼리 0.1cm 를 다투느라
        품질(비평이 짚는 과신전·우회)이 영원히 발언권을 잃는다.
        """
        return (-counts['collision'], counts['success'],
                -round(counts.get('miss_cm', 99.0), 1), counts['quality'])

    base = evaluate(current, tasks, natural_min)
    report = evaluate(current, exam, natural_min)
    best_ever = {'quality': report['quality'], 'success': report['success'],
                 'params': dict(current), 'natural_min': natural_min}
    stagnant = 0
    perfect_streak = 0
    duel_flips = 0      # opus 가 계측 선택을 뒤집은 횟수 (보정 감사 신호)
    zero_streak = 0     # 성공 0 연속 - 지속되면 best_ever 로 복귀
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = os.path.join(HERE, '..', 'sim_data', 'improve-%s.json' % stamp)
    # 개선 과정을 눈으로 볼 수 있게 - 비평 에피소드 영상/스트립, 결투 비교.
    media = os.path.join(HERE, '..', 'sim_data', 'improve-%s-media' % stamp)
    os.makedirs(media, exist_ok=True)
    print('과정 영상: %s' % os.path.normpath(media), flush=True)

    def checkpoint():
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump({'history': history, 'best_ever': best_ever},
                      fh, ensure_ascii=False, indent=1)

    def fmt(c):
        return '충돌%d 부자연%d 미도달%d 성공%d 품질%.0f' % (
            c['collision'], c['unnatural'], c['missed'], c['success'],
            c['quality'])

    print('반복 0 (기준, 문턱 %d): 훈련 %s | 시험 %s'
          % (natural_min, fmt(base), fmt(report)), flush=True)
    history.append({'iter': 0, 'counts': {k: base[k] for k in C.STAGES},
                    'quality': round(base['quality'], 1),
                    'natural_min': natural_min,
                    'params': dict(current), 'ts': time.strftime('%H:%M')})

    it = 0
    while True:
        it += 1
        if deadline is not None:
            if time.time() >= deadline:
                break
        elif it > iters:
            break

        try:
            # 태스크 교체: 같은 4개에 과적합되지 않게. 교체하면 기준 재측정.
            if it % REFRESH_EVERY == 0:
                tasks = new_tasks(it, fallback=tasks)
                base = evaluate(current, tasks, natural_min)
                print('[%s] 반복 %d: 태스크 교체 -> %s'
                      % (time.strftime('%H:%M'), it, fmt(base)), flush=True)

            # opus 비평 (주기적으로; 실패해도 루프는 계속)
            advice, naturalness, rubric = None, None, None
            if client is not None and it % CRITIQUE_EVERY == 1:
                try:
                    A.BEHAVIOR.update(current)
                    ok, q, frames = episode(tasks[0], keep_frames=True)
                    # 전 과정을 필름 스트립 한 장으로 - 중간 프레임까지 다 본다.
                    crit = C.critique(client, frames,
                                      tasks[0]['instruction'], q)
                    if crit:
                        naturalness = crit.get('naturalness')
                        rubric = crit.get('rubric')
                        advice = crit.get('advice') or None
                    if frames:
                        A._write_video(frames, os.path.join(
                            media, 'iter%04d.mp4' % it))
                        C.save_strip(frames, os.path.join(
                            media, 'iter%04d_strip.jpg' % it))
                except Exception as e:
                    print('  비평 실패(계속 진행): %s' % e, flush=True)

            # 정체하면 탐색 폭을 넓힌다 (담금질). 개선되면 되돌린다.
            scale = min(3.0, 1.0 + stagnant * 0.35)

            base_rank = rank(base)
            best = (base_rank, base, 'current', dict(current))
            runner = None       # 결투 자격 도전자: 충돌·성공 계수가 같은 최고
            pool = cem_sample(center, sigma, rng, 5, scale)
            if advice:
                pool += [pc for pc in propose(current, advice, rng, 0, scale)
                         if pc[0] == 'advice']
            scored = [(base_rank, base, 'current', dict(current))]
            for name, cand in pool:
                c = evaluate(cand, tasks, natural_min)
                # 문지기: 충돌은 기준보다 늘 수 없다 (안전 후퇴 금지).
                if c['collision'] > base['collision']:
                    continue
                scored.append((rank(c), c, name, cand))
                if rank(c) > best[0]:
                    best = (rank(c), c, name, cand)
                if (rank(c)[:2] == base_rank[:2]
                        and (runner is None or rank(c) > runner[0])):
                    runner = (rank(c), c, name, cand)
            # CEM 분포 갱신: 상위 3(엘리트)의 평균/표준편차로.
            elites = [sc[3] for sc in sorted(scored, reverse=True,
                                             key=lambda x: x[0])[:3]]
            for k in BOUNDS:
                vals = [e[k] for e in elites]
                mu = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / len(vals)
                center[k] = mu
                sigma[k] = min(0.25 * span[k],
                               max(0.02 * span[k], var ** 0.5 * 1.15))

            # 쌍대 결투: 결정론 품질이 박빙일 때 opus 가 A/B 로 가른다.
            # 계측이 근소하게 고른 승자를 사람 눈이 거부하면 채택을 물리고
            # (거부권), 근소하게 진 도전자를 사람 눈이 고르면 승격시킨다.
            duel_pick, duel_detail = None, None
            challenger = best if best[2] != 'current' else runner
            if (client is not None and challenger is not None
                    and challenger[2] != 'current'
                    and challenger[0][:2] == base_rank[:2]
                    and abs(challenger[1]['quality'] - base['quality'])
                    <= DUEL_MARGIN):
                try:
                    duel_pick, duel_detail = hold_duel(
                        client, tasks[0], current, challenger[3], natural_min,
                        save_to=os.path.join(media, 'iter%04d_duel.jpg' % it))
                    if duel_pick:
                        duel_detail['winner'] = duel_pick
                except Exception as e:
                    print('  결투 실패(계속 진행): %s' % e, flush=True)
                if duel_pick == 'B' and best[2] == 'current':
                    best = challenger                    # 승격
                    duel_flips += 1
                elif duel_pick == 'A' and best[2] != 'current':
                    best = (base_rank, base, 'current', dict(current))  # 거부권
                    duel_flips += 1

            improved = best[0] > base_rank or (
                duel_pick == 'B' and best[2] != 'current')
            stagnant = 0 if improved else stagnant + 1
            base = best[1]
            current = best[3]
            A.BEHAVIOR.update(current)
            A.save_behavior(current)

            # 시험지 채점: 훈련셋은 선택용, 기록·래칫·최고 갱신은 전부
            # 고정 시험지로 - 추이가 태스크 교체에 흔들리지 않는다.
            report = evaluate(current, exam, natural_min)
            if (report['collision'] == 0 and report['success'] >= len(exam)
                    and report['quality'] > best_ever['quality']):
                best_ever = {'quality': report['quality'],
                             'success': report['success'],
                             'params': dict(current), 'iter': it,
                             'natural_min': natural_min}

            # 탈출구: 시험지 성공 0 이 오래가면 못 도달하는 구석에 갇힌
            # 것이다. 전원 성공이었던 best_ever 로 되돌려 다시 시작.
            zero_streak = zero_streak + 1 if report['success'] == 0 else 0
            if zero_streak >= 10 and best_ever.get('params'):
                current = _clamp(dict(best_ever['params']))
                A.BEHAVIOR.update(current)
                A.save_behavior(current)
                center = dict(current)
                sigma = {k: 0.10 * span[k] for k in BOUNDS}
                base = evaluate(current, tasks, natural_min)
                report = evaluate(current, exam, natural_min)
                zero_streak = 0
                stagnant = 0
                print('[%s] 시험 성공 0 지속 - best_ever(반복 %s) 로 복귀'
                      ' -> 시험 %s' % (time.strftime('%H:%M'),
                                       best_ever.get('iter', '?'),
                                       fmt(report)), flush=True)

            # 평가 래칫: 시험지 전원 성공이 2회 이어지면 문턱을 올린다.
            if (exam and report['collision'] == 0
                    and report['success'] >= len(exam)):
                perfect_streak += 1
                if perfect_streak >= 2 and natural_min < 85:
                    natural_min += 5
                    perfect_streak = 0
                    base = evaluate(current, tasks, natural_min)
                    report = evaluate(current, exam, natural_min)
                    print('[%s] 평가 강화: 문턱 -> %d (시험 재측정 %s)'
                          % (time.strftime('%H:%M'), natural_min,
                             fmt(report)), flush=True)
            else:
                perfect_streak = 0

            history.append({'iter': it,
                            'counts': {k: base[k] for k in C.STAGES},
                            'quality': round(base['quality'], 1),
                            'exam': {k: report[k] for k in C.STAGES},
                            'exam_quality': round(report['quality'], 1),
                            'exam_miss_cm': round(report['miss_cm'], 1),
                            'natural_min': natural_min,
                            'naturalness': naturalness, 'rubric': rubric,
                            'duel': duel_pick, 'duel_detail': duel_detail,
                            'picked': best[2],
                            'stagnant': stagnant,
                            'params': dict(current),
                            'ts': time.strftime('%H:%M')})
            # 발산 감사: 품질은 오르는데 시험 성공이 내리면 목적함수 오용
            # 신호다 - 1099회 사고의 패턴을 이제 기계가 감시한다.
            if it % 15 == 0 and len(history) >= 31:
                older = history[-31:-16]
                newer = history[-15:]
                s_old = sum(h2['exam']['success'] for h2 in older
                            if 'exam' in h2) / max(len(older), 1)
                s_new = sum(h2['exam']['success'] for h2 in newer
                            if 'exam' in h2) / max(len(newer), 1)
                q_old = sum(h2.get('exam_quality', 0) for h2 in older) / max(len(older), 1)
                q_new = sum(h2.get('exam_quality', 0) for h2 in newer) / max(len(newer), 1)
                if s_new < s_old - 0.5 and q_new > q_old + 2:
                    print('[감사] 발산 의심: 시험 성공 %.1f->%.1f 인데 품질'
                          ' %.0f->%.0f - 목적함수/지표 오용 점검 필요'
                          % (s_old, s_new, q_old, q_new), flush=True)
            # 결투가 계측 선택을 자꾸 뒤집으면 품질 가중치가 사람 눈과
            # 어긋난 것이다 - 지표 감사 원칙대로 신호를 남긴다.
            if duel_flips == 3:
                duel_flips += 1     # 한 번만 알린다
                print('[감사] 결투가 계측 선택을 3회 뒤집음 - quality() '
                      '가중치(효율/저크)가 사람 눈과 어긋날 수 있음', flush=True)
            nat = ' 자연 %s/10' % naturalness if naturalness else ''
            if duel_pick:
                nat += ' 결투:%s' % duel_pick
            print('[%s] 반복 %d(문턱 %d): %s 채택, 훈련 %s | 시험 %s%s%s'
                  % (time.strftime('%H:%M'), it, natural_min, best[2],
                     fmt(base), fmt(report), nat,
                     ' (정체 %d)' % stagnant if stagnant else ''), flush=True)
            checkpoint()

            # 주기적 end-to-end 검증: opus 접지 포함 실전 배치가 여전히 되나.
            if client is not None and it % VALIDATE_EVERY == 0:
                try:
                    import sim_pipeline as PL
                    rid, summ, flags = PL.run_batch(
                        4, ('look', 'reach'), 'broker', url, token,
                        seed=it, tag='improve-check%d' % it)
                    print('[%s] end-to-end 검증: %d/%d, 감사 %s'
                          % (time.strftime('%H:%M'), summ['success'],
                             summ['n'], '깨끗' if not flags else flags[0][:40]),
                          flush=True)
                except Exception as e:
                    print('  검증 실패(계속 진행): %s' % e, flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # 장기 실행: 어떤 예외도 루프를 죽이지 않는다. 기록하고 계속.
            print('[%s] 반복 %d 예외(계속): %s: %s'
                  % (time.strftime('%H:%M'), it, type(e).__name__, e),
                  flush=True)
            time.sleep(5)

    checkpoint()
    qs = [h.get('exam_quality', h['quality']) for h in history]
    print('품질 추이: %.0f -> %.0f (최고 %.0f, 최종 문턱 %d, 반복 %d회)'
          % (qs[0], qs[-1], best_ever['quality'], natural_min,
             len(history) - 1))
    print('기록: %s  |  채택 파라미터: config/behavior_params.json' % out)
    return history


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--hours', type=float, help='시간 예산 (주면 iters 무시)')
    ap.add_argument('--tasks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--token')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    args = ap.parse_args()
    run(args.iters, args.tasks, args.seed, args.token, args.url,
        hours=args.hours)


if __name__ == '__main__':
    main()
