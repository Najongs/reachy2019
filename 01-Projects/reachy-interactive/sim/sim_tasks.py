"""태스크 예시들을 만들어 둔다 - MuJoCo 장면 + 지시문 + 성공 판정.

파이프라인의 목표는 이 태스크들을 문제없이 수행하는 것이다. 한 태스크는:
  scene         테이블 위 물체 목록 (sim_world.make_object)
  instruction   로봇에게 주는 한국어 지시 (접지 모델이 읽는다)
  kind          유형 (look/point/reach/pick) - 성공 판정과 계획에 쓴다
  target        관련 물체 이름
  success(world) 실행 뒤 성공했는지 (참값으로 판정)

유형:
  look    지정 물체를 시선 중앙에 담기        (목/시선만)
  point   지정 물체 쪽으로 팔을 뻗어 가리키기  (팔)
  reach   손끝을 물체 근처로 가져가기          (팔)
  pick    물체를 집어 쟁반에 놓기              (팔, 어려움)

성공 판정은 전부 참값 기반이다 - 실물이 아니라 '시뮬에서 태스크가 됐나' 를
객관적으로 재기 위한 것. 접지 모델은 이 참값을 보지 못하고 카메라만 본다.

사용:
    python3 sim/sim_tasks.py --list          # 예시 태스크를 만들어 보여준다
    python3 sim/sim_tasks.py --out tasks.json
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_world as world                          # noqa: E402
import motion_exec as me                           # noqa: E402

# 오른손 작업 영역(책상 오른쪽). 물체는 이 안에 놓는다.
X_RANGE = (0.26, 0.35)
Y_RANGE = (-0.24, 0.06)
TRAY_XY = (0.30, 0.15)          # 왼쪽 보관함 자리


def _rng(seed):
    import random
    return random.Random(seed)


def _place_objects(rng, kinds, min_gap=0.09, table_top=None):
    """물체들을 겹치지 않게 작업 영역에 흩는다."""
    placed = []
    for i, kind in enumerate(kinds):
        for _try in range(200):
            xy = (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE))
            if all(math.hypot(xy[0] - p[1][0], xy[1] - p[1][1]) > min_gap
                   for p in placed):
                placed.append((kind, xy))
                break
        else:
            placed.append((kind, xy))
    tt = world.TABLE_TOP if table_top is None else table_top
    return [world.make_object('%s%d' % (k, n), k, xy, table_top=tt)
            for n, (k, xy) in enumerate(placed)]


# --- 성공 판정들 (참값) -----------------------------------------------------

def _succ_look(name):
    def f(w):
        # 물체가 시선 화면의 가운데 30% 안에 오면 성공.
        px = w.pixel_of(w.object_pos(name))
        if px is None:
            return False
        return (abs(px[0] - w.width / 2) < w.width * 0.15
                and abs(px[1] - w.height / 2) < w.height * 0.15)
    return f


def _clean(w, name):
    """안전 판정: 다른 물체는 스침도 금지, 대상은 관통(-1cm)만 금지."""
    others = w.clearance(ignore=('table', name))[0] > -0.5
    return others and w.clearance_of(name) > -1.0


def _succ_point(name):
    def f(w):
        # 손끝에서 어깨로의 방향이 물체를 향하고, 팔이 뻗어 있으면 성공.
        hand = w.hand()
        obj = w.object_pos(name)
        shoulder = (0.0, -0.19, 0.0)
        v_arm = [hand[i] - shoulder[i] for i in range(3)]
        v_obj = [obj[i] - shoulder[i] for i in range(3)]
        na = math.sqrt(sum(c * c for c in v_arm)) or 1
        no = math.sqrt(sum(c * c for c in v_obj)) or 1
        cos = sum(v_arm[i] * v_obj[i] for i in range(3)) / (na * no)
        reach = math.sqrt(sum((hand[i] - shoulder[i]) ** 2 for i in range(3)))
        return cos > 0.94 and reach > 0.35 and _clean(w, name)
    return f


def _succ_reach(name, within=0.14):
    def f(w):
        # 손이 물체 근처에 오되, 물체를 뚫지 않아야 진짜 성공.
        near = me._dist(w.hand(), w.object_pos(name)) < within
        return near and _clean(w, name)
    return f


def _succ_pick(name, tray):
    def f(w):
        op = w.object_pos(name)
        tp = w.object_pos(tray)
        if tray.startswith('basket'):
            # 바구니: 안쪽에 들어가야 성공 - 내벽 안 + 바닥 위 + 테두리 근처까지.
            hx, hy, hz = world.OBJECT_LIBRARY['basket']['size']
            wall = world.BASKET_WALL
            return (abs(op[0] - tp[0]) < hx - wall * 2
                    and abs(op[1] - tp[1]) < hy - wall * 2
                    and tp[2] < op[2] < tp[2] + hz * 2 + 0.06)
        return (abs(op[0] - tp[0]) < 0.08 and abs(op[1] - tp[1]) < 0.07
                and op[2] > w.table_top + 0.02)
    return f


def _succ_lift(name):
    def f(w):
        # 물체가 테이블에서 확실히 들려 있고(중심 +12cm), 손에 쥐어져 있다.
        op = w.object_pos(name)
        return op[2] > w.table_top + 0.12 and w.clearance_of(name) < 3.0
    return f


SUCCESS = {'look': _succ_look, 'point': _succ_point,
           'reach': _succ_reach, 'pick': _succ_pick, 'lift': _succ_lift}


# --- 태스크 생성 ------------------------------------------------------------

KO = {'cup': '컵', 'block': '블록', 'ball': '공', 'can': '캔',
      'bottle': '병', 'book': '책', 'apple': '사과'}


# 환경 손잡이 기본값과 하드 한계. 오케스트라가 experiment 로 조절하되
# 한계 밖은 코드가 자른다 - 모델 제안이 물리적으로 말이 안 되는 환경을
# 만들 수 없게 (지표 감사 원칙의 환경판).
ENV_DEFAULT = {'table': (-0.36, -0.27), 'objects': (1, 3), 'min_gap': 0.09}
ENV_HARD = {'table': (-0.40, -0.24), 'objects': (1, 4),
            'min_gap': (0.05, 0.14)}


def clamp_env(env):
    """오케스트라 제안 환경을 하드 한계 안으로."""
    out = dict(ENV_DEFAULT)
    if not isinstance(env, dict):
        return out
    try:
        if 'table' in env:
            lo, hi = sorted(float(v) for v in env['table'][:2])
            out['table'] = (max(ENV_HARD['table'][0], lo),
                            min(ENV_HARD['table'][1], hi))
        if 'objects' in env:
            lo, hi = sorted(int(v) for v in env['objects'][:2])
            out['objects'] = (max(ENV_HARD['objects'][0], lo),
                              min(ENV_HARD['objects'][1], hi))
        if 'min_gap' in env:
            g = float(env['min_gap'])
            out['min_gap'] = min(ENV_HARD['min_gap'][1],
                                 max(ENV_HARD['min_gap'][0], g))
    except (TypeError, ValueError, IndexError):
        pass
    return out


def generate(n=8, seed=0, env=None):
    """다양한 태스크 n 개. env 로 환경(높이/물체수/간격)을 조절한다."""
    rng = _rng(seed)
    e = clamp_env(env) if env else dict(ENV_DEFAULT)
    kinds_pool = ['cup', 'block', 'ball', 'can', 'bottle', 'book', 'apple']
    tasks = []
    templates = ['look', 'point', 'reach', 'pick', 'lift']
    for i in range(n):
        kind = templates[i % len(templates)]
        n_obj = rng.randint(*e['objects'])
        chosen = [rng.choice(kinds_pool) for _ in range(n_obj)]
        # 테이블 높이도 환경 변수다 - 낮을수록 넘을 턱이 낮다.
        tt = round(rng.uniform(*e['table']), 3)
        light = [round(rng.uniform(-0.6, 1.3), 2),
                 round(rng.uniform(-1.0, 1.0), 2),
                 round(rng.uniform(1.0, 2.2), 2)]
        scene = _place_objects(rng, chosen, min_gap=e['min_gap'],
                               table_top=tt)
        # 파지 포켓 정합 (감독 지적: 시험 시드 40개 전부 포켓 밖이라
        # 성공이 구조적으로 0). 잡기류 표적의 6할은 실측 스위트 창에
        # 두어 '가능한' 과제가 늘 섞이게 한다 - 나머지 4할은 전 범위로
        # 다양성을 지킨다.
        if kind in ('pick', 'lift') and rng.random() < 0.6:
            sxy = (round(rng.uniform(*SWEET_X), 3),
                   round(rng.uniform(*SWEET_Y), 3))
            first = scene[0]
            scene[0] = world.make_object(first['name'], first['kind'], sxy,
                                         table_top=tt)
        dest = None
        if kind == 'pick':
            # 목적지는 쟁반(위에 올리기) 또는 바구니(테두리 넘겨 넣기).
            dkind = rng.choice(['tray', 'basket'])
            dest = dkind + '0'
            scene.append(world.make_object(dest, dkind, TRAY_XY,
                                           table_top=tt))
        target = scene[0]['name']
        tko = KO.get(scene[0]['kind'], '물건')
        instr = {
            'look': '책상 위 %s 을(를) 똑바로 바라봐.' % tko,
            'point': '%s 이(가) 있는 쪽을 손으로 가리켜 봐.' % tko,
            'reach': '%s 에 손을 가까이 가져가 봐.' % tko,
            'pick': ('%s 을(를) 집어서 왼쪽 바구니에 넣어 줘.' % tko
                     if dest == 'basket0' else
                     '%s 을(를) 집어서 왼쪽 쟁반에 놓아 줘.' % tko),
            'lift': '%s 을(를) 잡아서 들어 올려 봐.' % tko,
        }[kind]
        tasks.append({
            'id': 'task%02d_%s' % (i, kind),
            'table_top': tt,
            'light': light,
            'kind': kind,
            'instruction': instr,
            'target': target,
            'tray': dest,
            'scene': [{'name': o['name'], 'kind': o['kind'],
                       'xy': [round(o['pos'][0], 3), round(o['pos'][1], 3)]}
                      for o in scene],
        })
    return tasks


# 팔 바로 앞의 '잡기 좋은 창' - 커리큘럼용 (실측 슬롯 잔차가 작던 영역)
SWEET_X = (0.28, 0.33)
SWEET_Y = (-0.20, -0.06)


def generate_sweet(n, seed=0, kinds=('pick', 'lift'), env=None):
    """스위트스팟 커리큘럼: 팔의 실현가능 포켓 안 쉬운 태스크들.

    실현가능 필터가 빈손일 때 무작위 어려운 태스크로 훈련하면 성공 0 에
    신호가 부족분 거리뿐이다. 잡기 좋은 자리·중간 원통·기본 높이로 성공
    밀도를 올려 파지 신호(맞물림/정렬)가 흐르게 한다. 판정은 그대로.
    """
    rng = _rng(seed)
    tasks = []
    # 현재 그리퍼 기하에서 Oracle 8/8 기준선이 성립하는 가장 작은
    # 커리큘럼은 컵이다. 캔/병은 파지 폭·깊이 단계에서 따로 승격한다.
    easy_kinds = ['cup']
    kinds = list(kinds) or ['pick', 'lift']
    for i in range(n):
        kind = kinds[i % len(kinds)]
        okind = easy_kinds[i % len(easy_kinds)]
        tt = world.TABLE_TOP                    # 기본 높이 (턱 변주 없음)
        xy = (round(rng.uniform(*SWEET_X), 3),
              round(rng.uniform(*SWEET_Y), 3))
        scene = [world.make_object('%s0' % okind, okind, xy, table_top=tt)]
        dest = None
        if kind == 'pick':
            dest = 'tray0'                      # 쟁반이 바구니보다 쉬움
            scene.append(world.make_object(dest, 'tray', TRAY_XY,
                                           table_top=tt))
        tko = KO.get(okind, '물건')
        instr = ('%s 을(를) 집어서 왼쪽 쟁반에 놓아 줘.' % tko
                 if kind == 'pick' else
                 '%s 을(를) 잡아서 들어 올려 봐.' % tko)
        tasks.append({
            'id': 'sweet%02d_%s' % (i, kind),
            'table_top': tt,
            'light': [0.3, 0.0, 1.6],
            'kind': kind,
            'instruction': instr,
            'target': scene[0]['name'],
            'tray': dest,
            'scene': [{'name': o['name'], 'kind': o['kind'],
                       'xy': [round(o['pos'][0], 3), round(o['pos'][1], 3)]}
                      for o in scene],
        })
    return tasks


def feasible(task, quick_steps=25):
    """이 태스크에 '안전한 해가 존재하는가' 를 오라클 서보로 빠르게 확인.

    이 로봇은 물리적으로 못 하는 게 많다(작은 컵 위에서 집기, 정중선 왼쪽,
    좁은 작업영역). 못 푸는 태스크를 배치에 넣으면 모델 호출만 낭비되고
    데이터도 실패 더미가 된다. 통과한 태스크만 채택한다.
    """
    import sim_agent as A

    w = build_world(task)
    saved = dict(A.BEHAVIOR)
    A.BEHAVIOR.update(A.DEFAULTS)      # 기준 파라미터로 판정 (순환 방지)
    try:
        planner = A.OraclePlanner()
        view = planner.perceive(w, task)
        if not view.get('seen'):
            return False
        plan = planner.plan(w, task, view)
        # 실현가능성 검사는 복귀 생략 (수백 번 돌므로 빠르게). 성공은
        # check 시점(임무 완료 순간)에 잰다 - lift 는 제자리에 되돌려
        # 놓으므로 사후 판정이면 무조건 미달로 나온다.
        trace = []
        ok = A.execute(w, plan, retreat=False, trace=trace,
                       check=lambda: success_fn(task)(w))
        # 최종 상태만 깨끗해도 중간에 테이블/다른 물체를 통과했으면
        # 실현 가능한 과제가 아니다. 학습 전에 경로 전체를 문지기로 쓴다.
        path_table = min((x['table'] for x in trace), default=99)
        path_object = min((x['obj'] for x in trace), default=99)
        path_target = min((x.get('tgt', 99) for x in trace), default=99)
        return bool(ok and path_table >= -0.5 and path_object >= -0.5
                    and path_target >= -1.0 and _clean(w, task['target']))
    except Exception:
        return False
    finally:
        A.BEHAVIOR.update(saved)
        w.close()


def generate_feasible(n, seed=0, kinds=('look', 'reach'), max_tries=6,
                      env=None):
    """실현 가능한 태스크만 n 개. 유형은 kinds 를 순환."""
    out = []
    attempt = 0
    # generate 는 look/point/reach/pick 을 순환하므로, 원하는 유형이 고루
    # 나오게 한 번에 4개(한 순환)씩 만들어 거른다. 유형 균형은 라운드로빈.
    want_idx = 0
    while len(out) < n and attempt < n * max_tries:
        batch = generate(6, seed=seed * 1000 + attempt, env=env)
        wanted = kinds[want_idx % len(kinds)]
        for t in batch:
            if t['kind'] != wanted:
                continue
            if feasible(t):
                t['id'] = 'task%02d_%s' % (len(out), t['kind'])
                out.append(t)
                want_idx += 1
                break
        attempt += 1
    return out


def build_world(task, **kw):
    """태스크의 scene 을 실제 World 로 (테이블 높이 포함)."""
    tt = task.get('table_top', world.TABLE_TOP)
    objs = [world.make_object(o['name'], o['kind'], tuple(o['xy']),
                              table_top=tt)
            for o in task['scene']]
    return world.World(objs, table_top=tt, light=task.get('light'), **kw)


def success_fn(task):
    kind = task['kind']
    if kind == 'pick':
        return SUCCESS['pick'](task['target'], task['tray'])
    return SUCCESS[kind](task['target'])


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()

    tasks = generate(args.n, args.seed)
    if args.list or not args.out:
        for t in tasks:
            objs = ', '.join('%s@(%.2f,%.2f)' % (o['name'], *o['xy'])
                             for o in t['scene'])
            print('  [%s] %s' % (t['kind'], t['instruction']))
            print('        대상 %s | 장면: %s' % (t['target'], objs))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'tasks': tasks}, fh, ensure_ascii=False, indent=2)
        print('\n저장: %s (%d개)' % (args.out, len(tasks)))


if __name__ == '__main__':
    main()
