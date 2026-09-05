"""시뮬레이션에 먹일 상황을 만든다 - 사람, 말, 장면.

동작을 자동으로 다듬으려면 먼저 '무엇을 시킬지' 가 있어야 한다. 손으로 쓴
프리셋 9개로는 금방 바닥난다. 그래서 로컬 모델(ollama, 무료)에게 사람과 발화와
장면을 만들게 한다. opus 는 아껴서 동작 설계와 평가에만 쓴다.

작은 모델에 JSON 을 시키면 자주 깨진다. 그래서 한 줄에 하나씩 뱉게 하고 줄
단위로 읽는다 - 형식이 틀려도 대부분의 줄은 건진다.

사용:
    python3 sim/sim_scenarios.py --personas 5
    python3 sim/sim_scenarios.py --commands 20 --out config/sim_commands.json
"""

import json
import os
import re
import sys
import urllib.request


OLLAMA = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
MODEL = os.environ.get('SIM_MODEL', 'exaone3.5:7.8b')

# 로봇이 실제로 할 수 있는 것. 여기서 벗어난 요청을 만들면 시뮬이 통째로
# 낭비된다 - 걷기, 물건 던지기, 손가락 하나만 펴기 같은 것.
CAN_DO = """이 로봇은 두 팔(어깨·팔꿈치·전완·손목), 오른손 그리퍼, 머리 위 안테나
두 개만 움직일 수 있다. 다리도 손가락도 없고, 제자리에서 상반신만 움직인다.
물건을 집는 것은 오른손만 가능하고, 왼손은 받침 쟁반이다."""


def ask(prompt, num_predict=400, temperature=0.9):
    """로컬 모델에 한 번 묻는다. 실패하면 빈 문자열."""
    body = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        # 페르소나가 잘리지 않게 넉넉히. 기본 2048 은 앞에서부터 자른다.
        'options': {'num_predict': num_predict, 'num_ctx': 8192,
                    'temperature': temperature},
    }).encode('utf-8')
    req = urllib.request.Request('http://{}/api/chat'.format(OLLAMA), data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode('utf-8'))['message']['content']
    except Exception as e:
        print('  로컬 모델 호출 실패: %s' % e, file=sys.stderr)
        return ''


def _lines(text, want=None):
    """줄 단위로 읽되 번호·따옴표·군더더기를 떼어 낸다."""
    out = []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        line = re.sub(r'^\s*[-*\d]+[.)\]]?\s*', '', line).strip()
        line = line.strip('"\'`').strip()
        # 모델이 종종 서두를 붙인다 ("다음은 ... 입니다:")
        if line.endswith(':') or line.endswith('：'):
            continue
        if len(line) < 3 or len(line) > 120:
            continue
        out.append(line)
        if want and len(out) >= want:
            break
    return out


def personas(n=5):
    """복도에서 마주칠 만한 사람들. [{'who','style'}]"""
    text = ask(
        '한국로봇융합연구원 사무실 복도에 로봇이 서 있다. 그 앞을 지나가며 '
        '로봇에게 말을 걸 만한 사람을 %d명 만들어라.\n'
        '한 줄에 한 명씩, "누구인가 | 말투" 형식으로만 써라. 설명하지 마라.\n'
        '예시: 견학 온 초등학생 | 반말로 신나게 이것저것 시킨다\n' % n)
    out = []
    for line in _lines(text, want=n * 2):
        if '|' not in line:
            continue
        who, _, style = line.partition('|')
        who, style = who.strip(), style.strip()
        if who and style:
            out.append({'who': who, 'style': style})
        if len(out) >= n:
            break
    return out


def commands(persona=None, n=8):
    """이 사람이 로봇에게 시킬 만한 '동작 명령' 들."""
    who = persona['who'] if persona else '복도를 지나가는 사람'
    style = persona['style'] if persona else '평범한 말투'
    text = ask(
        '%s\n\n"%s"(%s)가 이 로봇에게 몸을 움직여 달라고 시키는 말을 '
        '%d개 만들어라.\n'
        '- 한 줄에 하나씩, 그 사람이 실제로 할 법한 말 그대로만 써라\n'
        '- 로봇이 할 수 있는 범위 안에서만 (걷기·달리기·점프는 안 된다)\n'
        '- 번호나 설명을 붙이지 마라\n'
        % (CAN_DO, who, style, n), num_predict=500)
    return _lines(text, want=n)


def situations(n=5):
    """책상 위 상황. 물건이 있는 장면을 말로 준다."""
    text = ask(
        '%s\n\n로봇 앞 책상에 물건이 놓인 장면을 %d개 만들어라.\n'
        '한 줄에 하나씩 "장면 | 로봇에게 시킬 말" 형식으로만 써라.\n'
        '예시: 오른쪽 앞에 종이컵이 하나 있다 | 저 컵 좀 집어서 쟁반에 놓아줘\n'
        % (CAN_DO, n), num_predict=500)
    out = []
    for line in _lines(text, want=n * 2):
        if '|' not in line:
            continue
        scene, _, say = line.partition('|')
        if scene.strip() and say.strip():
            out.append({'scene': scene.strip(), 'say': say.strip()})
        if len(out) >= n:
            break
    return out


def build(n_personas=4, per_persona=6, n_situations=4):
    """페르소나 + 발화 + 장면을 한 번에. 학습 루프가 이걸 먹는다."""
    people = personas(n_personas)
    if not people:
        people = [{'who': '복도를 지나가는 직원', 'style': '존댓말로 짧게'}]
        print('  페르소나 생성 실패 - 기본값 하나로 진행합니다', file=sys.stderr)

    tasks = []
    for p in people:
        for say in commands(p, per_persona):
            tasks.append({'who': p['who'], 'style': p['style'],
                          'say': say, 'scene': None})
    for s in situations(n_situations):
        tasks.append({'who': '물건을 시키는 사람', 'style': '평범한 말투',
                      'say': s['say'], 'scene': s['scene']})

    # 같은 말이 여러 번 나오면 시뮬만 낭비다.
    seen, unique = set(), []
    for t in tasks:
        key = re.sub(r'\s+', '', t['say'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return {'personas': people, 'tasks': unique}


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--personas', type=int, default=4)
    ap.add_argument('--per-persona', type=int, default=6)
    ap.add_argument('--situations', type=int, default=4)
    ap.add_argument('--out', help='결과를 이 json 에 저장')
    args = ap.parse_args()

    data = build(args.personas, args.per_persona, args.situations)
    print('페르소나 %d명, 시킬 말 %d개' % (len(data['personas']),
                                          len(data['tasks'])))
    for p in data['personas']:
        print('  · %s — %s' % (p['who'], p['style']))
    print()
    for t in data['tasks'][:12]:
        print('  "%s"%s' % (t['say'],
                            '   [장면: %s]' % t['scene'] if t['scene'] else ''))
    if len(data['tasks']) > 12:
        print('  ... 그리고 %d개 더' % (len(data['tasks']) - 12))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print('\n저장: %s' % args.out)


if __name__ == '__main__':
    main()
