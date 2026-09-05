"""Drive the virtual robot from the keyboard - no mic, no TTS, no hardware.

Runs the exact same motion pipeline (opus generation -> safety validation ->
50 Hz executor with hand-follow) against the io='ws' virtual robot, so you
can watch everything in sim_viewer.html before it ever touches the real arms.

Usage (on the Pi; stop any session using port 6171 first):
    python3 sim_play.py --list                        # show presets
    python3 sim_play.py --preset wave                 # play a preset
    python3 sim_play.py --say "왼팔 천천히 들어봐" \
        --url http://127.0.0.1:8080 --token ...       # opus-generated motion
    python3 sim_play.py --repl --url ... --token ...  # type commands in a loop
"""

import argparse
import logging
import time


logger = logging.getLogger(__name__)


SIM_PORT = 6172   # 실물 미러(6171)와 동시 운용을 위한 시뮬 전용 포트


def build_robot(port=SIM_PORT):
    # ws.py 의 WsServer 는 6171 고정이라, 시뮬 전용 포트로 패치해서 띄운다.
    from reachy.io import ws as wsio

    original = wsio.WsServer

    class PortedServer(original):
        def __init__(self, host='0.0.0.0', _port=None):
            original.__init__(self, host, port)

    wsio.WsServer = PortedServer
    try:
        from base_pose import connect
        return connect(io='ws', with_head=True)
    finally:
        wsio.WsServer = original


def play(executor, validate, moves, label, unsafe_preview=False):
    from motion_exec import ValidationError

    seed = {j: 0.0 for j in executor.available_joints()}
    try:
        segments = validate(moves, seed, executor.available_joints())
    except ValidationError as e:
        print('⚠ 안전층 거부 ({}): {}'.format(label, e))
        if not unsafe_preview:
            print('  (실물이었다면 실행되지 않음. 시뮬로 보려면 --unsafe-preview)')
            return
        print('  → 시뮬 전용 미리보기: 충돌검사만 끄고 재생 (각도/속도 제한은 유지)')
        try:
            segments = validate(moves, seed, executor.available_joints(),
                                check_collision=False)
        except ValidationError as e2:
            print('  형식 자체가 불량이라 미리보기도 불가:', e2)
            return

    print('실행: {} ({} 세그먼트)'.format(label, len(segments)))
    ok, reason = executor.execute(segments)
    print('결과:', 'OK' if ok else reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', help='play this preset on the virtual robot')
    parser.add_argument('--say', help='generate a motion from this sentence (needs broker)')
    parser.add_argument('--repl', action='store_true', help='interactive loop')
    parser.add_argument('--candidates', metavar='FILE', nargs='?',
                        const='motion_candidates.json',
                        help='browse generated candidates, adopt good ones as presets')
    parser.add_argument('--list', action='store_true', help='list presets and exit')
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--token')
    parser.add_argument('--unsafe-preview', action='store_true',
                        help='play safety-REJECTED motions in the sim anyway '
                             '(collision check off; never affects the real robot)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from motion_presets import PRESETS, validate_all
    validate_all()

    if args.list:
        for name, info in sorted(PRESETS.items()):
            print('{:15s} {}'.format(name, info['description']))
        return

    if not (args.preset or args.say or args.repl or args.candidates):
        parser.error('choose --preset, --say, --repl, --candidates or --list')

    print('가상 로봇 기동 중... 뷰어에서 ws://<Pi IP>:%d 연결 (실물 미러와 동시 운용 가능)' % SIM_PORT)
    reachy = build_robot()

    from motion_exec import MotionExecutor, validate
    executor = MotionExecutor(reachy)

    client = None
    if args.say or args.repl:
        from llm_client import BrokerClient
        client = BrokerClient(args.url, token=args.token, session='sim-play')

    def browse_candidates(path):
        import json as _json
        cands = _json.load(open(path, encoding='utf-8'))
        print('후보 %d개. 각 동작 재생 후 채택 여부를 묻습니다.' % len(cands))
        adopted = {}
        for i, cd in enumerate(cands):
            print('\n[%d/%d] %s' % (i+1, len(cands), cd['idea']))
            print('   say:', cd.get('say'))
            # 재생하기 전에 시연 안전 등급을 먼저 보여 준다. 실행기의 검증기는
            # 팔을 굵기 없는 중심선으로만 보므로, 통과했다고 시연에서 안전한
            # 것은 아니다 - 사람 앞에서 돌리기 전에 알고 있어야 한다.
            try:
                import sim_safety
                print('   시연 안전:', sim_safety.describe(
                    sim_safety.audit(cd['moves'], hz=40)))
            except Exception:
                pass
            play(executor, validate, cd['moves'], 'cand%d' % i)
            ans = input('   채택? 프리셋 이름 입력(엔터=건너뜀, r=다시, q=종료): ').strip()
            while ans == 'r':
                play(executor, validate, cd['moves'], 'cand%d' % i)
                ans = input('   채택? 이름(엔터=skip, r=다시, q=종료): ').strip()
            if ans == 'q':
                break
            if ans:
                adopted[ans] = {'description': cd['idea'], 'moves': cd['moves']}
                print('   채택:', ans)
        if adopted:
            out = 'adopted_presets.py'
            with open(out, 'w', encoding='utf-8') as f:
                f.write('# sim_play --candidates 로 채택한 동작. motion_presets.PRESETS 에 병합하세요.\n')
                f.write('ADOPTED = ' + repr(adopted) + '\n')
            print('\n%d개 채택 -> %s (motion_presets 에 병합 후 재배포)' % (len(adopted), out))

    def handle(text):
        if text in PRESETS:
            play(executor, validate, PRESETS[text]['moves'], 'preset:' + text,
                 unsafe_preview=args.unsafe_preview)
            return
        if client is None:
            print('(브로커 미설정 - 프리셋 이름만 가능)')
            return
        motion = client.ask_motion(text)
        if motion is None:
            print('동작 생성 실패 (브로커 확인)')
            return
        print('리치:', motion.get('say'))
        preset = motion.get('preset')
        if preset and preset in PRESETS:
            play(executor, validate, PRESETS[preset]['moves'], 'preset:' + preset,
                 unsafe_preview=args.unsafe_preview)
        elif motion.get('moves'):
            play(executor, validate, motion['moves'],
                 '생성 %d키프레임' % len(motion['moves']),
                 unsafe_preview=args.unsafe_preview)
        else:
            print('(동작 아님)')

    try:
        time.sleep(1.0)   # let the viewer connect and see the start pose

        if args.candidates:
            browse_candidates(args.candidates)
        if args.preset:
            handle(args.preset)
        if args.say:
            handle(args.say)

        while args.repl:
            try:
                line = input('명령(프리셋명 또는 문장, 빈줄 종료)> ').strip()
            except EOFError:
                break
            if not line:
                break
            handle(line)

        time.sleep(1.0)
    finally:
        executor.shutdown()
        reachy.close()


if __name__ == '__main__':
    main()
