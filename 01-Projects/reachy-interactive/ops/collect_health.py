"""사진 수집이 잘 돌고 있나 한눈에 본다.

이번 여름에 수집이 조용히 멈춘 적이 여러 번 있었다 - 카메라가 걸려 감시
스레드가 통째로 멈췄고, 시험하다 서비스를 멈춰 놓고 되살리지 않았고, 얼굴
검출기가 사람을 못 잡았다. 어느 경우에도 겉으로는 아무 일 없어 보였다.

그래서 '지금 수집이 살아 있나'를 한 번에 확인한다. 문제가 있으면 무엇을 보라고
알려 준다.

사용:
    python3 ops/collect_health.py
    python3 ops/collect_health.py --quiet     # 문제가 있을 때만 출력
"""

import argparse
import subprocess
import sys

SSH = ['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       'pi@localhost']

# 로봇에서 한 번에 확인한다 (왕복을 줄인다).
PROBE = r'''
import os, sqlite3, subprocess, time
def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except Exception:
        return ''
print('service=' + sh('systemctl is-active voice_chat'))
print('restarts=' + (sh('systemctl show voice_chat -p NRestarts --value') or '?'))
print('uptime_s=' + sh("ps -o etimes= -p $(systemctl show voice_chat -p MainPID --value) 2>/dev/null | tr -d ' '"))
st = os.statvfs('/home/pi')
print('free_mb=%d' % (st.f_bavail * st.f_frsize / 1048576))
db = '/home/pi/reachy_logs/persons/persons.db'
if os.path.exists(db):
    c = sqlite3.connect(db)
    print('photos=%d' % c.execute('SELECT COUNT(*) FROM persons').fetchone()[0])
    print('today=%d' % c.execute(
        "SELECT COUNT(*) FROM persons WHERE day=date('now','localtime')").fetchone()[0])
    last = c.execute('SELECT MAX(ts) FROM persons').fetchone()[0] or ''
    print('last=' + last)
    # 경과 시간은 로봇에서 잰다. DGX 는 UTC, 로봇은 KST 라 여기서 계산하지
    # 않으면 9시간이 어긋난다(실제로 -7.4시간으로 나왔다).
    if last:
        import datetime
        age = (datetime.datetime.now()
               - datetime.datetime.strptime(last, '%Y-%m-%d %H:%M:%S'))
        print('age_s=%d' % age.total_seconds())
else:
    print('photos=0'); print('today=0'); print('last=')
log = '/home/pi/reachy_logs/voice_chat.log'
if os.path.exists(log):
    text = open(log, errors='ignore').read()
    for key, needle in (('stalled', '사람 감시가'), ('noframe', '프레임을 못 받고'),
                        ('collect_on', 'People collection ON'),
                        ('ssd_on', '사람 검출(SSD)'), ('tts_edge', 'TTS engine: edge')):
        print('%s=%d' % (key, text.count(needle)))
'''


def probe():
    out = subprocess.run(SSH + ['python3 -'], input=PROBE, capture_output=True,
                         text=True, timeout=60)
    if out.returncode != 0:
        return None, (out.stderr or '').strip()[:200]
    got = {}
    for line in out.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            got[k] = v
    return got, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quiet', action='store_true', help='문제가 있을 때만 출력')
    ap.add_argument('--stale-hours', type=float, default=6.0,
                    help='마지막 촬영이 이보다 오래되면 알린다 (기본 6시간)')
    args = ap.parse_args()

    got, err = probe()
    if got is None:
        print('로봇에 연결하지 못했습니다: %s' % err)
        print('  터널 확인:  systemctl status pi_tunnel')
        return 1

    problems = []
    if got.get('service') != 'active':
        problems.append('로봇 서비스가 꺼져 있습니다 (%s). '
                        '켜기: sudo systemctl start voice_chat'
                        % got.get('service'))
    if not int(got.get('collect_on', 0) or 0):
        problems.append('사진 수집이 켜지지 않았습니다 '
                        '(--collect-people 인자 확인)')
    if not int(got.get('ssd_on', 0) or 0):
        problems.append('사람 검출(SSD)이 연결되지 않았습니다 '
                        '(mnssd 모델 확인: ops/install_object_vision.sh)')
    if int(got.get('stalled', 0) or 0):
        problems.append('사람 감시가 멈춘 적이 있습니다 (로그의 "사람 감시가")')
    if int(got.get('noframe', 0) or 0):
        problems.append('카메라 프레임을 못 받은 적이 있습니다')
    free = int(got.get('free_mb', 0) or 0)
    if free < 1000:
        problems.append('SD 여유가 %dMB 뿐입니다 - 수집이 멈춥니다 '
                        '(sync_persons.py --purge-remote 로 비우세요)' % free)

    last = got.get('last') or ''
    stale = None
    if got.get('age_s'):
        try:
            stale = float(got['age_s']) / 3600.0
        except ValueError:
            stale = None

    if not args.quiet or problems:
        up = int(got.get('uptime_s') or 0)
        print('로봇: %s · 가동 %d시간 %d분 · 재시작 %s회'
              % (got.get('service'), up // 3600, (up % 3600) // 60,
                 got.get('restarts')))
        print('사진: 모두 %s장 · 오늘 %s장 · SD 여유 %dMB'
              % (got.get('photos'), got.get('today'), free))
        if last:
            print('마지막 촬영: %s%s'
                  % (last, ' (%.1f시간 전)' % stale if stale is not None else ''))
        else:
            print('마지막 촬영: 아직 없음')

    if stale is not None and stale > args.stale_hours:
        problems.append('마지막 촬영이 %.1f시간 전입니다 - 복도가 한산한 것일 수도, '
                        '수집이 멎은 것일 수도 있습니다' % stale)

    if problems:
        print('\n확인이 필요합니다:')
        for p in problems:
            print('  · %s' % p)
        return 1
    if not args.quiet:
        print('\n수집 정상.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
