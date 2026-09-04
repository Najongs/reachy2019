"""Pi 에 쌓인 사람 사진을 DGX 로 모아 중앙 데이터셋을 만든다.

로봇의 SD 카드는 작고(8GB대) 보관기간도 30일로 잘라 두었지만, 학습용 데이터는
오래 그리고 많이 모을수록 좋다. 그래서 사진은 Pi 에서 찍고, 이 스크립트가
주기적으로 DGX 로 옮겨 중앙 DB에 합친다.

중앙 저장소 (기본): 04-Archives/person-dataset/persons/
    persons.db                 합쳐진 메타데이터
    <robot>/YYYY-MM-DD/*.jpg   원본 사진 (로봇별로 분리)

같은 사진을 두 번 넣지 않는다 (로봇 이름 + Pi 쪽 행 id 를 유일키로 둔다).
그래서 몇 번을 돌려도 안전하고, 중간에 끊겨도 다시 돌리면 이어서 받는다.

사용:
    python3 ops/sync_persons.py                # 받아오고 DB 합치기
    python3 ops/sync_persons.py --purge-remote # 옮긴 뒤 Pi 쪽 사진 삭제(SD 확보)
    python3 ops/sync_persons.py --stats        # 중앙 DB 현황만 보기
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

# 저장소 안(04-Archives/person-dataset)에 둔다. 대화 로그와 같은 성격이라
# 같은 자리에 모이는 편이 찾기 쉽다. .gitignore 에 들어 있어 커밋에는 안 섞인다.
DEFAULT_LOCAL = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '04-Archives', 'person-dataset', 'persons'))
REMOTE_DIR = 'reachy_logs/persons'
# ssh 는 포트가 -p, scp 는 -P 다. 섞어 쓰면 scp 가 -p 를 "시각 보존"으로 읽어
# 엉뚱하게 실패한다.
SSH_ARGS = ['-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
SCP_ARGS = ['-P', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
REMOTE = 'pi@localhost'


def ensure_db(path):
    """중앙 DB 를 열고 (없으면 만들고) 스키마를 보장한다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            robot       TEXT NOT NULL,      -- 어느 로봇에서 왔는지
            src_id      INTEGER NOT NULL,   -- Pi 쪽 원래 행 id
            ts          TEXT, day TEXT,
            visit_id    TEXT, seq INTEGER,
            image       TEXT, face_image TEXT,
            face_x INTEGER, face_y INTEGER, face_w INTEGER, face_h INTEGER,
            frame_w INTEGER, frame_h INTEGER,
            sharpness   REAL,
            interacted  INTEGER DEFAULT 0,
            -- 사람 검출기가 알려 준 사람 위치 (나중에 사람만 잘라 쓰기 위해)
            person_x    INTEGER, person_y INTEGER,
            person_w    INTEGER, person_h INTEGER,
            person_conf REAL,
            -- 이 상자를 누가 찍었나. 'pi-ssd' 는 로봇이 실시간으로 쓴 가벼운
            -- 검출기, 'dgx-frcnn' 은 여기서 제대로 된 모델로 다시 잡은 것.
            -- 로봇은 '사람이 있다'만 판단하면 되고, 인식 DB 는 여기서 만든다.
            box_source  TEXT,
            UNIQUE(robot, src_id)           -- 같은 사진을 두 번 넣지 않는다
        )''')
    for col, typ in (('person_x', 'INTEGER'), ('person_y', 'INTEGER'),
                     ('person_w', 'INTEGER'), ('person_h', 'INTEGER'),
                     ('person_conf', 'REAL'), ('box_source', 'TEXT')):
        try:
            conn.execute('ALTER TABLE persons ADD COLUMN %s %s' % (col, typ))
        except Exception:
            pass                # 이미 있는 칼럼
    conn.execute('CREATE INDEX IF NOT EXISTS idx_day ON persons(day)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_visit ON persons(robot, visit_id)')
    conn.commit()
    return conn


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pull(local_root, robot):
    """Pi 의 사진과 DB 를 받아온다. 받은 임시 DB 경로를 돌려준다."""
    img_dir = os.path.join(local_root, robot)
    os.makedirs(img_dir, exist_ok=True)

    ssh_cmd = 'ssh ' + ' '.join(SSH_ARGS)
    # 사진: 이미 있는 건 건너뛴다 (증분)
    r = run(['rsync', '-az', '--partial', '-e', ssh_cmd,
             '--include=*/', '--include=*.jpg', '--exclude=*',
             '%s:%s/' % (REMOTE, REMOTE_DIR), img_dir + '/'])
    if r.returncode != 0:
        print('  사진 동기화 실패:', (r.stderr or '').strip()[:200])
        return None

    # DB: 통째로 임시 파일에 받아 온다 (원본은 Pi 가 계속 쓰므로 건드리지 않는다)
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    r = run(['scp'] + SCP_ARGS + ['-q',
            '%s:%s/persons.db' % (REMOTE, REMOTE_DIR), tmp.name])
    if r.returncode != 0:
        print('  DB 가져오기 실패:', (r.stderr or '').strip()[:200])
        os.unlink(tmp.name)
        return None
    return tmp.name


def merge(conn, remote_db, robot, img_dir=None):
    """Pi DB 의 행을 중앙 DB 에 합친다. (새로 들어간 수, 사진 없어 건너뛴 수)."""
    src = sqlite3.connect(remote_db)
    # Pi 쪽 DB 가 아직 예전 스키마일 수 있으니 있는 칼럼만 고른다.
    have = {r[1] for r in src.execute('PRAGMA table_info(persons)')}
    extra = [c for c in ('person_x', 'person_y', 'person_w', 'person_h',
                         'person_conf') if c in have]
    cols = ['id', 'ts', 'day', 'visit_id', 'seq', 'image', 'face_image',
            'face_x', 'face_y', 'face_w', 'face_h',
            'frame_w', 'frame_h', 'sharpness', 'interacted'] + extra
    rows = src.execute('SELECT %s FROM persons' % ', '.join(cols)).fetchall()
    n_base = 15
    target = ['robot', 'src_id', 'ts', 'day', 'visit_id', 'seq', 'image',
              'face_image', 'face_x', 'face_y', 'face_w', 'face_h',
              'frame_w', 'frame_h', 'sharpness', 'interacted'] + extra
    sql = ('INSERT OR IGNORE INTO persons (%s) VALUES (%s)'
           % (', '.join(target), ', '.join('?' * len(target))))
    added = skipped = 0
    for r in rows:
        # 사진이 아직 안 온 행은 넣지 않는다. 다음 번에 사진과 함께 들어온다.
        # (파일 없는 행이 쌓이면 학습용으로 뽑을 때 빈 손이 된다.)
        if r[5] and img_dir and not os.path.exists(os.path.join(img_dir, r[5])):
            skipped += 1
            continue
        try:
            conn.execute(sql, (robot,) + tuple(r[:n_base]) + tuple(r[n_base:]))
            if conn.total_changes:
                added += 1
                conn.execute(
                    '''UPDATE persons SET box_source='pi-ssd'
                       WHERE robot=? AND src_id=? AND box_source IS NULL''',
                    (robot, r[0]))
        except Exception:
            pass
    # interacted 는 나중에 갱신될 수 있으므로 항상 최신값으로 맞춰 준다
    for r in rows:
        conn.execute('UPDATE persons SET interacted=? WHERE robot=? AND src_id=?',
                     (r[14], robot, r[0]))
    conn.commit()
    src.close()
    return added, skipped


def purge_remote(before_day=None):
    """옮긴 사진을 Pi 에서 지워 SD 를 비운다 (DB 행도 함께)."""
    script = (
        "python3 - <<'EOF'\n"
        "import os, sqlite3\n"
        "root=os.path.expanduser('~/reachy_logs/persons')\n"
        "db=os.path.join(root,'persons.db')\n"
        "c=sqlite3.connect(db)\n"
        "rows=c.execute('SELECT id,image,face_image FROM persons').fetchall()\n"
        "n=0\n"
        "for rid,img,face in rows:\n"
        "    for rel in (img,face):\n"
        "        if rel:\n"
        "            try: os.remove(os.path.join(root,rel))\n"
        "            except OSError: pass\n"
        "    n+=1\n"
        "c.execute('DELETE FROM persons'); c.commit()\n"
        "for name in os.listdir(root):\n"
        "    d=os.path.join(root,name)\n"
        "    if os.path.isdir(d) and not os.listdir(d): os.rmdir(d)\n"
        "print('purged', n)\n"
        "EOF")
    r = run(['ssh'] + SSH_ARGS + [REMOTE, script])
    return (r.stdout or '').strip()


def stats(conn, local_root):
    q = lambda s: conn.execute(s).fetchone()[0]
    total = q('SELECT COUNT(*) FROM persons')
    if not total:
        print('중앙 데이터셋: 아직 사진 없음')
        return
    print('중앙 데이터셋 (%s)' % local_root)
    print('  사진 %d장 · 방문 %d회 · %d일치'
          % (total, q('SELECT COUNT(DISTINCT robot||visit_id) FROM persons'),
             q('SELECT COUNT(DISTINCT day) FROM persons')))
    print('  얼굴 크롭 %d장 · 대화로 이어진 방문 %d회'
          % (q('SELECT COUNT(*) FROM persons WHERE face_image IS NOT NULL'),
             q('SELECT COUNT(DISTINCT robot||visit_id) FROM persons WHERE interacted=1')))
    sharp = conn.execute(
        'SELECT COUNT(*) FROM persons WHERE sharpness > 200').fetchone()[0]
    print('  선명도 200 이상: %d장 (학습에 쓸 만한 것)' % sharp)
    try:
        pb = q('SELECT COUNT(*) FROM persons WHERE person_w IS NOT NULL')
        good = q("SELECT COUNT(*) FROM persons WHERE box_source='dgx-frcnn'")
        todo = q("SELECT COUNT(*) FROM persons "
                 "WHERE box_source IS NULL OR box_source<>'dgx-frcnn'")
        print('  사람 위치가 기록된 것: %d장 (여기서 제대로 잡은 것 %d장)' % (pb, good))
        if todo:
            print('  아직 여기서 안 잡은 것: %d장 (다음 업데이트에서 처리)' % todo)
    except Exception:
        pass
    for day, n in conn.execute(
            'SELECT day, COUNT(*) FROM persons GROUP BY day ORDER BY day DESC LIMIT 7'):
        print('    %s  %d장' % (day, n))
    size = 0
    for dirpath, _, files in os.walk(local_root):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    print('  디스크 사용: %.1f MB' % (size / 1024 / 1024))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--local', default=DEFAULT_LOCAL, help='중앙 저장소 위치')
    ap.add_argument('--robot', default='reachy', help='로봇 이름 (여러 대일 때 구분)')
    ap.add_argument('--purge-remote', action='store_true',
                    help='옮긴 뒤 Pi 쪽 사진 삭제 (SD 확보)')
    ap.add_argument('--stats', action='store_true', help='현황만 출력')
    args = ap.parse_args()

    local_root = os.path.expanduser(args.local)
    conn = ensure_db(os.path.join(local_root, 'persons.db'))

    if args.stats:
        stats(conn, local_root)
        return 0

    print('── Pi 에서 사람 사진 가져오기')
    remote_db = pull(local_root, args.robot)
    if remote_db is None:
        print('  가져오지 못했습니다 (터널 확인: systemctl status pi_tunnel)')
        stats(conn, local_root)
        return 1

    added, skipped = merge(conn, remote_db, args.robot,
                           img_dir=os.path.join(local_root, args.robot))
    os.unlink(remote_db)
    print('  새로 추가된 사진: %d장' % added)
    if skipped:
        print('  사진이 아직 안 온 행 %d건은 건너뜀 (다음 번에 함께 들어옵니다)'
              % skipped)

    if args.purge_remote:
        print('── Pi 쪽 정리')
        print(' ', purge_remote())

    print()
    stats(conn, local_root)
    conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
