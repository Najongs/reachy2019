"""복도를 지나가는 사람을 찍어 모으는 데이터셋.

얼굴이 감지될 때마다 사진(전체 프레임 + 얼굴 크롭)을 저장하고, 언제·얼마나 크게
·얼마나 선명하게 찍혔는지를 SQLite 에 기록한다. 나중에 조건을 걸어 뽑아 쓰기
좋게 하려는 것 (예: "선명도 200 이상이고 얼굴이 80px 이상인 것만").

한 사람이 지나갈 때 여러 장을 남긴다(각도·거리가 달라 데이터로서 가치가 있다).
다만 복도에 하루 종일 두는 장비이므로 다음을 기본으로 둔다:
  - 방문 1회당 장수 제한, 촬영 간격 제한
  - 하루 총량 제한
  - RETENTION_DAYS 지난 사진 자동 삭제 (DB 행도 함께)
  - SD 여유가 MIN_FREE_MB 아래면 촬영 중단

저장 위치: ~/reachy_logs/persons/
    persons.db              메타데이터 (sqlite)
    YYYY-MM-DD/<id>.jpg     전체 프레임
    YYYY-MM-DD/<id>_face.jpg 얼굴 크롭
"""

import logging
import os
import sqlite3
import time

logger = logging.getLogger('reachy.persondb')

DEFAULT_ROOT = os.path.expanduser('~/reachy_logs/persons')

RETENTION_DAYS = 30      # 이 기간 지난 사진·행은 자동 삭제
MIN_FREE_MB = 700        # SD 여유가 이보다 적으면 촬영 중단
PER_VISIT_MAX = 6        # 한 사람이 머무는 동안 최대 장수
VISIT_INTERVAL = 4.0     # 같은 방문 중 촬영 간격(초)
DAILY_MAX = 800          # 하루 총 장수
FACELESS_INTERVAL = 12.0 # 얼굴이 안 보이는(움직임만) 사진은 더 드물게
AUTO_VISIT_GAP = 90.0    # 이만큼 조용하면 다음 사진은 새 방문으로 친다
MIN_SHARPNESS = 60.0     # 이보다 흐리면 버린다 (머리가 도는 중에 찍힌 사진)


class PersonDB(object):
    """사람 사진 + 메타데이터 저장소."""

    def __init__(self, root=DEFAULT_ROOT, retention_days=RETENTION_DAYS,
                 per_visit_max=PER_VISIT_MAX, visit_interval=VISIT_INTERVAL,
                 daily_max=DAILY_MAX, min_free_mb=MIN_FREE_MB):
        self.root = os.path.expanduser(root)
        self.retention_days = retention_days
        self.per_visit_max = per_visit_max
        self.visit_interval = visit_interval
        self.daily_max = daily_max
        self.min_free_mb = min_free_mb

        os.makedirs(self.root, exist_ok=True)
        self.db_path = os.path.join(self.root, 'persons.db')
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,       -- 'YYYY-MM-DD HH:MM:SS'
                day         TEXT NOT NULL,       -- 'YYYY-MM-DD' (정리용)
                visit_id    TEXT,                -- 같은 방문끼리 묶는 키
                seq         INTEGER,             -- 그 방문에서 몇 번째 장
                image       TEXT,                -- 전체 프레임 경로 (root 기준 상대)
                face_image  TEXT,                -- 얼굴 크롭 경로
                face_x      INTEGER, face_y INTEGER,
                face_w      INTEGER, face_h INTEGER,
                frame_w     INTEGER, frame_h INTEGER,
                sharpness   REAL,                -- 선명도 (낮으면 학습에 부적합)
                interacted  INTEGER DEFAULT 0    -- 이 방문에서 실제로 대화했나
            )''')
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_persons_day ON persons(day)')
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_persons_visit ON persons(visit_id)')
        self._conn.commit()

        self._visit_id = None
        self._visit_count = 0
        self._last_shot = 0.0
        self._visit_auto = False   # 이 방문을 스스로 열었는지(복도가 연 것과 구분)
        self._auto_seq = 0
        self._day = time.strftime('%Y-%m-%d')
        self._day_count = self._count_today()
        self.prune()

    # -- 상태 -----------------------------------------------------------------

    def _count_today(self):
        try:
            cur = self._conn.execute('SELECT COUNT(*) FROM persons WHERE day=?',
                                     (time.strftime('%Y-%m-%d'),))
            return cur.fetchone()[0]
        except Exception:
            return 0

    def _free_mb(self):
        try:
            st = os.statvfs(self.root)
            return st.f_bavail * st.f_frsize / (1024 * 1024)
        except Exception:
            return 1e9

    def start_visit(self, visit_id):
        """새 방문 시작 (복도 그리터의 'appeared' 시점)."""
        self._visit_id = visit_id
        self._visit_count = 0
        self._last_shot = 0.0
        self._visit_auto = False

    def end_visit(self, interacted=False):
        """방문 종료. 대화가 있었으면 그 방문의 행에 표시해 둔다."""
        if interacted and self._visit_id:
            try:
                self._conn.execute(
                    'UPDATE persons SET interacted=1 WHERE visit_id=?',
                    (self._visit_id,))
                self._conn.commit()
            except Exception:
                logger.debug('interacted update failed', exc_info=True)
        self._visit_id = None
        self._visit_count = 0        # 다음 방문을 위해 비워 둔다
        self._visit_auto = False

    def _autostart_visit(self):
        """방문을 열어 주는 쪽(복도 로직)이 없을 때 스스로 연다.

        얼굴 없이 움직임만으로 찍히는 사진은 복도 인사 로직을 거치지 않는다.
        그대로 두면 visit_id 가 비고, 장수 카운터가 리셋되지 않아 6장 뒤로
        수집이 영영 멈춘다.
        """
        # 복도 로직이 연 방문은 그쪽이 닫을 때까지 건드리지 않는다. 스스로 연
        # 방문만, 한참 조용했으면 다음 사람으로 보고 새로 연다.
        idle = time.time() - self._last_shot
        if self._visit_id is None or (self._visit_auto and idle > AUTO_VISIT_GAP):
            self._auto_seq += 1
            self._visit_id = time.strftime('auto-%Y%m%d-%H%M%S-') + str(self._auto_seq)
            self._visit_count = 0
            self._visit_auto = True

    def should_capture(self, faceless=False):
        """지금 한 장 더 찍어도 되는지 (간격·장수·용량 제한)."""
        now = time.strftime('%Y-%m-%d')
        if now != self._day:                 # 날짜가 바뀌면 카운터 초기화
            self._day = now
            self._day_count = 0
            self.prune()
        if self._day_count >= self.daily_max:
            return False
        if self._visit_count >= self.per_visit_max:
            return False
        gap = FACELESS_INTERVAL if faceless else self.visit_interval
        if time.time() - self._last_shot < gap:
            return False
        if self._free_mb() < self.min_free_mb:
            logger.warning('SD 여유 부족 - 사람 촬영 중단')
            return False
        return True

    # -- 저장 -----------------------------------------------------------------

    def add(self, frame, face=None, sharpness=None):
        """프레임 한 장을 데이터셋에 넣는다. 저장했으면 행 id, 아니면 None.

        Args:
            frame: BGR ndarray (전체 프레임)
            face: (x, y, w, h) 얼굴 위치 - 전체 프레임 좌표
            sharpness: 선명도 지표 (없으면 계산)
        """
        if frame is None:
            return None
        self._autostart_visit()
        if not self.should_capture(faceless=(face is None)):
            return None

        import cv2

        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        day = time.strftime('%Y-%m-%d')
        day_dir = os.path.join(self.root, day)
        os.makedirs(day_dir, exist_ok=True)

        if sharpness is None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                sharpness = None

        # 흔들려서 흐린 사진은 분류 학습에 쓸 수 없다. 저장 자리와 SD 만 먹으므로
        # 여기서 버린다. 버린 것은 촬영 간격에도 반영하지 않아, 곧바로 다음
        # (선명한) 프레임을 다시 노릴 수 있다.
        if sharpness is not None and sharpness < MIN_SHARPNESS:
            logger.debug('흐린 사진 버림 (선명도 %.0f)', sharpness)
            return None

        stamp = '%s_%03d' % (time.strftime('%H%M%S'), self._visit_count)
        rel = os.path.join(day, stamp + '.jpg')
        cv2.imwrite(os.path.join(self.root, rel), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        rel_face = None
        fx = fy = fw = fh = None
        if face is not None:
            fx, fy, fw, fh = [int(v) for v in face]
            # 얼굴 주변을 조금 넉넉히 잘라 둔다 (나중에 정렬·재검출 여지).
            pad = int(0.35 * max(fw, fh))
            x0, y0 = max(0, fx - pad), max(0, fy - pad)
            x1 = min(frame.shape[1], fx + fw + pad)
            y1 = min(frame.shape[0], fy + fh + pad)
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                rel_face = os.path.join(day, stamp + '_face.jpg')
                cv2.imwrite(os.path.join(self.root, rel_face), crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        h, w = frame.shape[:2]
        cur = self._conn.execute(
            '''INSERT INTO persons
               (ts, day, visit_id, seq, image, face_image,
                face_x, face_y, face_w, face_h, frame_w, frame_h, sharpness)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (ts, day, self._visit_id, self._visit_count, rel, rel_face,
             fx, fy, fw, fh, w, h, sharpness))
        self._conn.commit()

        self._visit_count += 1
        self._day_count += 1
        self._last_shot = time.time()
        logger.info('사람 사진 저장 %s (방문 %s, %d/%d장)',
                    rel, self._visit_id, self._visit_count, self.per_visit_max)
        return cur.lastrowid

    # -- 관리 -----------------------------------------------------------------

    def prune(self):
        """보관 기간이 지난 사진과 행을 지운다."""
        if not self.retention_days:
            return 0
        cutoff = time.strftime(
            '%Y-%m-%d', time.localtime(time.time() - self.retention_days * 86400))
        removed = 0
        try:
            rows = self._conn.execute(
                'SELECT id, image, face_image FROM persons WHERE day < ?',
                (cutoff,)).fetchall()
            for rid, img, face in rows:
                for rel in (img, face):
                    if rel:
                        try:
                            os.remove(os.path.join(self.root, rel))
                        except OSError:
                            pass
                removed += 1
            self._conn.execute('DELETE FROM persons WHERE day < ?', (cutoff,))
            self._conn.commit()
            # 빈 날짜 폴더 정리
            for name in os.listdir(self.root):
                d = os.path.join(self.root, name)
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
        except Exception:
            logger.debug('prune failed', exc_info=True)
        if removed:
            logger.info('보관기간(%d일) 지난 사진 %d건 삭제', self.retention_days, removed)
        return removed

    def stats(self):
        """수집 현황 요약."""
        try:
            total = self._conn.execute('SELECT COUNT(*) FROM persons').fetchone()[0]
            days = self._conn.execute(
                'SELECT COUNT(DISTINCT day) FROM persons').fetchone()[0]
            visits = self._conn.execute(
                'SELECT COUNT(DISTINCT visit_id) FROM persons').fetchone()[0]
            faces = self._conn.execute(
                'SELECT COUNT(*) FROM persons WHERE face_image IS NOT NULL').fetchone()[0]
            talked = self._conn.execute(
                'SELECT COUNT(DISTINCT visit_id) FROM persons WHERE interacted=1').fetchone()[0]
            return {'photos': total, 'days': days, 'visits': visits,
                    'with_face': faces, 'visits_talked': talked,
                    'today': self._day_count}
        except Exception:
            return {}

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def main():
    """현황 보기:  python3 person_db.py  /  --prune 로 즉시 정리."""
    import argparse

    ap = argparse.ArgumentParser(description='사람 데이터셋 현황')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--prune', action='store_true', help='보관기간 지난 것 즉시 삭제')
    ap.add_argument('--days', type=int, default=RETENTION_DAYS)
    args = ap.parse_args()

    db = PersonDB(root=args.root, retention_days=args.days)
    if args.prune:
        print('삭제:', db.prune(), '건')
    s = db.stats()
    if not s:
        print('데이터 없음')
        return
    print('사람 데이터셋 (%s)' % args.root)
    print('  사진 %d장 · 방문 %d회 · %d일치' % (s['photos'], s['visits'], s['days']))
    print('  얼굴 크롭 있는 것 %d장' % s['with_face'])
    print('  대화까지 이어진 방문 %d회' % s['visits_talked'])
    print('  오늘 %d장 (하루 상한 %d)' % (s['today'], DAILY_MAX))
    db.close()


if __name__ == '__main__':
    main()
