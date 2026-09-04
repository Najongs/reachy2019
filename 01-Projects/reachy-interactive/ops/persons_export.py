"""중앙 사람 데이터셋에서 쓸 만한 것만 골라 내보낸다.

모으는 것과 쓰는 것은 다른 일이다. 복도에서 자동으로 찍으면 흔들린 사진, 빈
복도, 같은 순간의 중복이 섞인다. 이 도구는 조건을 걸어 걸러낸 뒤, 학습에 바로
쓸 수 있는 폴더 구조로 복사하고 목록(CSV)을 함께 남긴다.

사용:
    python3 ops/persons_export.py --list            # 어떤 게 걸러지는지 미리 보기
    python3 ops/persons_export.py --out             # 기본 폴더로 내보내기
    python3 ops/persons_export.py --faces-only --min-face 80 --out ~/ds/faces
    python3 ops/persons_export.py --day 2026-09-04 --out ~/ds/oneday

기본 조건:
    선명도 >= 120        흔들린 사진 제외 (200 이상이면 아주 선명)
    방문당 최대 4장       같은 사람 같은 각도가 데이터셋을 채우지 않게
"""

import argparse
import csv
import os
import shutil
import sqlite3
import sys

_BASE = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '04-Archives', 'person-dataset'))
DEFAULT_DB = os.path.join(_BASE, 'persons', 'persons.db')
DEFAULT_ROOT = os.path.join(_BASE, 'persons')
DEFAULT_OUT = os.path.join(_BASE, 'dataset')


def select(conn, args):
    """조건에 맞는 행을 고른다. 방문당 장수 제한은 파이썬 쪽에서 건다."""
    where, params = ['1=1'], []
    if args.faces_only:
        where.append('face_image IS NOT NULL')
    if args.min_sharpness:
        where.append('(sharpness IS NULL OR sharpness >= ?)')
        params.append(args.min_sharpness)
    if args.min_face:
        where.append('face_w >= ?')
        params.append(args.min_face)
    if args.day:
        where.append('day = ?')
        params.append(args.day)
    if args.since:
        where.append('day >= ?')
        params.append(args.since)
    if args.interacted_only:
        where.append('interacted = 1')

    if args.min_person:
        where.append('person_w >= ?')
        params.append(args.min_person)

    rows = conn.execute(
        '''SELECT id, robot, ts, day, visit_id, image, face_image,
                  face_w, face_h, sharpness, interacted,
                  person_x, person_y, person_w, person_h, person_conf
           FROM persons WHERE %s
           ORDER BY day, ts, seq''' % ' AND '.join(where), params).fetchall()

    # 방문당 상한: 한 사람이 오래 서 있었다고 그 사람만 잔뜩 들어가면 안 된다.
    # 선명한 것부터 남긴다.
    if args.per_visit:
        by_visit = {}
        for r in rows:
            by_visit.setdefault((r[1], r[4]), []).append(r)
        kept = []
        for group in by_visit.values():
            group.sort(key=lambda r: -(r[9] or 0))
            kept.extend(group[:args.per_visit])
        kept.sort(key=lambda r: (r[3], r[2]))
        rows = kept
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--root', default=DEFAULT_ROOT, help='사진이 있는 폴더')
    ap.add_argument('--out', nargs='?', const=DEFAULT_OUT,
                    help='내보낼 폴더 (기본: 04-Archives/person-dataset/dataset)')
    ap.add_argument('--list', action='store_true', help='목록만 출력')
    ap.add_argument('--faces-only', action='store_true',
                    help='얼굴이 잡힌 사진만 (움직임만으로 찍힌 것 제외)')
    ap.add_argument('--min-sharpness', type=float, default=120.0,
                    help='이보다 흐린 사진 제외 (기본 120)')
    ap.add_argument('--min-face', type=int, default=0,
                    help='얼굴 가로 픽셀 하한 (예: 80)')
    ap.add_argument('--min-person', type=int, default=0,
                    help='사람 가로 픽셀 하한 (예: 150). 멀리 찍힌 것 제외')
    ap.add_argument('--crop-persons', action='store_true',
                    help='사람마다 잘라 persons/ 에 저장 + persons.csv (학습용)')
    ap.add_argument('--per-visit', type=int, default=4,
                    help='한 방문에서 최대 몇 장 (0 이면 제한 없음)')
    ap.add_argument('--day', help='이 날짜만 (YYYY-MM-DD)')
    ap.add_argument('--since', help='이 날짜 이후만')
    ap.add_argument('--interacted-only', action='store_true',
                    help='실제로 대화까지 이어진 방문만')
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print('중앙 DB가 없습니다: %s' % args.db)
        print('먼저 가져오세요:  python3 ops/sync_persons.py')
        return 1

    conn = sqlite3.connect(args.db)
    total = conn.execute('SELECT COUNT(*) FROM persons').fetchone()[0]
    rows = select(conn, args)

    print('전체 %d장 중 조건에 맞는 것 %d장' % (total, len(rows)))
    if not rows:
        print('  (조건을 완화해 보세요: --min-sharpness 0 --per-visit 0)')
        return 0

    days = sorted({r[3] for r in rows})
    visits = len({(r[1], r[4]) for r in rows})
    faces = sum(1 for r in rows if r[6])
    print('  %d일치 · 방문 %d회 · 얼굴 크롭 있는 것 %d장' % (len(days), visits, faces))

    if args.list or not args.out:
        for r in rows[:30]:
            print('   %s  %s  선명도 %-6s %s'
                  % (r[2], r[4], ('%.0f' % r[9]) if r[9] else '-', r[5]))
        if len(rows) > 30:
            print('   ... 외 %d장' % (len(rows) - 30))
        if not args.out:
            print('\n실제로 내보내려면 --out <폴더> 를 주세요.')
        return 0

    # 내보내기: 날짜별 폴더 + 목록 CSV. 원본은 그대로 두고 복사만 한다.
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    img_out = os.path.join(out, 'images')
    face_out = os.path.join(out, 'faces')
    person_out = os.path.join(out, 'persons')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(face_out, exist_ok=True)
    crop_cv = None
    if args.crop_persons:
        try:
            import cv2 as crop_cv          # noqa: F401  (자르기에만 쓴다)
        except ImportError:
            print('! --crop-persons 를 쓰려면 opencv 가 필요합니다 '
                  '(pip3 install opencv-python-headless). 자르기는 건너뜁니다.')
            crop_cv = None
        else:
            os.makedirs(person_out, exist_ok=True)

    copied = missing = 0
    person_rows = []
    with open(os.path.join(out, 'index.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'face_file', 'person_file', 'robot', 'ts', 'day',
                    'visit_id', 'face_w', 'face_h', 'person_x', 'person_y',
                    'person_w', 'person_h', 'person_conf', 'sharpness',
                    'interacted'])
        for r in rows:
            (rid, robot, ts, day, visit, image, face,
             fw, fh, sharp, inter, px, py, pw, ph, pconf) = r
            src = os.path.join(args.root, robot, image) if image else None
            if not src or not os.path.exists(src):
                missing += 1
                continue
            name = '%s_%s_%d.jpg' % (robot, day.replace('-', ''), rid)
            shutil.copy2(src, os.path.join(img_out, name))

            face_name = ''
            if face:
                fsrc = os.path.join(args.root, robot, face)
                if os.path.exists(fsrc):
                    face_name = name.replace('.jpg', '_face.jpg')
                    shutil.copy2(fsrc, os.path.join(face_out, face_name))
            # 사람 상자대로 잘라 두면 학습 때 바로 쓸 수 있다. 한 사진에 사람이
            # 여럿이면 전부 잘라 낸다 - 한 명만 남기면 나머지가 버려진다.
            person_name = ''
            if crop_cv is not None:
                boxes = conn.execute(
                    '''SELECT seq, x, y, w, h, conf FROM person_boxes
                       WHERE photo_id=? ORDER BY seq''', (rid,)).fetchall()
                if not boxes and pw:
                    boxes = [(0, px, py, pw, ph, pconf)]   # 예전 방식 대비
                made = []
                try:
                    img = crop_cv.imread(src) if boxes else None
                except Exception:
                    img = None
                for seq, bx, by, bw, bh, bconf in (boxes if img is not None else []):
                    try:
                        pad = int(0.06 * bw)
                        x0, y0 = max(0, bx - pad), max(0, by - pad)
                        x1 = min(img.shape[1], bx + bw + pad)
                        y1 = min(img.shape[0], by + bh + pad)
                        crop = img[y0:y1, x0:x1]
                        if not crop.size:
                            continue
                        cname = name.replace('.jpg', '_p%d.jpg' % seq)
                        crop_cv.imwrite(os.path.join(person_out, cname), crop,
                                        [int(crop_cv.IMWRITE_JPEG_QUALITY), 88])
                        made.append(cname)
                        person_rows.append([cname, name, robot, ts, day, visit,
                                            seq, bx, by, bw, bh, bconf])
                    except Exception:
                        continue
                person_name = ';'.join(made)
            w.writerow([name, face_name, person_name, robot, ts, day, visit,
                        fw, fh, px, py, pw, ph, pconf, sharp, inter])
            copied += 1

    # 사람 단위 목록: 학습에는 이쪽이 바로 쓰인다 (한 줄에 사람 하나).
    if person_rows:
        with open(os.path.join(out, 'persons.csv'), 'w', newline='') as fh:
            pw_ = csv.writer(fh)
            pw_.writerow(['file', 'source_image', 'robot', 'ts', 'day',
                          'visit_id', 'seq', 'x', 'y', 'w', 'h', 'conf'])
            pw_.writerows(person_rows)

    print('\n내보냄: %s' % out)
    print('  images/ %d장 · faces/ · index.csv' % copied)
    if crop_cv is not None:
        n = len(os.listdir(person_out)) if os.path.isdir(person_out) else 0
        multi = len({r[1] for r in person_rows}) if person_rows else 0
        print('  persons/ %d명 (사진 %d장에서) · persons.csv' % (n, multi))
    if missing:
        print('  ! 파일을 찾지 못한 행 %d건 (sync_persons.py 를 다시 돌려 보세요)' % missing)
    return 0


if __name__ == '__main__':
    sys.exit(main())
