"""가져온 사진에서 사람 인식 DB 를 만든다 (DGX 에서 제대로 된 모델로).

역할을 나눈다.
  로봇(Pi)  - '사람이 있다'만 판단해서 사진을 모은다. armv7l 이라 가벼운
              MobileNet-SSD 밖에 못 쓴다. 상자는 대충 잡힌다(box_source='pi-ssd').
  여기(DGX) - 가져온 사진을 torchvision 의 COCO 사전학습 Faster R-CNN 으로 다시
              훑어 사람 위치를 제대로 잡는다(box_source='dgx-frcnn').

실측으로 같은 사진 15장에서 SSD 는 4장, Faster R-CNN 은 13장을 신뢰도
0.99~1.00 으로 잡았다. 멀리 있거나 일부만 보이는 사람도 훨씬 잘 잡는다.

torch 는 학습용 venv 에 들어 있으므로 그 파이썬으로 돌린다:

    /home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3 \\
        ops/data/backfill_person_boxes.py --write

GPU 는 기본으로 쓰지 않는다. 이 DGX 는 다른 학습이 8장을 100% 로 쓰고 있어서,
끼어들면 그쪽이 느려진다. 사진 수백 장 정도는 CPU 로 충분하다. 급하면
--device cuda 를 준다.

사용:
    ... ops/data/backfill_person_boxes.py            # 무엇이 채워질지 보기
    ... ops/data/backfill_person_boxes.py --write    # 실제로 DB 에 기록
    ... ops/data/backfill_person_boxes.py --all      # 이미 채워진 것까지 다시
"""

import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import para  # PARA 기준 경로 (위치 계산은 para.py 한 곳에만)
BASE = para.PERSON_DATASET
DEFAULT_DB = os.path.join(para.PERSONS, 'persons.db')
DEFAULT_ROOT = para.PERSONS
VENV_PY = '/home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3'

COCO_PERSON = 1        # torchvision COCO 라벨에서 'person'


MIN_FACE = 60          # 얼굴 인식에 쓰려면 이 정도 폭은 되어야 한다(px)
# 키포인트 점수 하한. 이 모델은 가려진 부위도 점수와 함께 추측해서, 낮게 잡으면
# 뒤통수가 얼굴로 들어온다(실측: 뒤통수 코 6.9 대 정면 20.1).
FACE_KP_MIN = 10.0


def load_pose(device):
    """얼굴 위치를 뽑을 키포인트 모델 (사람 검출기와 별개)."""
    from torchvision.models import detection

    weights = detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = detection.keypointrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    return model, weights.transforms()


def face_boxes(model, transform, device, img):
    """사진에서 (얼굴상자, 그 사람의 몸상자) 쌍을 뽑는다. 원본 좌표.

    얼굴을 몸에 짝지을 때 '몸 상자 안에 얼굴 중심이 들어가나' 로 판단하면
    틀린다 - 가까이 있는 사람의 큰 상자 안에 뒷사람 머리가 들어가서, 앞사람의
    얼굴로 뒤통수가 배정됐다(실측). 키포인트 모델은 사람마다 상자와 키포인트를
    함께 주므로, 그 짝을 그대로 쓰고 나중에 IoU 로 맞춘다.

    코와 눈이 함께 뚜렷해야 얼굴로 친다. 귀만 보이는 것은 뒤통수다.
    """
    out = model([transform(img).to(device)])[0]
    pairs = []
    for kps, ksc, score, pbox in zip(out['keypoints'], out['keypoints_scores'],
                                     out['scores'], out['boxes']):
        if float(score) < 0.8:
            continue
        nose_s = float(ksc[0])
        eye_s = max(float(ksc[1]), float(ksc[2]))
        if nose_s < FACE_KP_MIN or eye_s < FACE_KP_MIN:
            continue
        nose_x = float(kps[0][0])
        lx, rx = float(kps[1][0]), float(kps[2][0])
        if not (min(lx, rx) - 5 <= nose_x <= max(lx, rx) + 5):
            continue                       # 코가 두 눈 사이가 아니면 정면이 아니다

        vis = [i for i in range(5) if float(ksc[i]) > FACE_KP_MIN]
        xs = [float(kps[i][0]) for i in vis]
        ys = [float(kps[i][1]) for i in vis]
        side = max(max(xs) - min(xs), 1.0) * 2.2
        side = max(side, 40.0)
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        face = (max(0, int(cx - side / 2)), max(0, int(cy - side * 0.55)),
                int(side), int(side * 1.2))
        pairs.append((face, [float(v) for v in pbox]))
    return pairs


def load_detector(device):
    """COCO 사전학습 검출기를 올린다. (모델, 변환) 을 돌려준다."""
    import torch
    from torchvision.models import detection

    weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model, weights.transforms()


def others_in_crop(model, transform, device, img, box, pad_frac=0.06):
    """이 사람을 잘라 냈을 때 크롭 안에 보이는 '다른 사람'의 크기 비율.

    잘라 낸 그림을 검출기에 다시 넣어, 주인공 말고 두 번째로 큰 사람이 크롭을
    얼마나 차지하는지 본다. 상자끼리의 겹침으로 계산하면 틀린다 - 큰 사람의
    사각형 상자가 옆 사람 크롭에 걸쳐도 실제로는 안 보이는 경우가 있다.
    """
    import torch

    _, x, y, w, h = box
    H, W = img.shape[1], img.shape[2]
    pad = int(pad_frac * w)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0.0
    crop = img[:, y0:y1, x0:x1]
    ch, cw = crop.shape[1], crop.shape[2]
    out = model([transform(crop).to(device)])[0]

    found = []
    for b, l, sc in zip(out['boxes'], out['labels'], out['scores']):
        if int(l) != COCO_PERSON or float(sc) < 0.7:
            continue
        found.append([float(v) for v in b])
    if len(found) < 2:
        return 0.0

    # 가장 큰 것이 주인공. 검출기는 같은 사람을 머리/몸으로 두 번 잡기도 해서,
    # 주인공과 많이 겹치는 상자는 '다른 사람'으로 세면 안 된다(실측: 한 명뿐인
    # 크롭이 0.50 으로 표시됐다).
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    found.sort(key=area, reverse=True)
    main = found[0]
    worst = 0.0
    for b in found[1:]:
        ix = max(0.0, min(main[2], b[2]) - max(main[0], b[0]))
        iy = max(0.0, min(main[3], b[3]) - max(main[1], b[1]))
        inter = ix * iy
        union = area(main) + area(b) - inter
        if union > 0 and inter / union > 0.3:
            continue                   # 주인공을 겹쳐 잡은 것
        worst = max(worst, area(b) / float(cw * ch))
    return round(worst, 3)


def all_persons(model, transform, device, path, min_conf):
    """사진에 있는 사람 전부. [(신뢰도, x, y, w, h)] 를 큰 사람부터.

    한 사진에 사람이 여럿이면 전부 남긴다 - 한 명만 잘라 내면 나머지가 버려진다
    (실측: 사진 25장 중 6장에 두 명 이상 있었다).
    """
    import torch
    from torchvision.io import decode_image

    img = decode_image(path)               # uint8 CHW
    if img.shape[0] == 1:                  # 흑백이면 3채널로
        img = img.repeat(3, 1, 1)
    batch = [transform(img).to(device)]
    out = model(batch)[0]

    got = []
    for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
        if int(label) != COCO_PERSON or float(score) < min_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        got.append((float(score), int(x1), int(y1),
                    int(x2 - x1), int(y2 - y1)))
    got.sort(key=lambda g: -(g[3] * g[4]))      # 큰 사람부터
    return got, img


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--root', default=DEFAULT_ROOT, help='사진이 있는 폴더')
    ap.add_argument('--write', action='store_true', help='실제로 DB 에 기록')
    ap.add_argument('--all', action='store_true',
                    help='여기서 이미 잡은 것까지 전부 다시 검사')
    ap.add_argument('--device', default='cpu',
                    help="'cpu'(기본) 또는 'cuda'. 이 DGX 는 다른 학습이 GPU 를 "
                         '꽉 채우고 있어 기본은 cpu 다')
    ap.add_argument('--min-conf', type=float, default=0.7,
                    help='이 신뢰도 이상만 사람으로 본다 (기본 0.7)')
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print('DB 가 없습니다: %s' % args.db)
        return 1

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        print('torch/torchvision 이 없습니다. 학습용 venv 로 돌리세요:')
        print('  %s \\' % VENV_PY)
        print('      ops/data/backfill_person_boxes.py %s'
              % ('--write' if args.write else ''))
        return 1

    conn = sqlite3.connect(args.db)
    # 보통은 sync_persons 가 먼저 만들어 두지만, 이것만 따로 돌릴 수도 있다.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS person_boxes (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,     -- persons.id
            seq      INTEGER,              -- 그 사진에서 몇 번째 사람(큰 순)
            x INTEGER, y INTEGER, w INTEGER, h INTEGER,
            conf     REAL,
            source   TEXT,                 -- 누가 잡았나 ('dgx-frcnn')
            -- 이 사람을 잘라 냈을 때 크롭 안에 '다른 사람'이 얼마나 보이나
            -- (0~1). 상자끼리의 겹침으로는 못 잰다 - 상자는 사각형이라 큰
            -- 사람의 상자가 옆 사람 크롭에 걸쳐도 실제로는 안 보이는 경우가
            -- 있다(실측으로 확인). 그래서 잘라 낸 그림을 검출기에 다시 넣어
            -- 잰다. 군집할 때 오염된 크롭을 걸러 내는 데 쓴다.
            other_in_crop REAL,
            -- 얼굴 위치(원본 프레임 좌표). 며칠에 걸쳐 같은 사람을 알아보려면
            -- 얼굴이 필요하다 - 옷/체형 임베딩은 옷을 갈아입으면 무너진다.
            -- 로봇의 Haar 는 이 복도에서 거의 안 잡히므로 여기서 키포인트로 뽑는다.
            face_x INTEGER, face_y INTEGER, face_w INTEGER, face_h INTEGER,
            -- 나중에 임베딩으로 군집을 지어 같은 사람을 엮을 자리.
            -- 지금은 비어 있고, 군집 도구가 채운다.
            identity_id TEXT,
            UNIQUE(photo_id, seq)
        )''')
    for _col, _typ in (('identity_id', 'TEXT'), ('other_in_crop', 'REAL'),
                       ('face_x', 'INTEGER'), ('face_y', 'INTEGER'),
                       ('face_w', 'INTEGER'), ('face_h', 'INTEGER')):
        try:
            conn.execute('ALTER TABLE person_boxes ADD COLUMN %s %s'
                         % (_col, _typ))
        except Exception:
            pass                           # 이미 있는 칼럼
    conn.commit()

    # 로봇이 잡은 상자(pi-ssd)도 다시 잡는다. 인식 DB 는 여기서 만드는 것이
    # 원칙이고, 실측 차이가 크다(15장 중 4장 대 13장).
    where = ('' if args.all else
             "WHERE box_source IS NULL OR box_source <> 'dgx-frcnn'")
    rows = conn.execute(
        'SELECT id, robot, image FROM persons %s ORDER BY id' % where).fetchall()
    if not rows:
        print('채울 사진이 없습니다 (이미 전부 기록돼 있습니다).')
        return 0

    print('%d장 검사 (%s, 신뢰도 %.2f 이상)'
          % (len(rows), args.device, args.min_conf))
    model, transform = load_detector(args.device)
    pose, pose_tf = load_pose(args.device)

    found = empty = missing = people = dirty = usable_faces = 0
    for rid, robot, image in rows:
        path = os.path.join(args.root, robot, image)
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            got, img = all_persons(model, transform, args.device, path,
                                   args.min_conf)
        except Exception as exc:
            print('  %-28s 실패: %s' % (image, exc))
            missing += 1
            continue
        if not got:
            empty += 1
            # 사람을 못 찾았어도 '여기서 봤다'고 남긴다. 안 그러면 매번 다시
            # 검사하게 된다.
            if args.write:
                conn.execute(
                    "UPDATE persons SET box_source='dgx-frcnn' WHERE id=?",
                    (rid,))
            continue
        conf, x, y, w, h = got[0]          # 가장 큰 사람
        found += 1
        people += len(got)
        print('  %-28s 사람 %d명 (가장 큰 사람 %.2f, %dx%d)'
              % (image, len(got), conf, w, h))
        if args.write:
            # 사진 행에는 가장 큰 사람을 남긴다(예전 질의 호환).
            conn.execute(
                '''UPDATE persons SET person_x=?, person_y=?, person_w=?,
                   person_h=?, person_conf=?, box_source='dgx-frcnn'
                   WHERE id=?''',
                (x, y, w, h, conf, rid))
            # 사람별 상자는 따로. 다시 돌려도 겹치지 않게 지우고 새로 넣는다.
            conn.execute('DELETE FROM person_boxes WHERE photo_id=?', (rid,))
            # 얼굴은 사람마다 짝지어 둔다. 며칠 뒤에도 같은 사람을 알아보려면
            # 옷이 아니라 얼굴이 필요하다.
            try:
                faces = face_boxes(pose, pose_tf, args.device, img)
            except Exception:
                faces = []

            def face_for(person):
                """이 사람의 얼굴. 키포인트 모델이 준 몸 상자와 IoU 로 맞춘다."""
                _, px_, py_, pw_, ph_ = person
                a = (px_, py_, px_ + pw_, py_ + ph_)
                best, best_iou = None, 0.4      # 이 정도는 겹쳐야 같은 사람
                for face, b in faces:
                    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                    inter = ix * iy
                    union = (pw_ * ph_ + (b[2] - b[0]) * (b[3] - b[1]) - inter)
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best, best_iou = face, iou
                return best

            rows_out = []
            for i, g in enumerate(got):
                # 사람이 하나뿐이면 크롭에 다른 사람이 있을 수 없다 - 검사 생략.
                other = (others_in_crop(model, transform, args.device, img, g)
                         if len(got) > 1 else 0.0)
                if other > 0.15:
                    dirty += 1
                fb = face_for(g)
                if fb and fb[2] >= MIN_FACE:
                    usable_faces += 1
                rows_out.append((rid, i, g[1], g[2], g[3], g[4], g[0], other,
                                 fb[0] if fb else None, fb[1] if fb else None,
                                 fb[2] if fb else None, fb[3] if fb else None))
            conn.executemany(
                '''INSERT INTO person_boxes
                   (photo_id, seq, x, y, w, h, conf, other_in_crop,
                    face_x, face_y, face_w, face_h, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'dgx-frcnn')''', rows_out)

    if args.write:
        conn.commit()

    print('\n사람을 찾은 사진 %d장 (사람 %d명) · 사람이 없던 사진 %d장 · '
          '못 읽은 사진 %d장' % (found, people, empty, missing))
    if dirty:
        print('  크롭에 다른 사람이 15%% 넘게 걸친 것 %d명 '
              '(persons.csv 의 other_in_crop 으로 거를 수 있다)' % dirty)
    if people:
        print('  얼굴이 %dpx 이상으로 찍힌 사람 %d명 / %d명 '
              '(며칠 뒤에도 알아보려면 이쪽이 쓰인다)'
              % (MIN_FACE, usable_faces, people))
    if not args.write:
        print('실제로 기록하려면 --write 를 주세요.')
    else:
        print('기록 완료. 이제 뽑아 내세요:')
        print('  python3 ops/persons_export.py --out --crop-persons')
    return 0


if __name__ == '__main__':
    sys.exit(main())
