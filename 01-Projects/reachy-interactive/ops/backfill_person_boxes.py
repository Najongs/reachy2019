"""이미 모아 둔 사진에 사람 위치를 채워 넣는다 (DGX 에서 제대로 된 모델로).

사람 위치(person_x/y/w/h)를 저장하기 시작한 것은 나중이라, 그 전에 찍힌 사진은
상자가 비어 있다. 상자가 없으면 사람만 잘라 낼 수 없다.

로봇(Pi, armv7l)에서는 가벼운 MobileNet-SSD 를 쓸 수밖에 없지만, 여기서는 그럴
이유가 없다. torchvision 의 COCO 사전학습 Faster R-CNN 을 쓴다 - 멀리 있거나
일부만 보이는 사람도 훨씬 잘 잡는다.

torch 는 학습용 venv 에 들어 있으므로 그 파이썬으로 돌린다:

    /home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3 \\
        ops/backfill_person_boxes.py --write

GPU 는 기본으로 쓰지 않는다. 이 DGX 는 다른 학습이 8장을 100% 로 쓰고 있어서,
끼어들면 그쪽이 느려진다. 사진 수백 장 정도는 CPU 로 충분하다. 급하면
--device cuda 를 준다.

사용:
    ... ops/backfill_person_boxes.py            # 무엇이 채워질지 보기
    ... ops/backfill_person_boxes.py --write    # 실제로 DB 에 기록
    ... ops/backfill_person_boxes.py --all      # 이미 채워진 것까지 다시
"""

import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                    '04-Archives', 'person-dataset'))
DEFAULT_DB = os.path.join(BASE, 'persons', 'persons.db')
DEFAULT_ROOT = os.path.join(BASE, 'persons')
VENV_PY = '/home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3'

COCO_PERSON = 1        # torchvision COCO 라벨에서 'person'


def load_detector(device):
    """COCO 사전학습 검출기를 올린다. (모델, 변환) 을 돌려준다."""
    import torch
    from torchvision.models import detection

    weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model, weights.transforms()


def best_person(model, transform, device, path, min_conf):
    """가장 확실한 사람의 (신뢰도, x, y, w, h) — 없으면 None."""
    import torch
    from torchvision.io import decode_image

    img = decode_image(path)               # uint8 CHW
    if img.shape[0] == 1:                  # 흑백이면 3채널로
        img = img.repeat(3, 1, 1)
    batch = [transform(img).to(device)]
    out = model(batch)[0]

    best = None
    for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
        if int(label) != COCO_PERSON:
            continue
        conf = float(score)
        if conf < min_conf or (best is not None and conf <= best[0]):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        best = (conf, int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    return best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--root', default=DEFAULT_ROOT, help='사진이 있는 폴더')
    ap.add_argument('--write', action='store_true', help='실제로 DB 에 기록')
    ap.add_argument('--all', action='store_true',
                    help='이미 상자가 있는 것까지 다시 검사')
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
        print('      ops/backfill_person_boxes.py %s'
              % ('--write' if args.write else ''))
        return 1

    conn = sqlite3.connect(args.db)
    where = '' if args.all else 'WHERE person_w IS NULL'
    rows = conn.execute(
        'SELECT id, robot, image FROM persons %s ORDER BY id' % where).fetchall()
    if not rows:
        print('채울 사진이 없습니다 (이미 전부 기록돼 있습니다).')
        return 0

    print('%d장 검사 (%s, 신뢰도 %.2f 이상)'
          % (len(rows), args.device, args.min_conf))
    model, transform = load_detector(args.device)

    found = empty = missing = 0
    for rid, robot, image in rows:
        path = os.path.join(args.root, robot, image)
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            got = best_person(model, transform, args.device, path, args.min_conf)
        except Exception as exc:
            print('  %-28s 실패: %s' % (image, exc))
            missing += 1
            continue
        if got is None:
            empty += 1
            continue
        conf, x, y, w, h = got
        found += 1
        print('  %-28s 사람 %.2f  (%d,%d %dx%d)' % (image, conf, x, y, w, h))
        if args.write:
            conn.execute(
                '''UPDATE persons SET person_x=?, person_y=?, person_w=?,
                   person_h=?, person_conf=? WHERE id=?''',
                (x, y, w, h, conf, rid))

    if args.write:
        conn.commit()

    print('\n사람을 찾은 사진 %d장 · 사람이 없던 사진 %d장 · 못 읽은 사진 %d장'
          % (found, empty, missing))
    if not args.write:
        print('실제로 기록하려면 --write 를 주세요.')
    else:
        print('기록 완료. 이제 뽑아 내세요:')
        print('  python3 ops/persons_export.py --out --crop-persons')
    return 0


if __name__ == '__main__':
    sys.exit(main())
