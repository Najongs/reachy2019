"""인지 층 - 모델이 세계를 '보는' 유일한 창구. 참값 방화벽이 여기 있다.

계획 모델(opus)이 참 3D 좌표를 보면 실물로 절대 못 넘어간다. 실물에는 참값이
없기 때문이다. 그래서 이 모듈이 경계다:

  * detect()      참값을 픽셀로 투영하되 **잡음과 미검출**을 섞어 실물 SSD 의
                  거동을 흉내낸다. 모델은 이것만 본다.
  * overlay()     눈 프레임에 검출 박스를 번호로 그린다 - opus 는 "몇 번 박스"
                  로만 대상을 지정한다(픽셀 좌표를 계산할 필요 없음).
  * SceneMemory   지속적 상황 인지(결정론). 시선을 훑는 동안의 검출을 누적해
                  '지금 화면 밖이지만 아까 저기 있었다' 를 기억한다. 자기 손이
                  물체를 가리거나 검출이 깜빡여도 버틴다. LLM 이 아니다 -
                  opus 는 이 기억의 요약 텍스트를 받는다.

기억은 검출로만 만들어지므로 방화벽이 유지된다. 3D 참값은 오직 채점
(sim_tasks.success_fn)에만 쓰인다.

사용:
    python3 sim/sim_perceive.py --demo     # 오버레이 + 기억 데모 프레임 저장
"""

import math
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')   # GPU0 만 쓴다

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import sim_world as W                              # noqa: E402

# 물체 종류별 대략적 픽셀 크기 계산용 실제 치수(가장 큰 축, m).
KIND_SIZE = {'cup': 0.10, 'block': 0.06, 'ball': 0.06, 'can': 0.12,
             'tray': 0.14}
KO = {'cup': '컵', 'block': '블록', 'ball': '공', 'can': '캔', 'tray': '쟁반'}


def detect(world, noise_px=2.0, dropout=0.05, rng=None):
    """지금 눈 프레임에서 보이는 물체들의 '검출'.

    실물 SSD 를 흉내낸다: 픽셀 중심에 가우시안 잡음, 가끔 통째로 미검출.
    반환: [{name(디버그용), kind, u, v, size_px}] - 화면에 실제로 잡힌 것만.
    name 은 채점·기억 대응용으로만 쓰고 모델 프롬프트에는 넣지 않는다
    (실물 검출기는 물체의 '이름' 을 모른다 - 종류만 안다).
    """
    import random
    rng = rng or random
    out = []
    for o in world.objects:
        if o['name'] == 'table':
            continue
        pos = world.object_pos(o['name'])
        px = world.pixel_of(pos)
        if px is None:
            continue
        if not (0 <= px[0] < world.width and 0 <= px[1] < world.height):
            continue
        if rng.random() < dropout:
            continue                                # 이번 프레임은 놓쳤다
        # 크기: 거리에 반비례하는 화면 폭 (fovy 58도 근사)
        cam_d = _cam_dist(world, pos)
        f = (world.height / 2.0) / math.tan(math.radians(58 / 2.0))
        size_px = KIND_SIZE.get(o['kind'], 0.08) * f / max(cam_d, 0.05)
        out.append({
            'name': o['name'],
            'kind': o['kind'],
            'u': float(px[0] + rng.gauss(0, noise_px)),
            'v': float(px[1] + rng.gauss(0, noise_px)),
            'size_px': float(size_px * (1 + rng.gauss(0, 0.06))),
        })
    return out


def _cam_dist(world, point):
    mj = world.mujoco
    cid = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_CAMERA, 'robot_eye')
    cam = world.data.cam_xpos[cid]
    return math.sqrt(sum((point[i] - cam[i]) ** 2 for i in range(3)))


def overlay(world, dets, note=''):
    """눈 프레임에 검출 박스를 번호로 그린다. (ndarray 반환)"""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(world.render_eye())
    d = ImageDraw.Draw(img)
    for i, det in enumerate(dets):
        r = max(10, det['size_px'] / 2)
        box = [det['u'] - r, det['v'] - r, det['u'] + r, det['v'] + r]
        d.rectangle(box, outline=(80, 220, 120), width=3)
        d.rectangle([box[0], box[1] - 16, box[0] + 30, box[1]],
                    fill=(80, 220, 120))
        d.text((box[0] + 4, box[1] - 15), '[%d]' % i, fill=(10, 20, 10))
    if note:
        d.text((10, 8), note, fill=(235, 235, 235))
    return np.asarray(img)


class SceneMemory(object):
    """시선을 훑는 동안 본 것들을 기억한다 - 지속적 상황 인지(결정론).

    항목: {kind, gaze(그때의 시선 y,z), u, v, size_px, conf, seen_at}.
    같은 종류·비슷한 방향의 검출은 같은 항목으로 갱신하고, 오래 못 보면
    신뢰도가 감쇠한다. '화면 밖이지만 아까 저기 있었다' 를 제공한다.
    """

    DECAY_S = 20.0          # 이 시간 못 보면 신뢰도가 절반으로
    MATCH_DEG = 8.0         # 방향 차가 이 안이면 같은 물체로 본다

    def __init__(self):
        self.items = []

    def _direction(self, world, det):
        """검출의 대략적 방향(팬/틸트 각도). 시선 + 픽셀 오프셋으로 계산.

        참값을 쓰지 않는다 - 시선 각도(우리가 명령한 값)와 화면 픽셀만으로
        방향을 복원한다. 실물에서도 똑같이 계산할 수 있는 값이다.
        """
        f = (world.height / 2.0) / math.tan(math.radians(58 / 2.0))
        du = math.degrees(math.atan2(det['u'] - world.width / 2, f))
        dv = math.degrees(math.atan2(det['v'] - world.height / 2, f))
        pan = world.pose.get('eye.pan', 0.0) - du
        tilt = world.pose.get('eye.tilt', 0.0) + dv
        return pan, tilt

    def update(self, world, dets):
        now = time.time()
        for det in dets:
            pan, tilt = self._direction(world, det)
            match = None
            for it in self.items:
                if it['kind'] != det['kind']:
                    continue
                if (abs(it['pan'] - pan) < self.MATCH_DEG
                        and abs(it['tilt'] - tilt) < self.MATCH_DEG):
                    match = it
                    break
            if match is None:
                match = {'kind': det['kind']}
                self.items.append(match)
            match.update({'pan': pan, 'tilt': tilt, 'u': det['u'],
                          'v': det['v'], 'size_px': det['size_px'],
                          'conf': 1.0, 'seen_at': now,
                          'name': det.get('name')})
        # 감쇠
        for it in self.items:
            age = now - it['seen_at']
            it['conf'] = 0.5 ** (age / self.DECAY_S)
        self.items = [it for it in self.items if it['conf'] > 0.1]

    def summary(self):
        """opus 에게 줄 요약. 참값 없음 - 방향과 마지막 목격뿐."""
        if not self.items:
            return '기억: 아직 본 물체 없음.'
        lines = ['기억(시선을 훑으며 본 것):']
        now = time.time()
        for it in sorted(self.items, key=lambda x: -x['conf']):
            ago = now - it['seen_at']
            where = '팬 %+.0f도 틸트 %+.0f도 방향' % (it['pan'], it['tilt'])
            when = '방금' if ago < 2 else '%.0f초 전' % ago
            lines.append('  - %s: %s, %s 목격' % (
                KO.get(it['kind'], it['kind']), where, when))
        return '\n'.join(lines)

    def find(self, kind):
        """이 종류를 마지막으로 본 방향. 없으면 None."""
        best = None
        for it in self.items:
            if it['kind'] == kind and (best is None or it['conf'] > best['conf']):
                best = it
        return best


def sweep(world, memory, pans=(-35, -18, 0, 18, 35), noise_px=2.0,
          dropout=0.05, rng=None, frames=None):
    """시선을 좌우로 훑으며 기억을 채운다. 훑은 프레임 수를 돌려준다."""
    for pan in pans:
        y = 0.5 * math.tan(math.radians(pan))
        world.glide_gaze(y, world.table_top - 0.08)
        dets = detect(world, noise_px, dropout, rng)
        memory.update(world, dets)
        if frames is not None:
            frames.append(overlay(world, dets, 'sweep pan %+d' % pan))
    return len(pans)


def gaze_toward(world, item):
    """기억 항목의 방향으로 시선을 되돌린다 (참값 없이)."""
    pan, tilt = item['pan'], item['tilt']
    y = 0.5 * math.tan(math.radians(pan))
    # tilt 각도 -> 시선 z (x=0.5 평면 기준)
    z = -math.hypot(0.5, y) * math.tan(math.radians(tilt))
    world.glide_gaze(y, z)


def main():
    import argparse
    import random

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--out', default='sim_data/gallery/perceive_demo.png')
    args = ap.parse_args()

    if not args.demo:
        ap.error('--demo 로 실행')

    rng = random.Random(3)
    objs = [W.make_object('cup0', 'cup', (0.31, -0.14)),
            W.make_object('ball0', 'ball', (0.29, 0.02)),
            W.make_object('can0', 'can', (0.35, -0.05))]
    world = W.World(objs)
    mem = SceneMemory()

    frames = []
    sweep(world, mem, rng=rng, frames=frames)
    print(mem.summary())

    # 화면 밖 기억 시험: 왼쪽 끝을 본 뒤에도 오른쪽 물체를 기억하나
    world.glide_gaze(0.45, world.table_top - 0.08)
    dets_now = detect(world, rng=rng)
    visible_kinds = {d['kind'] for d in dets_now}
    remembered = {it['kind'] for it in mem.items}
    print('\n지금 화면에 보이는 것: %s' % (sorted(visible_kinds) or '없음'))
    print('기억하는 것: %s' % sorted(remembered))
    off = remembered - visible_kinds
    print('화면 밖인데 기억: %s  <- 이게 지속적 상황 인지' % (sorted(off) or '없음'))

    # 기억으로 시선 복귀 시험
    item = mem.find('cup')
    if item:
        gaze_toward(world, item)
        back = detect(world, rng=rng)
        found = any(d['kind'] == 'cup' for d in back)
        print('기억 방향으로 시선 복귀 -> 컵이 다시 보이나: %s' % found)

    import numpy as np
    from PIL import Image
    strip = np.concatenate(frames[:3], axis=1)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    Image.fromarray(strip).save(args.out)
    print('\n오버레이 데모: %s' % args.out)
    world.close()


if __name__ == '__main__':
    main()
