"""결투 기록으로 quality() 가중치를 보정 제안한다 (제안만, 자동 적용 없음).

Codex 쌍대 결투는 '사람 눈'의 대체물이다. 결투마다 두 동작의 성분
(효율/저크/여유/시야)과 승자가 남으므로, 성분 차이 -> 승패를 로지스틱
회귀로 맞추면 사람 눈이 실제로 쓰는 가중치가 나온다. 계측 가중치
(35/25/20/10/10)와 크게 어긋나는 축이 보이면 그게 보정 대상이다.

지표 감사 원칙에 따라 이 도구는 **제안을 출력할 뿐** sim_critic 을
바꾸지 않는다 - 바꾸는 건 사람이 검토하고 한다.

사용:
    python3 sim/sim_calibrate.py                 # 모든 체크포인트에서
    python3 sim/sim_calibrate.py --min-n 30      # 표본이 이만큼일 때만
"""

import argparse
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'sim_data')

AXES = ('efficiency', 'jerk_deg', 'min_margin_cm', 'view_ratio')
# 계측 방향: 저크는 낮을수록 좋으므로 부호를 뒤집어 정렬한다.
SIGN = {'efficiency': 1.0, 'jerk_deg': -1.0, 'min_margin_cm': 1.0,
        'view_ratio': 1.0}
# 축별 대략 스케일 (성분 차이를 비슷한 크기로 맞춘다)
SCALE = {'efficiency': 0.2, 'jerk_deg': 2.0, 'min_margin_cm': 2.0,
         'view_ratio': 0.3}


def load_duels():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'improve-*.json'))):
        try:
            hist = json.load(open(f)).get('history', [])
        except Exception:
            continue
        for h in hist:
            d = h.get('duel_detail')
            if not d or not d.get('winner'):
                continue
            a, b = d.get('a', {}), d.get('b', {})
            if any(a.get(k) is None or b.get(k) is None for k in AXES):
                continue
            x = [SIGN[k] * (b[k] - a[k]) / SCALE[k] for k in AXES]
            y = 1.0 if d['winner'] == 'B' else 0.0
            rows.append((x, y))
    return rows


def fit_logistic(rows, lr=0.1, epochs=400):
    w = [0.0] * len(AXES)
    for _ in range(epochs):
        for x, y in rows:
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            g = (p - y)
            w = [wi - lr * g * xi / len(rows) * len(rows) ** 0.5
                 for wi, xi in zip(w, x)]
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-n', type=int, default=20)
    args = ap.parse_args()

    rows = load_duels()
    print('결투 표본: %d건' % len(rows))
    if len(rows) < args.min_n:
        print('표본 부족 (%d < %d) - 더 쌓인 뒤 다시.' % (len(rows), args.min_n))
        return
    w = fit_logistic(rows)
    tot = sum(abs(v) for v in w) or 1.0
    # 정확도 (자기 표본)
    hit = 0
    for x, y in rows:
        z = sum(wi * xi for wi, xi in zip(w, x))
        hit += int((z > 0) == (y > 0.5))
    print('적합 정확도: %.0f%% (동전 던지기=50%%)' % (100.0 * hit / len(rows)))
    print('사람 눈(결투) 기준 상대 가중치  vs  계측 가중치:')
    ref = {'efficiency': 35, 'jerk_deg': 25, 'min_margin_cm': 20,
           'view_ratio': 10}
    ref_tot = sum(ref.values())
    for k, wi in zip(AXES, w):
        print('  %-14s %5.1f%%   vs %5.1f%%%s' % (
            k, 100.0 * abs(wi) / tot, 100.0 * ref[k] / ref_tot,
            '   (방향 반대!)' if wi * SIGN[k] < 0 and abs(wi) / tot > 0.1
            else ''))
    print('큰 차이가 나는 축이 있으면 sim_critic.quality 배점을 검토하라.')


if __name__ == '__main__':
    main()
