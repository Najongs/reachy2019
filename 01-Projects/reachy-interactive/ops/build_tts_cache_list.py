"""로그에서 자주 나온 답변을 뽑아 미리 합성할 목록을 만든다.

로봇이 실제로 한 말 중 되풀이되는 것들(인사, 단골 질문의 답)은 미리 합성해 두면
두 가지가 좋아진다. 복도 와이파이가 끊겨도 자연스러운 목소리로 나오고, 합성을
기다리지 않아 바로 말한다(실측: 캐시된 문구는 0.004초 만에 재생 시작).

여기서 만든 목록(config/cached_lines.json)을 deploy.sh 가 Pi 로 보내고,
voice_chat 이 기동할 때 미리 합성한다.

한 번만 나온 답변은 넣지 않는다. LLM 이 그때그때 만든 문장은 다시 나올 일이
거의 없어서, 캐시에 넣어 봐야 자리만 차지한다.

사용:
    python3 ops/build_tts_cache_list.py                 # 목록 보기
    python3 ops/build_tts_cache_list.py --write         # config/cached_lines.json 갱신
    python3 ops/build_tts_cache_list.py --min-count 3   # 3번 이상 나온 것만
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

DEFAULT_LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', '..', '04-Archives',
                            'conversation-logs', 'pi')
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'config', 'cached_lines.json')

# 미리 합성해 둘 값이 없는 답변들.
#   - 너무 길면 캐시가 커지기만 하고, 그렇게 긴 말이 똑같이 반복되지도 않는다.
#   - 시각/온도처럼 그때그때 값이 달라지는 문장은 다시 쓰이지 않는다.
MAX_LEN = 120
MIN_LEN = 2          # "네?" 같은 짧은 맞장구도 캐시할 값이 있다
# 값이 그때그때 달라지는 문장만 거른다. '지금은' 같은 흔한 말까지 넣었더니
# "지금은 생각이 잘 안 나네요." 처럼 시각과 무관한 답변이 걸렸다.
VOLATILE = re.compile(
    r'\d\s*(시|분|초|도|℃|퍼센트|%)'      # "세 시 십오 분", "26도"
    r'|현재\s*시각|오전\s*\d|오후\s*\d')


def collect(log_root):
    """이벤트 로그에서 (답변 -> 나온 횟수) 를 센다."""
    counts = Counter()
    pattern = os.path.join(log_root, '*', 'events.jsonl')
    for path in glob.glob(pattern):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get('kind') not in ('chat', 'note', 'vision_chat'):
                    continue
                reply = (d.get('reply') or '').strip()
                if reply:
                    counts[reply] += 1
    return counts


def worth_caching(text):
    """미리 합성해 둘 만한 답변인가."""
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    if VOLATILE.search(text):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', default=DEFAULT_LOGS, help='이벤트 로그 폴더')
    ap.add_argument('--out', default=DEFAULT_OUT, help='만들 목록 파일')
    ap.add_argument('--min-count', type=int, default=2,
                    help='이 횟수 이상 나온 답변만 (기본 2)')
    ap.add_argument('--max-lines', type=int, default=60,
                    help='최대 몇 개까지 (기본 60)')
    ap.add_argument('--write', action='store_true', help='실제로 파일에 쓴다')
    args = ap.parse_args()

    counts = collect(os.path.abspath(args.logs))
    if not counts:
        print('로그에서 답변을 찾지 못했습니다: %s' % args.logs)
        print('먼저 가져오세요:  bash ops/daily_update.sh')
        return 1

    picked = [(t, n) for t, n in counts.most_common()
              if n >= args.min_count and worth_caching(t)][:args.max_lines]

    print('답변 %d종류 중 미리 합성할 것 %d개 (%d번 이상 나온 것)'
          % (len(counts), len(picked), args.min_count))
    for text, n in picked:
        print('  %2dx  %s' % (n, text[:70]))

    dropped = [t for t, n in counts.most_common()
               if n >= args.min_count and not worth_caching(t)]
    if dropped:
        print('\n반복되지만 캐시하지 않은 것 %d개 (너무 길거나 값이 바뀌는 문장):'
              % len(dropped))
        for t in dropped[:5]:
            print('   %s' % t[:70])

    if not args.write:
        print('\n실제로 반영하려면 --write 를 주세요. 그 다음 배포: bash ops/deploy.sh')
        return 0

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({
            '_comment': ('로그에서 자주 나온 답변. voice_chat 이 기동할 때 미리 '
                         '합성해 둔다(인터넷이 끊겨도 자연스러운 목소리로 나오고, '
                         '합성 대기가 없다). ops/build_tts_cache_list.py 가 만든다.'),
            'lines': [t for t, _ in picked],
        }, fh, ensure_ascii=False, indent=2)
    print('\n기록: %s (%d개)' % (out, len(picked)))
    print('배포: bash ops/deploy.sh')
    return 0


if __name__ == '__main__':
    sys.exit(main())
