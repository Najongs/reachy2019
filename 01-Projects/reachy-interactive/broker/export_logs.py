"""Export conversation/motion logs into the repo for easy browsing.

Collects two sources into <repo>/logs/:

  logs/dgx/<date>-<id>/transcript.md (+ extracted jpg frames)
      - readable versions of the broker's claude CLI session transcripts
        (~/.claude/projects/...-newcode/*.jsonl), which otherwise are one
        JSON blob per line with base64 images inside
  logs/pi/<session>/summary.md
      - readable versions of any Pi TurnLogger sessions copied into logs/pi/
        (scp the Pi's ~/reachy_logs/* folders in, then rerun this)

Also writes logs/INDEX.md. Safe to rerun anytime; already-exported sessions
are refreshed in place. Run on the DGX:

    python3 export_logs.py
"""

import base64
import glob
import json
import os
import time


TRANSCRIPT_DIR = os.path.expanduser(
    '~/.claude/projects/-home-kiro-ai-NAJY-reachy-2019-newcode')
import sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))
import para  # PARA 기준 경로 (위치 계산은 para.py 한 곳에만)
REPO_LOGS = para.CONVERSATION_LOGS


def _read_events(path):
    events = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events


def _classify(events):
    """'motion' if the assistant speaks the motion JSON dialect, else 'chat'."""
    for e in events:
        if e.get('type') != 'assistant':
            continue
        for block in e.get('message', {}).get('content', []):
            if block.get('type') == 'text' and block['text'].lstrip().startswith('{"say"'):
                return 'motion'
    return 'chat'


def export_dgx_session(path, out_root):
    events = _read_events(path)
    session_id = os.path.splitext(os.path.basename(path))[0]
    stamp = time.strftime('%Y%m%d-%H%M', time.localtime(os.path.getmtime(path)))
    kind = _classify(events)

    out_dir = os.path.join(out_root, '{}-{}-{}'.format(stamp, kind, session_id[:8]))
    os.makedirs(out_dir, exist_ok=True)

    lines = ['# {} 세션 {} ({})'.format(kind, session_id[:8], stamp), '']
    turns = 0
    img_n = 0

    for e in events:
        role = e.get('type')
        if role not in ('user', 'assistant'):
            continue
        content = e.get('message', {}).get('content')
        ts = (e.get('timestamp') or '')[:19].replace('T', ' ')

        blocks = content if isinstance(content, list) else [{'type': 'text', 'text': content or ''}]
        for block in blocks:
            if block.get('type') == 'text' and block.get('text', '').strip():
                who = '사용자' if role == 'user' else '리치'
                lines.append('**{}** ({}): {}'.format(who, ts, block['text'].strip()))
                lines.append('')
                turns += 1
            elif block.get('type') == 'image':
                img_n += 1
                name = 'img_{:03d}.jpg'.format(img_n)
                with open(os.path.join(out_dir, name), 'wb') as f:
                    f.write(base64.b64decode(block['source']['data']))
                lines.append('![카메라 프레임]({})'.format(name))
                lines.append('')

    with open(os.path.join(out_dir, 'transcript.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    first = next((l for l in lines if l.startswith('**사용자**')), '')
    return {'dir': os.path.basename(out_dir), 'kind': kind, 'turns': turns,
            'images': img_n, 'first': first[:110]}


def export_pi_session(session_dir):
    events_path = os.path.join(session_dir, 'events.jsonl')
    if not os.path.exists(events_path):
        return None

    lines = ['# Pi 세션 {}'.format(os.path.basename(session_dir)), '']
    counts = {}

    with open(events_path, encoding='utf-8') as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            kind = e.get('kind', '?')
            counts[kind] = counts.get(kind, 0) + 1

            lines.append('## [{}] {} — {}'.format(e.get('ts', ''), kind,
                                                  e.get('text', '')))
            if e.get('reply'):
                lines.append('- 응답: {}'.format(e['reply']))
            if kind == 'motion':
                motion = e.get('motion') or {}
                lines.append('- 결과: {} / preset={} / grasp={}'.format(
                    e.get('outcome'), motion.get('preset'), e.get('grasp_ok')))
                if e.get('reason'):
                    lines.append('- 사유: {}'.format(e['reason']))
                if motion.get('moves'):
                    lines.append('- 생성 키프레임 {}개'.format(len(motion['moves'])))
            if e.get('frame'):
                lines.append('![frame]({})'.format(e['frame']))
            lines.append('')

    with open(os.path.join(session_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return counts


def main():
    out_root = os.path.abspath(REPO_LOGS)
    dgx_root = os.path.join(out_root, 'dgx')
    pi_root = os.path.join(out_root, 'pi')
    os.makedirs(dgx_root, exist_ok=True)
    os.makedirs(pi_root, exist_ok=True)

    # Keep the whole logs tree out of git.
    with open(os.path.join(out_root, '.gitignore'), 'w') as f:
        f.write('*\n')

    rows = []
    for path in sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, '*.jsonl')),
                       key=os.path.getmtime, reverse=True):
        try:
            rows.append(export_dgx_session(path, dgx_root))
        except Exception as e:
            print('skip {}: {}'.format(os.path.basename(path), e))

    pi_rows = []
    for session_dir in sorted(glob.glob(os.path.join(pi_root, '*'))):
        if os.path.isdir(session_dir):
            counts = export_pi_session(session_dir)
            if counts:
                pi_rows.append((os.path.basename(session_dir), counts))

    index = ['# Reachy 로그 인덱스 (생성: {})'.format(time.strftime('%Y-%m-%d %H:%M')), '',
             '## DGX 세션 전사 ({}개, 최신순)'.format(len(rows)), '']
    for r in rows:
        index.append('- [{dir}]({p}/transcript.md) — {kind}, 턴 {turns}, 이미지 {images}'.format(
            p='dgx/' + r['dir'], **r))
        if r['first']:
            index.append('  - {}'.format(r['first']))
    index += ['', '## Pi 세션 (로봇 실행 로그, {}개)'.format(len(pi_rows)), '']
    if pi_rows:
        for name, counts in pi_rows:
            index.append('- [pi/{0}/summary.md](pi/{0}/summary.md) — {1}'.format(
                name, ', '.join('{} {}'.format(k, v) for k, v in sorted(counts.items()))))
    else:
        index.append('(아직 없음 — Pi의 ~/reachy_logs/* 폴더들을 logs/pi/ 로 복사한 뒤 재실행)')

    with open(os.path.join(out_root, 'INDEX.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(index))

    print('DGX 세션 {}개, Pi 세션 {}개 → {}'.format(len(rows), len(pi_rows), out_root))
    print('인덱스: {}'.format(os.path.join(out_root, 'INDEX.md')))


if __name__ == '__main__':
    main()
