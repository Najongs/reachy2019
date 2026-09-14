"""PARA 구조의 기준 경로 - 데이터를 읽고 쓰는 코드는 전부 여기서 얻는다.

저장소는 PARA 로 나뉜다 (reachy-2019/ 루트):
    01-Projects/   진행 중인 프로젝트 (이 코드가 사는 곳)
    02-Areas/      꾸준히 돌보는 영역 (robot-ops)
    03-Resources/  참고 자원 (SDK, 3D 모델)
    04-Archives/   보관고 - 대화 로그, 사람 데이터셋, 시뮬 런 산출물

왜 이 파일이 있나: 스크립트가 각자 '..' 개수로 루트를 세다가 폴더를
한 번 옮기자 깊이가 어긋나, 가짜 보관고(01-Projects/04-Archives)에
사람 DB 를 며칠간 써 버렸다. 위치 계산은 한 곳에만 둔다.

쓰는 법 (어느 깊이의 스크립트든 동일):
    sys.path.insert(0, <reachy-interactive 까지 올라간 경로>)
    import para
    para.PI_LOGS  # 등
"""
import os

# 이 파일은 반드시 01-Projects/reachy-interactive/ 바로 밑에 있어야 한다.
PROJECT = os.path.dirname(os.path.abspath(__file__))     # reachy-interactive/
ROOT = os.path.dirname(os.path.dirname(PROJECT))         # reachy-2019/

PROJECTS = os.path.join(ROOT, '01-Projects')
AREAS = os.path.join(ROOT, '02-Areas')
RESOURCES = os.path.join(ROOT, '03-Resources')
ARCHIVES = os.path.join(ROOT, '04-Archives')

# 보관고 3층 구조 (사용자 설계 2026-09-14)
#   RAW      원시 축적 - 로그·사진·런 산출물이 그대로 쌓이는 곳
#   DIGESTS  주기 요약 - 원시를 구간별로 접어 '지식화' 한 중간층
#   KNOWLEDGE 메인 지식 - 요약들을 종합한 사실 원장 + 사람이 읽는 한 장
# 읽는 쪽은 보통 메인 지식만 본다. 원시를 다시 여는 건 재집계뿐이고,
# 그마저 커서로 새 줄만 읽는다 (매번 전체 로그를 열지 않기 위해).
RAW = os.path.join(ARCHIVES, 'raw')
CONVERSATION_LOGS = os.path.join(RAW, 'conversation-logs')
PI_LOGS = os.path.join(CONVERSATION_LOGS, 'pi')
PERSON_DATASET = os.path.join(RAW, 'person-dataset')
PERSONS = os.path.join(PERSON_DATASET, 'persons')        # 원본 + persons.db
PERSON_EXPORT = os.path.join(PERSON_DATASET, 'dataset')  # 걸러 뽑은 것
SIM_RUNS = os.path.join(RAW, 'sim-runs')                 # 시뮬 런 보관
OPS_LOGS = os.path.join(RAW, 'ops-logs')                 # 브로커·워치독 로그
REFERENCE = os.path.join(ARCHIVES, 'reference')          # 참고 데모 소스
KNOWLEDGE = os.path.join(ARCHIVES, 'knowledge')          # 메인 지식
DIGESTS = os.path.join(KNOWLEDGE, 'digests')             # 주기 요약(중간층)

# 어긋난 배치를 이르게 잡는 안전핀
assert os.path.basename(ROOT) != '01-Projects', \
    'para.py 가 잘못된 깊이에 있다 - reachy-interactive/ 바로 밑이어야 함'
