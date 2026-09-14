"""저장소 위생 가드 - 가짜 중첩 보관고 재발 감지.

상대 깊이('..' 개수 / parents[N]) 로 경로를 세던 코드가 폴더 재배치 뒤
01-Projects 안에 가짜 04-Archives 를 만들어 사람 DB 를 며칠 오염시킨
사고가 두 번 있었다 (2026-09-08, 09-14 재발). 데이터 경로는 para.py
한 곳에서만 얻는 것이 규약이고, 이 테스트가 그 규약의 감시자다.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import para


class RepoHygieneTests(unittest.TestCase):
    def test_no_nested_fake_archives(self):
        """01-Projects 어디에도 04-Archives 가 있으면 안 된다."""
        projects = para.PROJECTS
        offenders = []
        for dirpath, dirnames, _ in os.walk(projects):
            if '04-Archives' in dirnames:
                offenders.append(os.path.join(dirpath, '04-Archives'))
            # 무거운 폴더는 내려가지 않는다
            dirnames[:] = [d for d in dirnames
                           if d not in ('.git', 'sim_data', 'meshes',
                                        '__pycache__', 'node_modules')]
        self.assertEqual(offenders, [],
                         '가짜 중첩 보관고 발견 - 어떤 코드가 상대 경로를 '
                         '세고 있다. para.py 를 쓰게 고칠 것: %s' % offenders)

    def test_para_points_inside_repo_root(self):
        self.assertTrue(os.path.isdir(para.ARCHIVES))
        self.assertEqual(os.path.basename(para.ROOT), 'reachy-2019')
        self.assertFalse(para.ARCHIVES.startswith(para.PROJECTS))

    def test_knowledge_ledger_uses_para(self):
        sys.path.insert(0, os.path.join(ROOT, 'ops', 'data'))
        import knowledge_ledger as K
        self.assertEqual(str(K.KNOWLEDGE_DIR), para.KNOWLEDGE)
        self.assertEqual(str(K.PI_LOGS), para.PI_LOGS)
