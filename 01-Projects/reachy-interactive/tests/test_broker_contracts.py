import json
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'broker'), os.path.join(ROOT, 'robot'),
                os.path.join(ROOT, 'sim')]

import llm_broker as broker
import sim_agent
import sim_orchestra


class _StructuredBackend(object):
    def __init__(self, result):
        self.result = result
        self.schema = None

    def reply_structured(self, text, session, image, schema):
        self.schema = schema
        return json.dumps(self.result)


class _FailedGroundingClient(object):
    last_error = 'quota exhausted'

    def ask_grounding(self, text, image=None):
        return None


class _PerceptionStub(object):
    @staticmethod
    def overlay(world, dets):
        return np.zeros((16, 16, 3), dtype=np.uint8)


class BrokerContractTests(unittest.TestCase):
    def test_structured_reply_uses_requested_schema(self):
        backend = _StructuredBackend({'winner': 'A', 'why': 'smoother'})
        out = broker.structured_reply(
            backend, 'compare', 'test', None, broker.EVAL_SCHEMAS['duel'])
        self.assertEqual(out['winner'], 'A')
        self.assertIs(backend.schema, broker.EVAL_SCHEMAS['duel'])

    def test_invalid_json_is_rejected(self):
        backend = _StructuredBackend({'ok': True})
        backend.reply_structured = lambda *args, **kwargs: 'not json'
        with self.assertRaises(RuntimeError):
            broker.structured_reply(
                backend, 'probe', 'test', None,
                broker.EVAL_SCHEMAS['health'])

    def test_json_that_violates_schema_is_rejected(self):
        backend = _StructuredBackend({'winner': 'C', 'why': 'guess'})
        with self.assertRaises(RuntimeError):
            broker.structured_reply(
                backend, 'compare', 'test', None,
                broker.EVAL_SCHEMAS['duel'])

    def test_grounding_failure_is_infrastructure_error(self):
        planner = sim_agent.BrokerPlanner.__new__(sim_agent.BrokerPlanner)
        planner.client = _FailedGroundingClient()
        planner.lessons = []
        planner.P = _PerceptionStub()
        result = planner.plan(object(), {'instruction': '컵을 들어 올려'},
                              {'dets': [], 'memory': None})
        self.assertTrue(result['infra_error'])
        self.assertIn('quota exhausted', result['why'])

    def test_audit_flags_backend_failure(self):
        records = [{
            'task': {'id': 't0', 'kind': 'lift'},
            'verdict': {'success': False, 'stage': 'infra_error'},
            'perception': {'seen': True},
        }]
        flags = sim_orchestra.audit(records)
        self.assertTrue(any('계획 백엔드 오류' in f for f in flags))


if __name__ == '__main__':
    unittest.main()
