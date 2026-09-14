import os
import sys
import unittest


os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'sim'), os.path.join(ROOT, 'robot')]

import sim_tasks
import sim_agent
import sim_critic


class SimulationBaselineTests(unittest.TestCase):
    def test_easy_oracle_curriculum_is_feasible(self):
        tasks = sim_tasks.generate_sweet(
            4, seed=1309, kinds=('pick', 'lift'))
        self.assertEqual([t['scene'][0]['kind'] for t in tasks], ['cup'] * 4)
        self.assertTrue(all(sim_tasks.feasible(t) for t in tasks))

    def test_grasp_target_contact_is_not_a_collision(self):
        stage, why = sim_critic.stage_verdict(
            path_tab=0.0, path_obj=0.0, path_tgt=-0.8,
            reached=True, quality_score=60, natural_min=55)
        self.assertEqual(stage, 'success')
        self.assertIsNone(why)

        stage, _ = sim_critic.stage_verdict(
            path_tab=0.0, path_obj=0.0, path_tgt=-1.1,
            reached=True, quality_score=60, natural_min=55)
        self.assertEqual(stage, 'collision')

    def test_round_objects_use_tighter_lower_bias_grasp_policy(self):
        round_policy = sim_agent._grasp_policy('apple')
        upright_policy = sim_agent._grasp_policy('cup')
        self.assertLess(round_policy['align_cm'], upright_policy['align_cm'])
        self.assertLess(round_policy['bias_scale'],
                        upright_policy['bias_scale'])
        self.assertGreaterEqual(round_policy['regrasp_attempts'], 2)


if __name__ == '__main__':
    unittest.main()
