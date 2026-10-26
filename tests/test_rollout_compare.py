import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.training import rollout_compare as module


class RetentionComparisonTests(unittest.TestCase):
    def test_complete_actual_cpu_fixture_audits_then_rejects_summary_and_schedule_tampering(self):
        from scripts.training.action_head import choose_class,template_action
        from controller.control_core import to_action_text
        from scripts.training.data import sha256_file
        from scripts.training.feedback import audit_feedback_episodes
        from scripts.training.rollout import run_episode,summarize_episodes
        # Tiny constant CPU policy, not a GameScaling execution/quality claim.
        label,details=choose_class([1.,0.,0.,0.,0.])
        def policy(row):
            return {'completion':to_action_text(template_action(label)),'generated_tokens':0,'timing_ms':{},
                    'decoder_record':{**details,'selected_class':label,'predicted_tokens':1}}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            episode=run_episode(policy,root,variant='reach',seed=30000,max_steps=1)
            (root/'episodes.jsonl').write_text(json.dumps(episode)+'\n')
            config={'arguments':{'decoder':'first-key','games':['reach'],'seed_start':30000,
                                 'episodes_per_game':1,'max_steps':1},
                    'generation':{'decoder':'first-key','effective_predicted_tokens':1,'do_sample':False},
                    'teacher_or_reward_inputs':False,'learning_during_rollout':False,
                    'source_code_sha256':{name:sha256_file(path) for name,path in module.source_paths().items()}}
            (root/'config.json').write_text(json.dumps(config))
            status={'state':'completed','failed_episodes':0,'active_episode':None,
                    'planned_episodes':1,'completed_episodes':1}
            (root/'run-status.json').write_text(json.dumps(status))
            summary={'episodes_sha256':sha256_file(root/'episodes.jsonl'),
                     'feedback_audit':audit_feedback_episodes([episode],root),'results':summarize_episodes([episode])}
            (root/'summary.json').write_text(json.dumps(summary))
            self.assertEqual(len(module.read_run(root)[1]),1)
            changed=copy.deepcopy(episode)
            changed['system_prompt']+=' Oracle: choose W.'
            (root/'episodes.jsonl').write_text(json.dumps(changed)+'\n')
            resigned={**summary,'episodes_sha256':sha256_file(root/'episodes.jsonl')}
            (root/'summary.json').write_text(json.dumps(resigned))
            with self.assertRaisesRegex(ValueError,'player prompt'):
                module.read_run(root)
            changed=copy.deepcopy(episode)
            changed['frames'][0]['timing_ms']['observation_to_action_start']+=5.
            (root/'episodes.jsonl').write_text(json.dumps(changed)+'\n')
            resigned={**summary,'episodes_sha256':sha256_file(root/'episodes.jsonl'),
                      'results':summarize_episodes([changed])}
            (root/'summary.json').write_text(json.dumps(resigned))
            with self.assertRaisesRegex(ValueError,'timestamp'):
                module.read_run(root)
            (root/'episodes.jsonl').write_text(json.dumps(episode)+'\n')
            (root/'summary.json').write_text(json.dumps(summary))
            altered=copy.deepcopy(summary)
            altered['results']['reach']['mean_return']=0.
            (root/'summary.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'metrics'):
                module.read_run(root)
            (root/'summary.json').write_text(json.dumps(summary))
            config['arguments']['seed_start']=30001
            (root/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'reset schedule'):
                module.read_run(root)

    def test_retention_counts_improvement_and_regression_even_when_mean_ties(self):
        before=[{'episode_id':'one','frames':[{'reward':.99,'terminal_reason':'goal'}]},
                {'episode_id':'two','frames':[{'reward':-.01,'terminal_reason':'horizon'}]}]
        after=[{'episode_id':'one','frames':[{'reward':-.01,'terminal_reason':'horizon'}]},
               {'episode_id':'two','frames':[{'reward':.99,'terminal_reason':'goal'}]}]
        report=module.paired_outcomes(before,after)
        self.assertEqual(report['return_wins_ties_losses'],[1,0,1])
        self.assertEqual(report['original_success_to_failure'],['one'])
        self.assertEqual(report['original_failure_to_success'],['two'])
        self.assertEqual(report['paired_mean_return_delta'],0.)

    def test_missing_pair_and_reordered_episodes_rejected(self):
        before=[{'episode_id':'a','frames':[{'reward':0.,'terminal_reason':'horizon'}]}]
        with self.assertRaises(ValueError):
            module.paired_outcomes(before,[])
        with self.assertRaises(ValueError):
            module.paired_outcomes(before,[{**before[0],'episode_id':'b'}])

    def test_logit_template_actual_action_binding_rejects_changed_decision(self):
        from scripts.training.action_head import choose_class,template_action
        from controller.control_core import to_action_text
        label,record=choose_class([1.,0.,0.,0.,0.])
        action=to_action_text(template_action(label))
        frame={'policy':{'strict_valid':True,'filtered':False,'fallback':False,'generated_tokens':0,
            'decoder_record':{**record,'selected_class':label,'predicted_tokens':1}},
            'proposed_action':action,'filtered_action':action,'actual_action':{'text':action}}
        module.audit_first_key(frame)
        rounded=copy.deepcopy(frame)
        import math
        rounded['policy']['decoder_record']['restricted_softmax_not_calibrated']['W']=math.nextafter(
            rounded['policy']['decoder_record']['restricted_softmax_not_calibrated']['W'],1.)
        module.audit_first_key(rounded)
        changed=copy.deepcopy(frame)
        changed['policy']['decoder_record']['raw_class_logits']['A']=2.
        with self.assertRaisesRegex(ValueError,'first-key'):
            module.audit_first_key(changed)


if __name__=='__main__':
    unittest.main()
