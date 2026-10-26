from collections import Counter
import copy
import importlib.util
from pathlib import Path
import unittest

from controller.control_core import to_action_text
from scripts.training.action_head import template_action

from scripts.training import audio_batch as module


class BalancedAudioBatchTests(unittest.TestCase):
    def fixture(self):
        rows,episodes=[],[]
        for context in range(16):
            for colors in (0,1):
                for tone in (0,1):
                    identifier=f'{context}-{colors}-{tone}'
                    label='A' if colors==tone else 'D'
                    rows.append({'sample_id':identifier+':0','episode_id':identifier,'frame_id':0,
                                 'game_id':'audio-cue-reach-v1','history':[],
                                 'rgb':{'pixel_sha256':f'{context}-{colors}'},
                                 'action':to_action_text(template_action(label))})
                    episodes.append({'episode_id':identifier,
                                     'cue_reset':{'variant':'reach','context':context,'colors':colors,'tone':tone}})
        return rows,episodes

    def test_full800_exposures_and_missing_assignments_are_identical(self):
        groups=module.group_training_rows(*self.fixture())
        exposures={}
        for arm in module.ARMS:
            seen=Counter()
            missing=Counter()
            mixed=0
            for step in range(200):
                batch,mask=module.batch_for_step(groups,arm=arm,step=step,seed=20261031)
                self.assertEqual(len(batch),4)
                self.assertEqual(Counter(module.target(row) for row in batch),{1:2,3:2})
                self.assertEqual(mask,step%8==7)
                seen.update(row['sample_id'] for row in batch)
                if mask:
                    missing.update(row['sample_id'] for row in batch)
                contexts={row['episode_id'].split('-')[0] for row in batch}
                mixed+=len(contexts)>1
                if arm=='paired-context' or mask:
                    self.assertEqual(len(contexts),1)
            self.assertEqual(sum(seen.values()),800)
            self.assertEqual(sum(missing.values()),100)
            self.assertEqual(len(seen),64)
            if arm=='balanced-unpaired':
                self.assertGreater(mixed,100)
            exposures[arm]=(seen,missing)
        self.assertEqual(exposures['paired-context'],exposures['balanced-unpaired'])

    def test_shared_block_multisets_and_epoch_coverage_include_final_half_epoch(self):
        groups=module.group_training_rows(*self.fixture())
        for start in range(0,200,8):
            exposure=[]
            for arm in module.ARMS:
                events=[]
                for step in range(start,start+8):
                    batch,mask=module.batch_for_step(groups,arm=arm,step=step,seed=20261031)
                    events.extend((row['sample_id'],mask) for row in batch)
                exposure.append(Counter(events))
            self.assertEqual(exposure[0],exposure[1])
        for epoch in range(12):
            for arm in module.ARMS:
                samples=[row['sample_id'] for step in range(epoch*16,(epoch+1)*16)
                         for row in module.batch_for_step(groups,arm=arm,step=step,seed=20261031)[0]]
                self.assertEqual(len(set(samples)),64)

    def test_resume_globalstep_schedule_and_input_order_are_deterministic(self):
        rows,episodes=self.fixture()
        groups=module.group_training_rows(rows,episodes)
        reverse=module.group_training_rows(list(reversed(rows)),list(reversed(episodes)))
        for arm in module.ARMS:
            for step in (0,2,3,4,7,8,15,16,198,199):
                self.assertEqual(module.batch_for_step(groups,arm=arm,step=step,seed=20261031),
                                 module.batch_for_step(reverse,arm=arm,step=step,seed=20261031))
        for kwargs in ({'arm':'wrong','step':0,'seed':1},{'arm':'paired-context','step':-1,'seed':1}):
            with self.assertRaises(ValueError):
                module.batch_for_step(groups,**kwargs)

    def test_corrupt_counterfactual_group_and_duplicate_rows_fail_closed(self):
        rows,episodes=self.fixture()
        bad=copy.deepcopy(rows)
        bad[1]['rgb']['pixel_sha256']='different'
        with self.assertRaisesRegex(ValueError,'RGB'):
            module.group_training_rows(bad,episodes)
        bad=copy.deepcopy(rows)
        bad[1]['action']=bad[0]['action']
        with self.assertRaises(ValueError):
            module.group_training_rows(bad,episodes)
        with self.assertRaises(ValueError):
            module.group_training_rows(rows+[rows[0]],episodes)


if __name__=='__main__':
    unittest.main()
