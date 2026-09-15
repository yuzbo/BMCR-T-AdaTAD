import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.fidelity_select import select_best
import tools.fidelity_dispatch as dispatch


class PeakSelectionTests(unittest.TestCase):
    def test_peak_and_terminal_are_separate_and_tie_is_earlier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for epoch,score in ((25,.64),(30,.66),(35,.66),(60,.65)):
                folder=root/f's_h65_test_epoch_{epoch:02}'; folder.mkdir()
                (folder/'metrics.json').write_text(json.dumps(dict(test_videos=211,test_windows=792,
                    initialization=dict(total_epochs=epoch,fidelity_revision='lr_identity_crop_validity_v1',
                                        checkpoint=f'epoch{epoch}.pth',state_key='state_dict_ema'),
                    metrics=dict(average_mAP=score))))
            result=select_best(root,'s')
            self.assertEqual(result['best']['total_epochs'],30)
            self.assertEqual(result['terminal']['total_epochs'],60)
            self.assertFalse(result['all_candidates_evaluated'])
            self.assertTrue(result['test_based_selection'])

    def run_scheduler(self, total_jobs=2, pending_test=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); runs=root/'runs'; runs.mkdir()
            table=dispatch.stages()
            for b in ('s','b'):
                for name,updates in ((f'preflight_{b}',2),(f'{b}_warm',2000)):
                    file=runs/table[name]['done']; file.parent.mkdir(parents=True)
                    file.write_text(json.dumps(dict(fidelity_revision='lr_identity_crop_validity_v1',successful_updates=updates)))
            table['s_h65']['job_id']=101; table['b_h65']['job_id']=102
            checkpoint=runs/'s_h65/epoch_05.pth'; checkpoint.parent.mkdir(); checkpoint.write_bytes(b'ready')
            rows=['101|RUNNING','102|RUNNING']
            if pending_test:
                table['b_test_25']['job_id']=103; rows.append('103|PENDING')
            while len(rows)<total_jobs: rows.append(f'{1000+len(rows)}|RUNNING')
            calls=[]
            def command(*args):
                calls.append(args)
                if args[0]=='squeue': return SimpleNamespace(returncode=0,stdout='\n'.join(rows),stderr='')
                if args[0]=='sbatch': return SimpleNamespace(returncode=0,stdout='700\n',stderr='')
                if args[:3]==('scontrol','show','partition'):
                    return SimpleNamespace(returncode=0,stdout='PartitionName=gpu Nodes=g[0014,0056,9999]',stderr='')
                if args[:3]==('scontrol','show','hostnames'):
                    return SimpleNamespace(returncode=0,stdout='g0014\ng0056\ng9999\n',stderr='')
                raise AssertionError(args)
            state=dict(stages=table)
            with patch.object(dispatch,'RUNS',runs),patch.object(dispatch,'STATE',root/'deployment.json'),\
                 patch.object(dispatch,'EXP',root),patch.object(dispatch,'command',command),\
                 patch.dict('os.environ',{'USER':'unit_test'}):
                dispatch.tick(state)
            return [c for c in calls if c[0]=='sbatch'],state

    def test_two_training_jobs_do_not_block_ready_milestone_test(self):
        calls,state=self.run_scheduler()
        self.assertEqual(len(calls),1)
        self.assertIn('--job-name=h65-fix-s_test_25',calls[0])
        self.assertEqual(state['stages']['s_test_25']['job_id'],700)
        self.assertIn('--exclude=g9999',calls[0])
        self.assertFalse(any(arg.startswith('--nodelist=') for arg in calls[0]))
        self.assertEqual(state['stages']['s_test_25']['eligible_nodes'],['g0014','g0056'])

    def test_missing_verified_node_inventory_prevents_submission(self):
        def response(text='',code=0):
            return SimpleNamespace(returncode=code,stdout=text,stderr='')
        for results in ([response(code=1)], [response()],
                        [response('Nodes=g9999'),response('g9999\n')],
                        [response('Nodes=g0014'),response(code=1)]):
            with patch.object(dispatch,'command',side_effect=results):
                self.assertIsNone(dispatch.nodes_for_submission())

    def test_account_cap_and_pending_test_slot_are_respected(self):
        self.assertEqual(self.run_scheduler(total_jobs=16)[0],[])
        self.assertEqual(self.run_scheduler(pending_test=True)[0],[])


if __name__=='__main__': unittest.main()
