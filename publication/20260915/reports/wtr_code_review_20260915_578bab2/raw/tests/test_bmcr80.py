import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.full.course import BMCR80_RECIPE,COMPONENT_ORDER,warm_origin,validate_scales,validate_resume
from tools import bmcr80_dispatch as dispatch
from tools.bmcr80_report import EPOCHS,select_best


def warm_payload(backbone='s'):
    return dict(completed_epochs=20,successful_updates=2000,state_dict_ema={},metadata=dict(
        backbone=backbone,variant='h65',phase='warm',train_videos=200,updates_per_epoch=100,
        phase_epochs=20,full_training=True,terminal_state='state_dict_ema',
        fidelity_revision='lr_identity_crop_validity_v1',initialization='recognition_k400'))


class BMCR80Test(unittest.TestCase):
    def test_warm_and_scale_provenance(self):
        payload=warm_payload();origin=warm_origin(payload,ROOT/'warm.pth','s')
        audit=dict(recipe=BMCR80_RECIPE,warm_origin=origin,component_order=COMPONENT_ORDER,scales=[.001,.002])
        self.assertEqual(validate_scales(audit,origin),[.001,.002])
        wrong=copy.deepcopy(audit);wrong['warm_origin']['backbone']='b'
        with self.assertRaises(ValueError):validate_scales(wrong,origin)
        wrong=copy.deepcopy(payload);wrong['successful_updates']=1900
        with self.assertRaises(ValueError):warm_origin(wrong,ROOT/'warm.pth','s')

    def test_reject_old_course_resume(self):
        origin=warm_origin(warm_payload(),ROOT/'warm.pth','s')
        metadata=dict(fidelity_revision='lr_identity_crop_validity_v1',backbone='s',phase='joint',variant='bmcr',
                      phase_epochs=60,recipe=BMCR80_RECIPE,course_total_epochs=80,warm_origin=origin)
        validate_resume(metadata,80,'joint','bmcr','s',origin)
        for key,value in [('phase_epochs',40),('course_total_epochs',60),('recipe','old')]:
            wrong={**metadata,key:value}
            with self.assertRaises(ValueError):validate_resume(wrong,80,'joint','bmcr','s',origin)

    def test_real_joint_initialization_function(self):
        import torch
        from tools import full_train
        class Tiny(torch.nn.Module):
            variant='bmcr'
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2));self.register_buffer('utility_scales',torch.ones(2))
        with tempfile.TemporaryDirectory() as temp:
            runs=Path(temp);(runs/'s_warm').mkdir();(runs/'s_audit').mkdir()
            parent=runs/'s_warm/terminal.pth';payload=warm_payload()
            payload['state_dict_ema']={'weight':torch.tensor([2.,3.]),'utility_scales':torch.ones(2)};torch.save(payload,parent)
            origin=warm_origin(payload,parent,'s');audit=dict(recipe=BMCR80_RECIPE,warm_origin=origin,
                                component_order=COMPONENT_ORDER,scales=[.002,.003])
            (runs/'s_audit/scales.json').write_text(json.dumps(audit));model=Tiny()
            with patch.object(full_train,'RUNS',runs):got,_=full_train.initialize_joint(model,'s',80)
            self.assertEqual(got,origin);self.assertTrue(torch.equal(model.weight,torch.tensor([2.,3.])))
            self.assertTrue(torch.allclose(model.utility_scales,torch.tensor([.002,.003])))

    def test_actual_scheduler_and_resume(self):
        import torch
        spec=importlib.util.spec_from_file_location('real_schedule',ROOT/'upstream/opentad/cores/scheduler.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def make():
            parameter=torch.nn.Parameter(torch.zeros(1));optim=torch.optim.SGD([parameter],lr=1e-4)
            sched,_=module.build_scheduler(dict(type='LinearWarmupCosineAnnealingLR',warmup_epoch=3,max_epoch=60),optim,100)
            return optim,sched
        optim,sched=make()
        for _ in range(4000):optim.step();sched.step()
        self.assertGreater(optim.param_groups[0]['lr'],1e-8)
        saved_optim=copy.deepcopy(optim.state_dict());saved_sched=copy.deepcopy(sched.state_dict())
        other,restored=make();other.load_state_dict(saved_optim);restored.load_state_dict(saved_sched)
        for _ in range(2000):
            optim.step();sched.step();other.step();restored.step()
        self.assertAlmostEqual(optim.param_groups[0]['lr'],1e-8,places=12)
        self.assertEqual(optim.param_groups[0]['lr'],other.param_groups[0]['lr'])

    def test_stages_and_dependencies(self):
        table=dispatch.stages();self.assertEqual(len(table),36)
        self.assertEqual(sum(v['kind']=='test' for v in table.values()),24)
        self.assertEqual(EPOCHS,tuple(range(25,81,5)))
        self.assertEqual(table['s_train']['dependencies'],['s_preflight','b_preflight'])
        self.assertEqual(table['b_test_80']['requires'],['b_bmcr/epoch_60.pth'])
        self.assertFalse(any(v['args'] and ('--phase' in v['args'] and v['args'][v['args'].index('--phase')+1]=='warm') for v in table.values()))

    def test_peak_tie_and_separate_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            runs=Path(temp)
            for e in EPOCHS:
                folder=runs/f's_bmcr_test_epoch_{e:02}';folder.mkdir()
                score=.7 if e in (55,60) else .6
                metrics={'average_mAP':score,**{f'mAP@0.{i}':score for i in range(3,8)}}
                data=dict(test_videos=211,test_windows=792,metrics=metrics,
                      initialization=dict(recipe=BMCR80_RECIPE,total_epochs=e,checkpoint=f'epoch_{e-20}.pth',state_key='state_dict_ema'))
                (folder/'metrics.json').write_text(json.dumps(data))
            selected=select_best(runs,'s');self.assertTrue(selected['all_candidates_evaluated'])
            self.assertEqual(selected['best']['total_epochs'],55)
            self.assertEqual(selected['at60']['total_epochs'],60);self.assertEqual(selected['at80']['total_epochs'],80)

    def test_two_training_jobs_still_allow_one_test(self):
        with tempfile.TemporaryDirectory() as temp:
            exp=Path(temp);runs=exp/'runs';(runs/'s_bmcr').mkdir(parents=True);(runs/'s_bmcr/epoch_05.pth').touch()
            table=dispatch.stages()
            for v in table.values():v['status']='WAITING'
            for b,j in [('s',101),('b',102)]:
                table[f'{b}_audit']['status']='COMPLETED';table[f'{b}_preflight']['status']='COMPLETED'
                table[f'{b}_train'].update(status='RUNNING',job_id=j)
            calls=[]
            def command(*args):
                calls.append(args)
                return SimpleNamespace(returncode=0,stdout='101|RUNNING\n102|PENDING\n' if args[0]=='squeue' else '103\n',stderr='')
            state=dict(recipe=BMCR80_RECIPE,stages=table)
            with patch.object(dispatch,'RUNS',runs),patch.object(dispatch,'EXP',exp),patch.object(dispatch,'save'),\
                 patch.object(dispatch,'command',side_effect=command),patch.object(dispatch,'nodes_for_submission',return_value=(['g0014'],[])),\
                 patch.object(dispatch,'artifact_complete',side_effect=lambda n,s,t:s['status']=='COMPLETED'),patch.object(dispatch,'build_report'):
                dispatch.tick(state)
            submitted=[x for x in calls if x[0]=='sbatch'];self.assertEqual(len(submitted),1)
            self.assertIn('--job-name=bmcr80-s_test_25',submitted[0]);self.assertEqual(table['s_test_25']['job_id'],103)

    def test_cancellation_never_queries_or_submits(self):
        with patch.object(dispatch,'command',side_effect=AssertionError('must not run')):
            self.assertTrue(dispatch.tick(dict(cancelled_by_user=True)))


if __name__=='__main__':unittest.main()
