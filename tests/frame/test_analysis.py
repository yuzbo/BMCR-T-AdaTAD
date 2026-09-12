import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from tools.frame_analyze import analyze, load_records, pareto

def test_manifest_and_pareto(tmp_path):
    rows=[{"id":"s0","status":"complete","backbone":"s","metrics":{"average_mAP":.4},"gflops":10,"latency_ms":2},
          {"id":"s1","status":"complete","backbone":"s","metrics":{"average_mAP":.3},"gflops":20}]
    p=tmp_path/'m.json'; p.write_text(json.dumps({'records':rows}))
    assert len(load_records(p))==2
    assert [r['id'] for r in pareto(rows)]==['s0']
    c=analyze(p,tmp_path/'out')
    assert c['figures']['map_gflops_pareto']['status']=='available'
    assert c['figures']['latency_cohort']['points']==1
    assert (tmp_path/'out'/'source_manifest.json').exists()

def test_missing_fields_are_reported(tmp_path):
    p=tmp_path/'m.json'; p.write_text(json.dumps({'records':[{'id':'x','metrics':{}}]}))
    c=analyze(p,tmp_path/'out',dry_run=True)
    assert c['figures']['map_gflops_pareto']['status']=='missing'
    assert c['figures']['five_threshold_map']['status']=='missing'

def test_backbones_and_incomplete_not_compared():
    rows=[dict(id='s',backbone='s',status='complete',gflops=10,metrics={'average_mAP':.8}),
          dict(id='b',backbone='b',status='complete',gflops=20,metrics={'average_mAP':.7}),
          dict(id='future',backbone='b',status='running',gflops=1,metrics={'average_mAP':1.})]
    assert {r['id'] for r in pareto(rows)}=={'s','b'}
    assert {r['id'] for r in pareto(rows,global_backbones=True)}=={'s'}
