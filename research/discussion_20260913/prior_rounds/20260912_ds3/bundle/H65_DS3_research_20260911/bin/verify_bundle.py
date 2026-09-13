#!/usr/bin/env python3
"""Validate generated artifacts only; never claims to validate H65 or CUDA."""
from pathlib import Path
import ast, json, subprocess, sys
root=Path(__file__).resolve().parents[1]
records=[]
for p in sorted(root.rglob('*.json')):
    json.loads(p.read_text(encoding='utf-8')); records.append({'file':str(p.relative_to(root)),'check':'json_parse','status':'PASS'})
for p in sorted(root.rglob('*.py')):
    ast.parse(p.read_text(encoding='utf-8')); records.append({'file':str(p.relative_to(root)),'check':'python_ast','status':'PASS'})
for p in sorted(list(root.rglob('*.sh'))+list(root.rglob('*.sbatch'))):
    subprocess.run(['bash','-n',str(p)],check=True)
    records.append({'file':str(p.relative_to(root)),'check':'bash_syntax','status':'PASS'})
r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'reference'),'-p','test_*.py','-v'],capture_output=True,text=True)
result={'schema':'artifact_verification_v1','checks':records,'reference_unittest_returncode':r.returncode,
        'reference_unittest_output':r.stdout+r.stderr,
        'h65_tests':'NOT_RUN','gpu_training':'NOT_RUN','gpu_inference':'NOT_RUN',
        'checkpoint_validation':'NOT_RUN','codex_cli_execution':'NOT_RUN'}
(root/'ARTIFACT_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'artifact_checks':len(records),'reference_tests_returncode':r.returncode,'H65':'NOT_RUN'},ensure_ascii=False))
if r.returncode: print(r.stdout+r.stderr); raise SystemExit(r.returncode)
