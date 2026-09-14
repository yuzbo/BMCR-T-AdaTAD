"""Prepare read-only shared dependencies for the isolated graph runtime."""
import json
import shutil
from pathlib import Path

base=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910')
source=base/'support_review_20260914';target=base/'graph_tad_20260914'
target.mkdir(exist_ok=True)
shared={'upstream':source/'upstream','references':source/'references','resources':base/'fpw_3d_20260913/resources'}
for name,src in shared.items():
    dst=target/name
    if not src.exists():raise RuntimeError('Missing known shared dependency: '+str(src))
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=True)
folder=target/'research/paper';folder.mkdir(parents=True,exist_ok=True)
resource=folder/'resources.local.json'
if not resource.exists():shutil.copy2(source/'research/paper/resources.local.json',resource)
print(json.dumps(dict(runtime=str(target),shared_dependencies={name:str((target/name).resolve()) for name in ('upstream','references','resources')},resources=str(resource),gpu_jobs_submitted=False)))
