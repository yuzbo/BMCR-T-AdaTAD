"""Curate the research release without changing experiment code or source identity."""
import json
import io
from pathlib import Path
import shutil
import subprocess
import zipfile
from deploy_probe import LOCAL,OUT

BASE=LOCAL.parents[1]
ATLAS=LOCAL.parent/'wtr_characterization_20260915'
RAW=LOCAL.parent/'wtr_raw_v1_20260915'
HUB=LOCAL/'research/wtr_rfv_release_20260915'
ASSETS=OUT/'github_release_assets'
HUB.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(exist_ok=True)

def revision(root):
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()

sources={'rfv_implementation':revision(LOCAL),'atlas_closeout':revision(ATLAS),'raw':revision(RAW),
 'D_S_science':'40552945ad5d56b7404f833cd86f998e8558e8b1',
 'rfv_bank':'2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32',
 'r1':'b644d870d1845abbc1e4fd5ab7780f29ff96a53a',
 'reverse_capture_rise_export':'800bcd10a66c69d0d57142ac02389adbc45d2975',
 'reverse_fit':'4aa1ca242c37b3f7e2aa727b05419628a74e4e90',
 'ds_fixed_diagnostic':'72459727f61454194f0d84865f31a16ed4e10448',
 'atlas_measure':'a50b84db1bcdafd86ceec4d9737156807f9daa9a',
 'interaction_measure':'49f74bd5601bc298997921a93193109156adf546',
 'interaction_analysis':'1b3850e6db36fbfaaf43fdc71ea1e39ea22af065',
 'interaction_display_analysis':'0a9380a6560a789bd9b58a534a8c8329f7fb55f6',
 'interaction_renderer':'5d2402a11873f81ddccb29c861c4be80c9289378'}

def copy_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)

archive=HUB/'history/README.before_release.md'
if not archive.exists():copy_file(LOCAL/'README.md',archive)
for name in ('FORMAL_MODEL_DESIGN.zh.md','OWNER_DISCUSSION.zh.md','DISCUSSION_RECORD.zh.md'):
    copy_file(BASE/'reports/wtr_formal_design_20260915'/name,HUB/'design'/name)
reviews=BASE/'reports/wtr_rfv_review_20260915'
for name in ('R1_GATE_REVIEW.zh.md','R1_GATE_RECHECK.json','REVERSE_CONSISTENCY_MINI_REVIEW.zh.md',
 'REVERSE_RISE_800_EXECUTION_REVIEW.zh.md','REVERSE_RISE_800_RECHECK.json',
 'REVERSE_800_NORMALIZATION_RECHECK.json','REVERSE_4AA_EXPERIMENT_REVIEW.zh.md','REVERSE_4AA_STAT_RECHECK.json',
 'ATLAS_INTERACTION_FINAL_ALIGNMENT.zh.md'):
    source=reviews/name
    if source.exists():copy_file(source,HUB/'reviews'/name)
copy_file(LOCAL/'research/rfv_sprint_20260915/evidence/DS_FIXED_FINAL_REVIEW.json',HUB/'reviews/DS_FIXED_FINAL_REVIEW.json')
copy_file(LOCAL/'research/rfv_sprint_20260915/evidence/DS_FIXED_FINAL_REVIEW.json',OUT/'DS_FIXED_FINAL_REVIEW.json')
shutil.copytree(OUT/'publication_snapshot',HUB/'courses',dirs_exist_ok=True)
for name in ('RAW_SPLIT_CLARIFICATION.zh.md','REPORT.zh.md'):
    copy_file(BASE/'reports/wtr_fasttrack_20260915/progress_20260915_1100'/name,HUB/'raw/history'/name)
copy_file(BASE/'reports/wtr_fasttrack_20260915/current_model_report/MODEL_AND_EVIDENCE.zh.md',
    HUB/'raw/history/MODEL_AND_EVIDENCE_1150.zh.md')
for name in ('RAW_INPUT.zh.md','RAW_V1_CONTRACT.zh.md'):
    copy_file(BASE/'reports/wtr_raw_input_20260915'/name,HUB/'raw/design_history'/name)
for path in (ATLAS/'research/atlas_20260915').glob('*.md'):
    copy_file(path,HUB/'atlas/reports'/path.name)
figure_dirs=['output/pdf_s/local_choice_revision','output/temporal','output/interaction_within_s','output/interaction_v1_s']
allowed={'.png','.svg','.pdf','.md','.json','.csv'}
for directory in figure_dirs:
    for path in (ATLAS/directory).iterdir():
        if path.is_file() and path.suffix in allowed and not path.stem.startswith(('qa','page_')):
            copy_file(path,HUB/'atlas'/path.relative_to(ATLAS))
for name in ('temporal_s_summary.json','temporal_b_summary.json','allocation_s_D_summary.json',
 'recovery_paired_summary.json','local_choice_final_qa.json','interaction_final_qa.json',
 'interaction_within_qa.json','action_export_manifest.json','paper_controls_s.json'):
    path=ATLAS/'research/atlas_20260915/receipts'/name
    if path.exists():copy_file(path,HUB/'atlas/receipts'/name)
for name in ('INTERACTION_GATE.json','execution_audit.json'):
    copy_file(ATLAS/'analysis/interaction_v1_s'/name,HUB/'atlas/analysis'/name)

def zip_paths(target,root,paths):
    entries=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        seen=set()
        for source in paths:
            members=sorted(source.rglob('*')) if source.is_dir() else [source]
            for path in members:
                if not path.is_file():continue
                relative=path.relative_to(root).as_posix()
                if relative in seen:continue
                if any(p in {'runtime_local','legacy_runtime','tmp','node_modules','.git','assets'} for p in path.relative_to(root).parts):continue
                if path.suffix.lower() in {'.pth','.pt','.bin','.mp4','.avi','.mov','.jpg','.jpeg','.pyc'}:continue
                if any(p in path.name.lower() for p in ('resources.local','source_resources','id_rsa','id_ed25519')):continue
                seen.add(relative);bundle.write(path,relative)
                entries.append(dict(path=relative,bytes=path.stat().st_size))
        bundle.writestr('RELEASE_CONTENTS.json',json.dumps(dict(sources=sources,files=entries),indent=2))
    return dict(name=target.name,files=len(entries),uncompressed_bytes=sum(x['bytes'] for x in entries),bytes=target.stat().st_size)

assets=[]
atlas_paths=[ATLAS/x for x in ('analysis/models/s','analysis/interaction_v1_s','analysis/interaction_within_s',
    'analysis/ap','analysis/temporal_only.json','output/pdf_s/action_manifest',
    'research/atlas_20260915/receipts','logs') if (ATLAS/x).exists()]
assets.append(zip_paths(ASSETS/'atlas-s-numerical-evidence.zip',ATLAS,atlas_paths))
assets.append(zip_paths(ASSETS/'atlas-publication-figures.zip',ATLAS,[ATLAS/x for x in figure_dirs]))
for label,root in [('atlas',ATLAS),('raw',RAW)]:
    rev=revision(root)
    top=set(subprocess.check_output(['git','ls-tree','--name-only',rev],cwd=root,text=True).splitlines())
    selected=[p for p in ('h65','tools','tests','configs','references','upstream','research') if p in top]
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',rev,'--',*selected],cwd=root,text=True).splitlines()
    target=ASSETS/(label+'-source-'+rev[:7]+'.zip')
    raw=subprocess.check_output(['git','archive','--format=zip',rev,'--',*selected],cwd=root)
    excluded=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
        for entry in original.infolist():
            if entry.is_dir():continue
            parts=Path(entry.filename).parts
            if 'legacy_runtime' in parts or any(word in entry.filename.lower() for word in
                    ('resources.local.json','source_resources.local','id_rsa','id_ed25519')):
                excluded.append(entry.filename);continue
            output.writestr(label+'-source/'+entry.filename,original.read(entry))
    assets.append(dict(name=target.name,source_revision=rev,bytes=target.stat().st_size,
        excluded_local_or_duplicate_files=excluded))
manifest=dict(schema='WTR_RESEARCH_PUBLICATION_V1',sources=sources,artifacts=assets,
    scope='Complete important code, measured summaries, reviews, figures and available logs; no new model runs',
    experiment_automation='PAUSED',course_snapshot='courses/SNAPSHOT.json',
    exclusions='Credentials/local resource files, original videos, model weights, duplicate legacy runtime and temporary files. Prior historical release remains available.',
    release_tag='wtr-rfv-evidence-20260915')
(HUB/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(OUT/'PUBLICATION_STAGING.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(hub=str(HUB),sources=sources,assets=assets),ensure_ascii=False))
