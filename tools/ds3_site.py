"""Create only this project's symlinks to the authorized shared resources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SHARED=Path('/data/run01/sczc063/yuzibo')
LINKS={
 'thumos14/annotations':SHARED/'thumos14/annotations',
 'thumos14/videos/validation':SHARED/'raw/Validation Data/validation',
 'thumos14/videos/test':SHARED/'raw/Test Data/TH14_test_set_mp4',
 'checkpoints/adatad_s_ema.pth':SHARED/'rime_prerequisites/adatad_thumos_actionformer_videomae_s_768x1_160_adapter_epoch_59_21dbb9ef.pth',
 'checkpoints/adatad_b_ema.pth':SHARED/'bcr_tad_v3_motivation/checkpoints/official_b_checkpoint.pth',
 'checkpoints/mobilenet_v3_small.pth':SHARED/'pretrained/mobilenet_v3_small-047dcff4.pth',
}
if __name__=='__main__':
    for name,target in LINKS.items():
        if not target.exists():raise FileNotFoundError(target)
        path=ROOT/'resources'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_symlink() and path.resolve()==target.resolve():continue
        path.symlink_to(target,target_is_directory=target.is_dir())
    (ROOT/'ds3_20260912/site/run_job.sh').chmod(0o755)
    print('Own DS3 links prepared; shared resources unchanged.')
