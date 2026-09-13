"""Read actual downloaded tensors and build the paper resource inventory."""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def main(args):
    from h65.paper.runtime import json_write
    base=Path(args.site_root);old=base/'fpw_3d_20260913'
    old_res=json.loads((old/'research/frame/resources.local.json').read_text())
    shared=(old/'resources').resolve();assets=ROOT/'research/paper/assets'
    paths={'anet:s':assets/'adatad_anet_s.pth','anet:b':assets/'adatad_anet_b.pth',
           'internvideo_mq':assets/'internvideo1_mq.pth'}
    if args.dry_run:
        print(json.dumps(dict(paths={k:str(v) for k,v in paths.items()},read_tensors=False)));return
    import torch
    torch.set_num_threads(1)
    verified={};pending={}
    expected_bytes={'anet:s':121536592,'anet:b':394503824,'internvideo_mq':1223627459}
    for kind,path in paths.items():
        if not path.exists() or path.stat().st_size!=expected_bytes[kind]:
            pending[kind]=dict(checkpoint=str(path),expected_bytes=expected_bytes[kind],
                               current_bytes=path.stat().st_size if path.exists() else 0,status='transfer_pending')
            continue
        payload=torch.load(path,map_location='cpu')
        state=payload.get('state_dict_ema',payload.get('state_dict',payload.get('model',payload)))
        state={k.removeprefix('module.'):v for k,v in state.items()}
        patch=next((v for k,v in state.items() if k.endswith('patch_embed.projection.weight')),None)
        layers=sorted({int(m.group(1)) for k in state if (m:=re.search(r'blocks\.(\d+)\.',k))})
        expected=24 if kind=='internvideo_mq' else 12
        channels=1024 if kind=='internvideo_mq' else 384 if kind.endswith(':s') else 768
        if patch is None or list(patch.shape)!=[channels,3,2,16,16] or layers!=list(range(expected)):
            raise ValueError('Wrong backbone architecture in '+str(path))
        if kind.startswith('anet:') and not any(k.startswith('rpn_head.') for k in state):
            raise ValueError('ANet asset is not a task-adapted detector checkpoint')
        verified[kind]=dict(checkpoint=str(path),bytes=path.stat().st_size,channels=channels,layers=expected,
                            tensor_keys=len(state),source_epoch=payload.get('epoch'),weights_loaded=True)
        json_write(assets/(kind.replace(':','_')+'_verified.json'),verified[kind])
        del state,payload
    thumos_ann=shared/'thumos14/annotations/thumos_14_anno.json'
    anet_root=Path('/data/run01/sczc063/yuzibo/datasets/activitynet/annotations')
    anet_ann=anet_root/'annotations/activity_net.v1-3.min.json'
    db=json.loads(anet_ann.read_text())['database']
    category=anet_root/'annotations/category_idx.txt'
    if not category.exists():
        category=assets/'anet_category_idx.txt'
        labels=sorted({a['label'] for v in db.values() for a in v.get('annotations',[])})
        category.write_text('\n'.join(labels)+'\n')
    blocked=anet_root/'annotations/blocked.json';blocked_ids=set(json.loads(blocked.read_text()))
    def ids(path,subset,omit=()):
        return sorted(k for k,v in json.loads(Path(path).read_text())['database'].items() if v['subset']==subset and k not in omit)
    datasets=dict(thumos=dict(annotations=str(thumos_ann),class_map=str(shared/'thumos14/annotations/category_idx.txt'),
                  train_videos=str(shared/'thumos14/videos/validation'),test_videos=str(shared/'thumos14/videos/test'),
                  train_ids=ids(thumos_ann,'training'),test_ids=ids(thumos_ann,'validation')),
                  anet=dict(annotations=str(anet_ann),class_map=str(category),blocked_annotations=str(blocked),
                  train_videos='/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/activitynet/15fps_short256',
                  test_videos='/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/activitynet/15fps_short256',
                  classifier=str(anet_root/'classifiers/cuhk_val_simp_7.json'),
                  train_ids=ids(anet_ann,'training',blocked_ids),test_ids=ids(anet_ann,'validation',blocked_ids),
                  prepared_report='/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/OpenTAD/reports/data/anet_preparation/preparation.json',
                  ready_file=str(assets/'anet_ready.json')))
    encoders={};teachers={};recovery={}
    for b in ('s','b'):
        anchor=old_res['anchors'][b]
        encoders['thumos:'+b]=dict(checkpoint=anchor['checkpoint'],kind='task',variant=anchor['variant'],scout_checkpoint=anchor['checkpoint'])
        encoders['anet:'+b]=dict(checkpoint=str(paths['anet:'+b]),kind='task',variant='h65',scout_checkpoint=old_res['anchors']['s']['checkpoint'])
        teachers['thumos:'+b]=old_res['official'][b];teachers['anet:'+b]=str(paths['anet:'+b])
        recovery['thumos:'+b]=str(old/f'research/frame/runs/R03_cross_{b}/epoch_20.pth')
    for ds in ('thumos','anet'):
        encoders[ds+':internvideo_mq']=dict(checkpoint=str(paths['internvideo_mq']),kind='recognition',variant='h65',
                                         scout_checkpoint=old_res['anchors']['s']['checkpoint'])
    value=dict(schema=2,datasets=datasets,encoders=encoders,teachers=teachers,recovery_initialization=recovery,decoder_pretrain=old_res.get('decoder_pretrain',{}),
               verified_downloads=verified,pending_downloads=pending,old_frame_root=str(old),legacy_bmcr_root=str(base/'bmcr80_20260913'))
    for spec in datasets.values():
        for key in ('annotations','class_map'):
            if not Path(spec[key]).is_file():raise FileNotFoundError(spec[key])
    for spec in encoders.values():
        if not Path(spec['scout_checkpoint']).is_file():raise FileNotFoundError(spec['scout_checkpoint'])
    json_write(args.output,value);json_write(assets/'weights_verified.json',verified)
    print(json.dumps({'output':args.output,'verified':verified,'pending':pending,'datasets':{k:{'train':len(v['train_ids']),'test':len(v['test_ids'])} for k,v in datasets.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site-root',required=True)
    parser.add_argument('--output',default=str(ROOT/'research/paper/resources.local.json'))
    parser.add_argument('--dry-run',action='store_true');main(parser.parse_args())
