"""Read-only official decoder key/shape audit; no CUDA context or invented keys."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--register-resources');p.add_argument('--backbone',choices=['s','b']);a=p.parse_args()
    from h65.frame.mae_init import read_pretraining,MAELatentDecoder
    from h65.frame.runtime import json_write
    state,spec=read_pretraining(a.checkpoint);del state
    model=MAELatentDecoder(**{k:spec[k] for k in ('channels','width','layers','heads')})
    result=model.load_pretraining(a.checkpoint);json_write(a.output,result)
    if a.register_resources:
        if a.backbone is None or spec['channels']!=dict(s=384,b=768)[a.backbone]:raise ValueError('Specify the matching S/B backbone')
        path=Path(a.register_resources);resources=json.loads(path.read_text())
        resources.setdefault('decoder_pretrain',{})[a.backbone]=dict(checkpoint=str(Path(a.checkpoint).resolve()),**{k:spec[k] for k in ('channels','width','layers','heads')})
        json_write(path,resources)
    print(json.dumps(result))

if __name__=='__main__':main()
