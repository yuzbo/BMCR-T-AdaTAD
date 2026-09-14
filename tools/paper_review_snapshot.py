"""Combine live receipts with the completed archive, preserving BMCR80 terminals."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write


def main(args):
    review=ROOT/'research/paper/review_5485'
    original=json.loads((review/'current_snapshot.json').read_text(encoding='utf-8'))
    live=json.loads(Path(args.live_snapshot).read_text(encoding='utf-8'))
    records={}
    prefix='/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/bmcr80_20260913/bmcr80_20260913/runs/'
    for values in original.get('evidence',{}).values():
        for item in values:
            # Initial snapshots contain only 22 BMCR metrics, before both final
            # tests completed. Use the 24 finished receipts below instead.
            if item['source'].startswith(prefix):continue
            records[item['source']]=item
    archive=review/'bmcr80_final/receipts/runs'
    completed=sorted(archive.glob('*_bmcr_test_epoch_*/completed.json'))
    if len(completed)!=24:raise RuntimeError('Expected all 24 archived BMCR80 full tests')
    for path in completed:
        record=json.loads(path.read_text(encoding='utf-8'))
        profile=path.with_name('profile.json')
        if profile.exists():
            measured=json.loads(profile.read_text(encoding='utf-8'))
            record.update(gflops=measured['matrix_conv_flops']/1e9,
                          representative_gflops=measured['matrix_conv_flops']/1e9,
                          latency_ms=1000*measured['latency_mean_seconds'])
        source=prefix+path.parent.name+'/completed.json'
        records[source]=dict(source=source,record=record)
    for values in live.get('evidence',{}).values():
        for item in values:records[item['source']]=item
    json_write(args.output,dict(time=live['time'],evidence=dict(all=[records[k] for k in sorted(records)]),
        archive_scope='Initial historical evidence plus all completed BMCR80 tests; live current results override identical receipt paths'))
    print(json.dumps(dict(archived_bmcr_tests=len(completed),receipt_sources=len(records),snapshot_time=live['time'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live-snapshot',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
