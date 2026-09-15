"""Measured full-window routing overhead, keeping it separate from dataset means."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    source=json.loads(Path(args.profiles).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));rows=[]
    for ax,bb in zip(axes,('s','b')):
        a=source['profiles']['v2_'+bb]['full'];b=source['profiles']['uniform_'+bb]['full']
        assert a['window_index']==b['window_index'] and a['valid_candidates']==b['valid_candidates']==768
        ma,mb=a['macs_by_component'],b['macs_by_component'];delta={k:2*(ma.get(k,0)-mb.get(k,0))/1e9 for k in ma.keys()|mb.keys()}
        keys=['heavy_ffn','attention_routing_qk','light_ffn'];values=[delta.get(k,0) for k in keys]
        values.append(sum(v for k,v in delta.items() if k not in keys));net=(a['matrix_conv_flops']-b['matrix_conv_flops'])/1e9
        assert abs(sum(values)-net)<1e-6
        values.append(net);labels=['Heavy FFN','Routing QK','Light FFN','Frame / other','Net change']
        bars=ax.bar(range(5),values,color=['#2b8068' if v<0 else '#c46747' for v in values],width=.65)
        ax.axhline(0,color='#5e6670',lw=.8);ax.set_xticks(range(5),labels,rotation=12)
        low=min(values);high=max(values);span=high-low
        ax.set_ylim(low-span*.18,high+span*.3)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+(span*.025 if value>=0 else -span*.025),f'{value:+.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
        ax.set(title=f'VideoMAE-{bb.upper()}: {b["matrix_conv_flops"]/1e9:.2f}G → {a["matrix_conv_flops"]/1e9:.2f}G',ylabel='GFLOPs change relative to Uniform');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        rows.append(dict(backbone=bb,window_index=a['window_index'],v2_plan=a['plan'],uniform_plan=b['plan'],
                         uniform_full_window_gflops=b['matrix_conv_flops']/1e9,v2_full_window_gflops=a['matrix_conv_flops']/1e9,
                         changes=dict(zip(labels,values)),all_component_changes=delta,
                         v2_source=source['paths']['v2_'+bb],uniform_source=source['paths']['uniform_'+bb]))
    fig.suptitle('Routing overhead can erase spatial FFN savings',fontsize=15,y=.99)
    fig.text(.5,.02,'Measured representative full windows: V2 D100/S75 versus Uniform D100/S100. Matrix/conv 2MAC accounting; not a dataset-mean decomposition.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    for ext in ('png','svg'):fig.savefig(out/f'routing_cost_decomposition.{ext}',dpi=200)
    plt.close(fig);(out/'routing_cost_decomposition.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profiles',required=True);parser.add_argument('--output',required=True);main(parser.parse_args())
