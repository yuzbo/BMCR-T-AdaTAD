"""Publication-size vector figure of the implemented Full-V2 (no carrier path).

Explicit ports, one arrowhead per edge, and local residual loops. Panel letters
replace zoom leader lines. PDF is 7.1 inches wide for a two-column placement.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/paper/review_5485/analysis/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})
fig, ax = plt.subplots(figsize=(7.1, 4.7))
ax.set(xlim=(0, 180), ylim=(0, 119)); ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
C = dict(ink='#24384A', muted='#607587', edge='#8496A4',
         teal='#238F89', teal_bg='#EEF8F5', gold='#B58335', gold_bg='#FCF5E8',
         violet='#8063AD', violet_bg='#F3EFF8', line='#587182')
labels = []
def txt(x,y,s,size=7.2,color='ink',ha='center',bold=False):
    a=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=C.get(color,color),
              weight='bold' if bold else 'normal',linespacing=1.25,zorder=7)
    labels.append(a); return a
def box(x,y,w,h,label='',kind='plain',size=7.2):
    edge,fill=(C['edge'],'white') if kind=='plain' else (C[kind],C[kind+'_bg'])
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.0',
                    edgecolor=edge,facecolor=fill,lw=.65,zorder=3)
    ax.add_patch(p)
    if label: txt(x+w/2,y+h/2,label,size)
    return p
def edge(points,color='line',dashed=False,head=True,lw=.8):
    # A single continuous path ensures elbows are joins, never stray arrows.
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    p=FancyArrowPatch(path=path,arrowstyle='-|>' if head else '-',
                     mutation_scale=6.3,linewidth=lw,color=C[color],
                     linestyle=(0,(3,2)) if dashed else '-',capstyle='round',
                     joinstyle='round',zorder=4)
    ax.add_patch(p)
def dot(x,y,color='line'):
    ax.add_patch(Circle((x,y),.42,facecolor=C[color],edgecolor='none',zorder=5))
def plus(x,y):
    ax.add_patch(Circle((x,y),1.65,facecolor='white',edgecolor=C['edge'],lw=.65,zorder=6))
    txt(x,y,'+',8.2)
def panel(x,y,w,h,title):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.5',
                              facecolor='#FCFDFE',edgecolor='#D7E1E7',lw=.65,zorder=0))
    txt(x+2,y+h-3.8,title,7.8,ha='left',bold=True)

# (a) Routing has its own upper lane; RGB/features use the horizontal lower lane.
txt(3,116.3,'(a) Full-V2: sparse computation, full-axis detection',8.3,ha='left',bold=True)
for i in (2,1,0):
    ax.add_patch(Rectangle((4+i*.75,77+i*.75),12,11,facecolor='#F7F9FB',edgecolor=C['edge'],lw=.6,zorder=2))
txt(10,82.3,'RGB',8,bold=True);txt(10,73.8,'768 frames',6.5)
box(4,101,18,8,'Cheap Scout\n96D','plain',7)
box(62,101,30,8,'Budget router\n15 T/D/S plans','violet',7)
box(27,91,25,8,'H65 / BMCR\n+ frame refinement','violet',6.7)
box(27,77,25,10,'Gather frames\nPack + embed','gold',7)

box(62,76,43,12,'','teal')
txt(83.5,84.8,'Sparse VideoMAE + TIA',7.2,bold=True)
txt(83.5,80.4,'4 dense → late A-MoD   (b)',6.7)
# Explicit original-layer schedule, not a misleading nested ×12.
for i in range(12):
    routed=i in (4,6,8,10)
    ax.add_patch(Rectangle((64+i*3.3,70.8),2.55,2.1,
                          facecolor=C['gold'] if routed else '#D2E5E1',edgecolor='none',zorder=3))
txt(83.5,68.3,'12 blocks; routed blocks 5 / 7 / 9 / 11',5.9,'muted')

box(119,76,26,12,'Cross recovery  (c)\nOriginal time axis','teal',7)
box(154,76,23,12,'Trainable TAD head\n768 positions','plain',6.5)
txt(132,72.8,'384 queries',6.3,'teal')
for x,y,w,col in ((158,69.9,8,'teal'),(167,69.1,7,'gold'),(162,68.3,6,'line')):
    ax.plot([x,x+w],[y,y],lw=1.4,color=C[col],solid_capstyle='butt')

# Observations and feature flow.
edge([(10,89.5),(10,101)])
edge([(16,82),(27,82)],'gold')
edge([(52,82),(62,82)],'gold')
edge([(105,82),(119,82)],'teal');txt(112,85.7,'L6/9/12',5.8,'teal')
edge([(145,82),(154,82)],'teal');txt(149.5,85.7,'resize',5.8,'muted')
edge([(165.5,76),(165.5,70.8)])

# Cheap evidence branches before budget and selection; no crossing edges.
edge([(22,105),(62,105)],'teal')
dot(24,105,'teal');edge([(24,105),(24,96),(27,96)],'teal')
edge([(62,103),(57,103),(57,94),(52,94)],'violet',True)
txt(55.1,98.1,'K',6.5,'violet')
edge([(77,101),(77,88)],'violet',True)
txt(82.3,94.6,'D / S',6.7,'violet')
edge([(39.5,91),(39.5,87)],'violet',True)
txt(44.7,89.1,r'$S_K$',6.6,'violet')
edge([(13,109),(13,112.1),(132,112.1),(132,88)],'teal')
txt(113,115,'Full-axis Scout context',6.5,'teal')
txt(139.1,99.7,'+ original\ntime positions',6.3,'teal',ha='left')
txt(28,72.8,'Individual-frame selection',6.1,'muted',ha='left')

# (b) Every residual has its own local lane. The two updates are piecewise
# functions on disjoint masks, not two whole-grid dense branches.
panel(3,22,116,43,'(b) State-preserving routed block')
txt(61,56.8,r'Attention scores → depth mask $M_D$; heavy FFN mask $M_H\subseteq M_D$',6.4,'muted')
txt(6.5,44.5,r'$X_l$',7.7)
box(13,37,28,14,'','gold')
txt(27,48.2,'Attention update',6.7,bold=True)
txt(27,43.7,'Selected Q / full KV',6.2)
txt(27,39.8,'Bypassed: depth-light',5.9,'teal')
edge([(9,44.5),(13,44.5)],'gold');dot(10,44.5,'gold')
edge([(41,44.5),(46.35,44.5)],'gold');plus(48,44.5)
edge([(10,44.5),(10,30.5),(48,30.5),(48,42.85)])
edge([(49.65,44.5),(58,44.5)],'gold');dot(54,44.5,'gold')
box(58,37,31,14,'','gold')
txt(73.5,48.2,'FFN update',6.7,bold=True)
txt(73.5,44.6,r'$M_H$: heavy',6.1)
txt(73.5,40.7,r'$M_D\backslash M_H$: light;  $\bar M_D$: depth-light',5.45,'teal')
edge([(89,44.5),(92.35,44.5)],'gold');plus(94,44.5)
edge([(54,44.5),(54,53.2),(94,53.2),(94,46.15)])
edge([(95.65,44.5),(100,44.5)],'teal')
box(100,38.5,9.5,12,'TIA\nSelected\ntimes','teal',5.8)
edge([(109.5,44.5),(112.6,44.5)],'teal')
txt(115.2,44.5,r'$X_{l+1}$',6.4,'teal')
txt(27.8,28.2,'Residual state retained',6.2,'muted')
txt(61,24.9,'KV: full packed attention domain  ·  TIA: selected K/2 timeline  ·  LN omitted',5.8,'muted')

# (c) Anchor interpolation is a residual base, not selected-slot replacement.
panel(122,22,55,43,'(c) Original-time latent recovery')
box(126,52,21,7,'Multi-layer\nanchors','gold',6.4)
box(153,52,21,7,'Query seeds¹','teal',6.4)
box(133,39,39,8,'Cross decoder\n192D · 2 layers','teal',6.7)
edge([(139,52),(139,47)],'gold')
edge([(163.5,52),(163.5,47)],'teal')
edge([(126,55.5),(124.4,55.5),(124.4,31.5),(129,31.5)],'gold')
box(129,28,20,7,'Interpolation','gold',6.1)
edge([(149,31.5),(154.35,31.5)],'gold');plus(156,31.5)
edge([(156,39),(156,33.15)],'teal');txt(158,36,'Residual',5.8,'teal',ha='left')
edge([(157.65,31.5),(173,31.5)],'teal')
txt(165.3,27.5,r'$\hat F_{1:T}$',7.2,'teal')
txt(149.5,24.7,'¹ Interpolation + Scout + time metadata',5.8,'muted')

# Objective strip is separate from the inference graph; no false teacher path.
box(3,5.9,174,12.5,'','violet')
txt(6,15.4,'TRAINING ONLY',6.2,'violet',ha='left',bold=True)
txt(34,10.8,'Same-support full-D/S reference\nIntermediate state alignment',6.1)
txt(89,10.8,'External dense teacher\nFeature supervision',6.1)
txt(144,10.8,'Shared-full student branch\nGT + self-distillation',6.1)
for x in (62,117): ax.plot([x,x],[8.1,15.7],color='#D7CCE6',lw=.55)
edge([(4,2.7),(12,2.7)],'teal');txt(14,2.7,'feature flow',5.8,'muted',ha='left')
edge([(45,2.7),(53,2.7)],'violet',True);txt(55,2.7,'budget / selection control',5.8,'muted',ha='left')
txt(176,2.7,'THUMOS · VideoMAE-S/B · C = 384/768',5.8,'muted',ha='right')

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
canvas=fig.bbox
clipped=[]
for label in labels:
    bb=label.get_window_extent(renderer)
    if bb.x0<canvas.x0 or bb.x1>canvas.x1 or bb.y0<canvas.y0 or bb.y1>canvas.y1:
        clipped.append(label.get_text())
if clipped: raise RuntimeError('Canvas clipping: '+repr(clipped))
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'model_full_v2_cvpr.{ext}',dpi=500,facecolor='white')
(OUT/'model_full_v2_cvpr.layout.json').write_text(json.dumps(dict(
    size_inches=[7.1,4.7],architecture='implemented Full-V2',
    source_files=['h65/paper/model.py','h65/paper/encoder.py','h65/paper/engine.py','h65/paper/decoder.py'],
    canvas_clipping=[],font_sizes_pt=[5.45,8.3],
    unimplemented_carrier_shown=False,performance_claims=False),indent=2),encoding='utf-8')
print(OUT/'model_full_v2_cvpr.pdf')
