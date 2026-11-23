"""Local draw.io-geometry exporter for this figure's explicit vector primitives.

No draw.io installation, network, browser, model, dataset or scientific execution.
The .drawio XML is read on every export. MathJax strings are converted to equivalent
Matplotlib mathtext. Formulas become vector glyph paths in SVG and embedded fonts
in PDF. Native source text, cards and edges remain editable in .drawio.
"""
from pathlib import Path
import os
import argparse
import re
import xml.etree.ElementTree as ET
import json

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'qa'/'mplconfig')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath

parser=argparse.ArgumentParser()
parser.add_argument('--pass-name',default='final')
args=parser.parse_args()
qa=HERE/'qa'
qa.mkdir(exist_ok=True)
tree=ET.parse(HERE/'fig01_pcf_framework.drawio')
model=tree.find('.//mxGraphModel')
W,H=float(model.get('pageWidth')),float(model.get('pageHeight'))
matplotlib.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'custom',
    'mathtext.rm':'DejaVu Sans','mathtext.it':'DejaVu Sans:oblique','mathtext.bf':'DejaVu Sans:bold',
    'pdf.fonttype':42,'svg.fonttype':'path'})
fig=plt.figure(figsize=(W/100,H/100),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W);ax.set_ylim(H,0);ax.set_axis_off()
texts=[]
def styles(c):
    return dict(s.split('=',1) for s in c.get('style','').split(';') if '=' in s)
def pt(g,name):
    el=g.find(f"mxPoint[@as='{name}']")
    return float(el.get('x')),float(el.get('y'))
cells=tree.findall('.//mxCell')
# Source order = card/label first, edges last visually; endpoints stay outside cards.
for c in cells:
    if c.get('vertex')!='1':continue
    g=c.find('mxGeometry');s=styles(c)
    x,y,w,h=[float(g.get(k,0)) for k in ('x','y','width','height')]
    fill=s.get('fillColor','none');stroke=s.get('strokeColor','none')
    if fill!='none' or stroke!='none':
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',
            facecolor=fill,edgecolor=stroke,linewidth=float(s.get('strokeWidth',2))*.72,
            zorder=1)
        ax.add_patch(p)
    label=c.get('value','')
    if not label:continue
    fs=float(s.get('fontSize',28));weight='bold' if s.get('fontStyle')=='1' else 'normal'
    halign=s.get('align','center'); tx=x+w/2 if halign=='center' else x+4
    valign=s.get('verticalAlign','middle')
    ty=y+h/2 if valign=='middle' else y+float(s.get('spacingTop',0))
    va='center' if valign=='middle' else 'top'
    if c.get('data-math')=='1':
        label='$'+label.removeprefix(r'\(').removesuffix(r'\)')+'$'
    artist=ax.text(tx,ty,label,ha=halign,va=va,fontsize=fs*.72,fontweight=weight,
          fontstyle='italic' if s.get('fontStyle')=='2' else 'normal',
          color=s.get('fontColor','#263238'),linespacing=1.12,zorder=3)
    texts.append((c.get('id'),artist,(x,y,w,h)))
for c in cells:
    if c.get('edge')!='1':continue
    g=c.find('mxGeometry');s=styles(c)
    points=[pt(g,'sourcePoint')]+[(float(p.get('x')),float(p.get('y'))) for p in g.findall("Array[@as='points']/mxPoint")]+[pt(g,'targetPoint')]
    color=s.get('strokeColor','#263238');lw=float(s.get('strokeWidth',2.5))*.72
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(PathPatch(path,fill=False,edgecolor=color,linewidth=lw,
        linestyle=(0,(7*.72,5*.72)) if s.get('dashed')=='1' else '-',zorder=2))
    # Explicit closed arrowhead, shared logical dimensions across all exports.
    a,b=points[-2:];dx,dy=b[0]-a[0],b[1]-a[1];length=(dx*dx+dy*dy)**.5
    ux,uy=dx/length,dy/length;size=10
    ax.add_patch(Polygon([b,(b[0]-size*ux+size*.43*uy,b[1]-size*uy-size*.43*ux),
                            (b[0]-size*ux-size*.43*uy,b[1]-size*uy+size*.43*ux)],
                         closed=True,facecolor=color,edgecolor=color,lw=.4,zorder=2))
fig.canvas.draw()
renderer=fig.canvas.get_renderer();overflows=[]
for cid,a,(x,y,w,h) in texts:
    bbox=a.get_window_extent(renderer)
    # display pixels have bottom-origin y, logical units have top-origin y.
    left,top,right,bottom=bbox.x0,H-bbox.y1,bbox.x1,H-bbox.y0
    if left<x-1 or right>x+w+1 or top<y-1 or bottom>y+h+1:
        overflows.append({'id':cid,'text_bounds':[left,top,right,bottom],'cell_bounds':[x,y,x+w,y+h]})
(qa/f'{args.pass_name}_render_bounds.json').write_text(json.dumps({'text_overflows':overflows},indent=2),encoding='utf-8')
for fmt in ('svg','pdf','png'):
    fig.savefig(HERE/f'fig01_pcf_framework.{fmt}',dpi=200 if fmt=='png' else 100,
                facecolor='white',metadata={'Creator':'Paper5 Figure 1 local vector exporter'} if fmt=='pdf' else None)
fig.savefig(qa/f'{args.pass_name}.png',dpi=100,facecolor='white')
plt.close(fig)
print(json.dumps({'exports':['drawio','svg','pdf','png'],'pass':args.pass_name,'text_overflows':overflows}))
