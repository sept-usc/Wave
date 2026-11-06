#!/usr/bin/env python3
"""Render Wave's original explanatory diagrams, editable sources and videos.

Requires Pillow; MP4 additionally requires ffmpeg or imageio-ffmpeg.
Uses the Lanshu diagram skill's MIT-licensed shape/Excalidraw primitives.
No experiment results are synthesized by this renderer.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '_tools'))
import lanshu_renderer as ls

W, H, FPS = 1280, 800, 10
BG, PANEL, INK, DIM = '#081012', '#101d20', '#f3f4ef', '#bac7c9'
TEAL, GREEN, AMBER, RED = '#7de3d2', '#a5e49a', '#f4c06a', '#ff9c91'
EDGE = '#34474c'
ls.THEME['bg'] = BG


class Board:
    def __init__(self, title, description):
        self.ex = ls.Excal(W, H)
        self.im = Image.new('RGBA', (W * 2, H * 2), BG)
        self.draw = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc><rect width="1280" height="800" fill="{BG}"/>']

    def rect(self, x, y, w, h, fill=PANEL, edge=EDGE, r=14, sw=1.5):
        ls.draw_rect(self.ex, self.draw, x, y, w, h, edge, fill, sw, r)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{edge}" stroke-width="{sw}"/>')

    def text(self, value, x, y, w, size=23, color=INK, align='left', bold=False, line=1.32):
        lines = value.split('\n')
        for i, value in enumerate(lines):
            font = ls.load_font(size, bold=bold)
            bbox = self.draw.textbbox((0, 0), value, font=font)
            tw = (bbox[2]-bbox[0])/2
            if tw > w + 2:
                raise ValueError(f'Text exceeds width {tw:.1f}>{w}: {value}')
            xx = x if align == 'left' else x + (w-tw)/2 if align == 'center' else x+w-tw
            yy = y + i*size*line
            self.draw.text((xx*2, yy*2), value, font=font, fill=color, anchor='lt')
            self.ex.text(value, xx, yy, tw, size*line, size, color)
            self.svg.append(f'<text x="{xx:.2f}" y="{yy+size*.88:.2f}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}"'+(' font-weight="700"' if bold else '')+'>'+html.escape(value)+'</text>')

    def line(self, points, color=INK, sw=2, arrow=False, dashed=False):
        ls.draw_line(self.ex, self.draw, points, color, sw, 'dashed' if dashed else 'solid', arrow)
        self.svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{sw}"'+(' stroke-dasharray="7 7"' if dashed else '')+'/>')
        if arrow:
            a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=10
            p=[b,(b[0]-length*math.cos(angle-.5),b[1]-length*math.sin(angle-.5)),(b[0]-length*math.cos(angle+.5),b[1]-length*math.sin(angle+.5))]
            self.svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in p)+f'" fill="{color}"/>')

    def dot(self,x,y,r,color=TEAL):
        ls.draw_ellipse(self.ex,self.draw,x-r,y-r,r*2,r*2,color,color,1)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def save(self, stem):
        self.im.resize((W,H),Image.Resampling.LANCZOS).convert('RGB').save(stem.with_suffix('.png'))
        stem.with_suffix('.svg').write_text('\n'.join(self.svg+['</svg>']))
        self.ex.write(stem.with_suffix('.excalidraw'))

    def image(self):
        return self.im.resize((W,H),Image.Resampling.LANCZOS).convert('RGB')


def header(b,title,sub,number):
    b.rect(18,18,1244,764,BG,EDGE,22)
    b.rect(48,44,112,47,'#173d32','#173d32',12)
    b.text('WAVE',60,53,88,29,GREEN,'center',True)
    b.text(title,180,50,800,33)
    b.text('verifiable.pro',1030,58,200,19,DIM,'right')
    b.text(sub,49,111,1180,23,DIM)
    b.text(number,48,748,160,15,DIM)


OBS_STEPS=[
    ('01  Execute privately','The model produces GPU kernels. Its weight values are not supplied to the verifier.'),
    ('02  Observe hardware activity','Arithmetic and memory counters constrain the work that could have occurred.'),
    ('03  Construct constraints','Operation templates and observed counts restrict feasible model dimensions.'),
    ('04  Search for a violation','The verifier asks whether any feasible architecture crosses the claimed size bound.'),
]
OBS_SPEC=ROOT/'wave-observation-spec.json'
if OBS_SPEC.exists():
    OBS_STEPS=[tuple(row) for row in json.loads(OBS_SPEC.read_text())['steps']]


def observation(t=11.99,static=False):
    step=min(3,int(t//3));phase=(t%3)/3
    b=Board('Wave: from GPU observations to model-size bounds','Conceptual observation flow. The secure authenticated measurement path is assumed, not supplied by the released Nsight Compute prototype.')
    header(b,'Observe execution, constrain architecture','Model oversight without giving the verifier the weight values.','01 / OBSERVATION')
    b.text('PRIVATE EXECUTION',54,172,300,15,TEAL)
    b.text('OBSERVATIONS AND REASONING',370,172,820,15,DIM)
    xs=[49,356,663,970];cw=260
    titles=['Private model','GPU counters','Constraints','Bound checks']
    labels=['Inference kernels\nWeights stay private','Arithmetic activity\nMemory load traffic','Operation templates\nFeasible dimensions','Minimum model size\nMaximum model size']
    for i,x in enumerate(xs):
        active=static or i==step
        b.rect(x,207,cw,238,PANEL,TEAL if active else EDGE,16,2.5 if active else 1.5)
        b.text(f'0{i+1}',x+19,226,40,18,TEAL if active else DIM)
        b.text(titles[i],x+19,261,cw-38,27)
        b.text(labels[i],x+19,375,cw-38,19,DIM)
        if i==0:
            for rr in range(2):
                for cc in range(5):
                    b.rect(x+22+cc*39,308+rr*22,27,14,'#29423e',GREEN if active else EDGE,3,1)
        elif i==1:
            for j,v in enumerate([170,115,198]):
                b.rect(x+22,305+j*18,210,9,'#233237','#233237',4,1)
                b.rect(x+22,305+j*18,max(4,int(v*(phase if i==step and not static else 1))),9,TEAL,TEAL,4,1)
        elif i==2:
            b.text('FLOPs ~ 2 m n k',x+20,309,cw-40,23,TEAL)
            b.text('loads constrain size',x+20,340,cw-40,16,DIM)
        else:
            b.text('Can a violating',x+20,306,cw-40,21,TEAL)
            b.text('model still fit?',x+20,334,cw-40,21,TEAL)
        if i<3:
            b.line([(x+cw+8,326),(xs[i+1]-11,326)],INK,2,True)
            if i==step and not static:
                b.dot(x+cw+8+(xs[i+1]-11-x-cw-8)*phase,326,5,TEAL)
    b.rect(49,467,1181,85,'#142b2c','#305450',12)
    title,caption=OBS_STEPS[step]
    if static:title,caption='The result is a bound under stated assumptions','It does not identify exact weights or establish that an answer is correct.'
    b.text(title,69,482,1141,23,TEAL)
    b.text(caption,69,517,1141,19,INK)
    b.rect(49,575,575,133,PANEL,AMBER,12)
    b.text('Required trust assumption',69,594,535,22,AMBER)
    b.text('Authenticated counters and a trace tied to the\ninference run. That secure path is not supplied.',69,632,535,20,DIM)
    b.rect(643,575,587,133,PANEL,EDGE,12)
    b.text('Released research prototype',663,594,547,22)
    b.text('Nsight Compute / CSV preprocessing / Z3.\nThe diagram is conceptual, not a deployment claim.',663,632,547,20,DIM)
    b.text('Exact model identity, output correctness and behavior remain outside this check.',260,748,970,15,DIM,'right')
    return b


CASES=[
    {'kind':'Lower bound','claim':'P ≥ 75','threshold':75,'query':'Search for a smaller feasible model.','result':'UNSAT','detail':'No feasible candidate lies below 75.','witness':None,'direction':'lower'},
    {'kind':'Lower bound','claim':'P ≥ 100','threshold':100,'query':'Search for a smaller feasible model.','result':'SAT','detail':'P = 90 is feasible and violates the claimed minimum.','witness':90,'direction':'lower'},
    {'kind':'Upper bound','claim':'P ≤ 125','threshold':125,'query':'Search for a larger feasible model.','result':'UNSAT','detail':'No feasible candidate lies above 125.','witness':None,'direction':'upper'},
    {'kind':'Upper bound','claim':'P ≤ 100','threshold':100,'query':'Search for a larger feasible model.','result':'SAT','detail':'P = 110 is feasible and violates the claimed maximum.','witness':110,'direction':'upper'},
]
BOUNDS_SPEC=ROOT/'wave-bounds-spec.json'
if BOUNDS_SPEC.exists():
    CASES=json.loads(BOUNDS_SPEC.read_text())['cases']


def bounds(t=5.5,static=False):
    step=min(3,int(t//4));phase=(t%4)/4;c=CASES[step]
    b=Board('Wave: what the bound search asks','Illustrative feasible parameter range from 80 to 120 in arbitrary units, not measured results or an actual solver trace. SAT means a violating candidate exists. An actual unknown outcome is inconclusive.')
    header(b,'Search for a counterexample','Toy example: observations allow parameter size P from 80 to 120.','02 / BOUND SEARCH')
    b.text(c['kind']+': claimed '+c['claim'],49,174,1181,30)
    b.text(c['query'],49,217,1181,23,DIM)
    x=lambda p: 100+(p-50)*10.8
    b.rect(x(80),300,x(120)-x(80),102,'#193c37','#386f63',5)
    b.text('Feasible models: 80–120',x(80)+15,317,x(120)-x(80)-30,23,GREEN,'center')
    if c['direction']=='lower':lo,hi=50,c['threshold'];label='Search below the minimum'
    else:lo,hi=c['threshold'],150;label='Search above the maximum'
    b.rect(x(lo),422,max(2,x(hi)-x(lo)),39,'#472e2a','#9c655c',4)
    b.text(label,x(lo)+5,433,x(hi)-x(lo)-10,17,RED,'center')
    b.line([(x(50),402),(x(150),402)],INK,2)
    for n in [50,60,70,80,90,100,110,120,130,140,150]:
        b.line([(x(n),402),(x(n),410)],DIM,1)
        b.text(str(n),x(n)-20,469,40,17,DIM,'center')
    tx=x(c['threshold'])
    b.line([(tx,270),(tx,464)],AMBER,2,False,True)
    b.text('Claimed bound',tx-77,265,154,17,AMBER,'center')
    searchpos=lo+(hi-lo)*min(1,phase*1.8)
    if c['witness'] is not None and (static or phase>.42):
        searchpos=c['witness'];b.dot(x(searchpos),375,10,RED)
        b.text('Violating candidate',x(searchpos)-103,350,206,17,RED,'center')
    elif not static:
        b.dot(x(searchpos),441,6,RED)
    b.rect(49,511,1181,104,'#14252a',EDGE,12)
    result=c['result'];color=RED if result=='SAT' else GREEN
    b.rect(68,530,123,62,'#2d2421' if result=='SAT' else '#203b2e',color,9)
    b.text(result,75,548,109,27,color,'center')
    b.text(c['detail'],215,532,990,25)
    b.text('Claim is not certified.' if result=='SAT' else 'No violation within this modeled feasible set.',215,568,990,22,DIM)
    for i,case in enumerate(CASES):
        xx=49+i*298
        b.rect(xx,640,287,62,PANEL,TEAL if i==step else EDGE,10,2 if i==step else 1)
        b.text(case['claim']+': '+case['result'],xx+15,661,257,21,TEAL if i==step else DIM,'center')
    b.text('Illustrative units, not paper measurements. An UNKNOWN result would remain inconclusive.',260,748,970,15,DIM,'right')
    return b


def ffmpeg_path():
    found=shutil.which('ffmpeg')
    if found:return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:return None


def render(name,fn,seconds,static_t):
    stem=ROOT/name
    fn(static_t,True).save(stem)
    frames=[fn(i/FPS).image() for i in range(seconds*FPS)]
    # Shared palette avoids text shimmer across animation frames.
    palette=Image.new('RGB',(W,H*5))
    for n,i in enumerate([0,len(frames)//4,len(frames)//2,3*len(frames)//4,len(frames)-1]):
        palette.paste(frames[i],(0,n*H))
    pal=palette.quantize(colors=192,method=Image.Quantize.MEDIANCUT)
    indexed=[f.quantize(palette=pal,dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(stem.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=100,loop=0,optimize=False,disposal=1)
    encoder=ffmpeg_path()
    if encoder:
        with tempfile.TemporaryDirectory(prefix='wave-frames-') as tmp:
            tmp=Path(tmp)
            for i,im in enumerate(frames):im.save(tmp/f'{i:04d}.png')
            cmd=[encoder,'-hide_banner','-loglevel','error','-y','-framerate',str(FPS),'-i',str(tmp/'%04d.png'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(stem.with_suffix('.mp4'))]
            subprocess.run(cmd,check=True)
    result={'png':str(stem.with_suffix('.png')),'gif':str(stem.with_suffix('.gif')),'excalidraw':str(stem.with_suffix('.excalidraw'))}
    checks=ls.check_outputs(result,{'canvas':{'width':W,'height':H,'fps':FPS,'frames':len(frames)}})
    # GIF may legally merge identical held frames. Validate timeline duration instead.
    with Image.open(result['gif']) as gif:
        durations=[]
        for i in range(gif.n_frames):gif.seek(i);durations.append(gif.info['duration'])
    checks['checks']=[x for x in checks['checks'] if x['name'] not in ['gif_frames','gif_fps']]
    checks['checks'].append({'name':'gif_duration_ms','ok':sum(durations)==seconds*1000,'expected':seconds*1000,'actual':sum(durations)})
    checks['checks'].append({'name':'gif_multiple_frames','ok':len(durations)>1,'actual':len(durations)})
    checks['ok']=all(x['ok'] for x in checks['checks'])
    report={'name':name,'dimensions':[W,H],'timeline_fps':FPS,'seconds':seconds,'svg':'editable vectors','mp4_codec':'H.264 / yuv420p' if encoder else None,'checks':checks,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob(name+'.*')}}
    (ROOT/(name+'-validation.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not checks['ok']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=['observation','bounds']);args=p.parse_args()
    if args.only in [None,'observation']:render('wave-observation',observation,12,11.9)
    if args.only in [None,'bounds']:render('wave-bounds',bounds,16,5.5)
