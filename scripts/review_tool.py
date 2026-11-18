"""Point-and-click editor for the board's vias, test pads, solder pads and parts.

Run from your project folder, then use the page that opens in your browser:
  python review_tool.py

It shows the map of each side with everything the detector found. You can add items it
missed, remove wrong ones, change what an item is, and click a piece of copper to see
what it joins. Every change is saved straight away to data/review.csv (a backup,
review.csv.bak, is kept), and the final lists (vias.csv, testpads.csv, pads.csv,
components.csv) are rewritten.

If review.csv is changed by something else while the page is open, the editor notices,
reloads the list and asks you to redo your last change, so nothing is overwritten.

Options:  --port 8765   --no-browser
The editor itself needs only the standard library; "Follow a trace" needs Pillow and numpy.
It listens on localhost only.
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board as BD  # noqa: E402

ROOT = BD.PROJECT
PCB = BD.DATA
MAPS = BD.MAPS
REVIEW = os.path.join(PCB, "review.csv")
FIELDS = ["action", "side", "kind", "id", "x_mm", "y_mm", "w_mm", "h_mm", "note"]
LOCK = threading.Lock()


def read_csv(name):
    path = os.path.join(PCB, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def version():
    return str(os.stat(REVIEW).st_mtime_ns) if os.path.exists(REVIEW) else "0"


def state():
    with LOCK:
        return dict(auto=read_csv("features-auto.csv"), review=read_csv("review.csv"),
                    known=read_csv("known-parts.csv"), boxes=read_csv("placeholders.csv"), version=version())


def save_review(rows, base):
    """Returns (ok, new_version). Refuses when review.csv changed since the page loaded it."""
    with LOCK:
        if base != version():
            return False, version()
        if os.path.exists(REVIEW):
            shutil.copyfile(REVIEW, REVIEW + ".bak")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in FIELDS})
        with open(REVIEW + ".tmp", "w", newline="") as fh:
            fh.write(buf.getvalue())
        os.replace(REVIEW + ".tmp", REVIEW)
        new = version()
        subprocess.run([sys.executable, os.path.join(HERE, "find_features.py")],
                       cwd=ROOT, capture_output=True, text=True)
    return True, new


class Tracer:
    """Builds the trace-following data on first use (about 15 s) and keeps it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.graph = None
        self.mod = None
        self.built = ""

    def get(self, rebuild=False):
        with self.lock:
            if self.graph is None or rebuild:
                import trace_graph
                self.mod = trace_graph
                self.graph = trace_graph.build()
                self.built = time.strftime("%H:%M")
            return self.graph


TRACER = Tracer()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            page = (PAGE.replace("__MARGIN__", "%g" % BD.MARGIN).replace("__MAPW__", "%g" % BD.MAP_W_MM)
                    .replace("__BW__", "%g" % BD.BW).replace("__BH__", "%g" % BD.BH))
            self.send(200, page, "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send(200, json.dumps(state()), "application/json")
        elif path == "/api/trace":
            q = urllib.parse.parse_qs(self.path.partition("?")[2])
            try:
                g = TRACER.get(rebuild=q.get("rebuild", ["0"])[0] == "1")
                out = TRACER.mod.trace(g, q["side"][0], float(q["x"][0]), float(q["y"][0]),
                                       q.get("vias", ["1"])[0] == "1", q.get("edges", ["1"])[0] == "1")
                out["built"] = TRACER.built
            except ImportError as e:
                out = dict(ok=False, reason="Trace following needs numpy and Pillow (%s)." % e)
            except Exception as e:  # keep the editor alive whatever happens in here
                out = dict(ok=False, reason="Trace following failed: %s" % e)
            self.send(200, json.dumps(out), "application/json")
        elif path in ("/copper/top.jpg", "/copper/bottom.jpg"):
            f = os.path.join(MAPS, "%s-copper.jpg" % path[8:-4])
            if not os.path.exists(f):
                self.send(404, "copper map not found: run map_traces.py", "text/plain")
                return
            with open(f, "rb") as fh:
                self.send(200, fh.read(), "image/jpeg")
        elif path in ("/map/top.jpg", "/map/bottom.jpg"):
            f = os.path.join(MAPS, "%s-grid.jpg" % path[5:-4])
            if not os.path.exists(f):
                self.send(404, "map not found: run build_grid_map.py", "text/plain")
                return
            with open(f, "rb") as fh:
                self.send(200, fh.read(), "image/jpeg")
        else:
            self.send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/review":
            self.send(404, "not found", "text/plain")
            return
        origin = self.headers.get("Origin")
        if origin and "localhost" not in origin and "127.0.0.1" not in origin:
            self.send(403, "forbidden", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            rows, base = body["rows"], str(body["base"])
            assert isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
        except Exception:
            self.send(409, json.dumps(dict(ok=False, reason="old page")), "application/json")
            return
        ok, new = save_review(rows, base)
        self.send(200 if ok else 409, json.dumps(dict(ok=ok, version=new)), "application/json")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PCB review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#14171a;--panel:#1e2327;--line:#353c42;--text:#e8eaec;--dim:#9aa3ab;--accent:#4aa3ff}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,Segoe UI,Arial,sans-serif}
#app{display:grid;grid-template-columns:280px minmax(0,1fr);grid-template-rows:minmax(0,1fr);height:100vh;overflow:hidden}
#side{background:var(--panel);border-right:1px solid var(--line);padding:12px;overflow:auto;min-height:0}
#view{position:relative;overflow:hidden;background:#000;min-width:0;min-height:0}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair}
h1{font-size:15px;margin:0 0 10px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:16px 0 6px}
button{background:#2a3137;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit;cursor:pointer;text-align:left}
button:hover{border-color:#5a646c}
button.on{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
.row{display:flex;gap:6px;flex-wrap:wrap}
.col{display:flex;flex-direction:column;gap:6px}
.col button{width:100%}
kbd{background:#0d1012;border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px ui-monospace,Consolas,monospace;float:right;color:var(--text)}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px;vertical-align:-1px;border:2px solid}
.sw.sq{border-radius:2px}
#counts div{display:flex;justify-content:space-between}
#sel{background:#0d1012;border:1px solid var(--line);border-radius:6px;padding:8px;min-height:64px}
#status{position:absolute;left:10px;bottom:10px;background:rgba(0,0,0,.72);padding:5px 9px;border-radius:6px;font-size:13px;pointer-events:none}
#saved{position:absolute;right:10px;bottom:10px;max-width:60%;background:rgba(0,0,0,.72);padding:5px 9px;border-radius:6px;font-size:13px;pointer-events:none}
#saved.bad{background:#b3261e;color:#fff;font-weight:600}
label{display:flex;align-items:center;gap:7px;margin:4px 0}
.dim{color:var(--dim)}
p{margin:6px 0}
</style></head><body>
<div id="app">
<div id="side">
  <h1>Board review</h1>
  <div class="row"><button id="bTop" class="on">Top side</button><button id="bBottom">Bottom side</button></div>
  <p class="dim" id="mirror"></p>

  <h2>What a click does</h2>
  <div class="col">
    <button data-mode="select">Select an item <kbd>S</kbd></button>
    <button data-mode="via"><span class="sw" style="border-color:#00ffff"></span>Add a via <kbd>V</kbd></button>
    <button data-mode="testpad"><span class="sw" style="border-color:#ff40ff"></span>Add a test pad <kbd>T</kbd></button>
    <button data-mode="pad"><span class="sw sq" style="border-color:#ff9a3c"></span>Add a solder pad (drag a box) <kbd>A</kbd></button>
    <button data-mode="part"><span class="sw sq" style="border-color:#ffe600"></span>Add a part (drag a box) <kbd>P</kbd></button>
    <button data-mode="delete">Remove items <kbd>D</kbd></button>
    <button data-mode="trace">Follow a trace <kbd>W</kbd></button>
  </div>

  <h2>Trace</h2>
  <div id="trace" class="dim">Choose "Follow a trace", then click a piece of copper.</div>
  <label><input type="checkbox" id="trVias" checked> Go through vias to the other side</label>
  <label><input type="checkbox" id="trEdges" checked> Carry on across 10 mm block edges</label>

  <h2>Selected item</h2>
  <div id="sel" class="dim">Nothing selected.</div>

  <h2>Counts on this side</h2>
  <div id="counts"></div>

  <h2>Show</h2>
  <label><input type="checkbox" id="showRemoved"> Removed items (red, click to bring back)</label>
  <label><input type="checkbox" id="showLabels" checked> Item numbers</label>
  <label><input type="checkbox" id="showKnown" checked> Main parts (U1, J1 ...)</label>
  <label><input type="checkbox" id="showGrid" checked> 10 mm grid</label>
  <label><input type="checkbox" id="showCopper"> Copper map (experimental) <kbd>C</kbd></label>

  <h2>Moving around</h2>
  <p class="dim">Scroll to zoom. Drag to move the board. While adding a solder pad or a part, drag draws the box, so move with the right mouse button or hold <b>Space</b>.</p>
  <div class="row"><button id="bFit">Fit board <kbd>F</kbd></button><button id="bUndo">Undo <kbd>Ctrl+Z</kbd></button></div>
  <p class="dim">Every change is saved straight away.</p>
</div>
<div id="view"><canvas id="cv"></canvas><div id="status"></div><div id="saved">Loading...</div></div>
</div>
<script>
const MARGIN=__MARGIN__, MAPW=__MAPW__, BW=__BW__, BH=__BH__;
const COL={via:'#00ffff',testpad:'#ff40ff',pad:'#ff9a3c',part:'#ffe600'};
const NAME={via:'via',testpad:'test pad',pad:'solder pad',part:'part'};
const BOXY={pad:true,part:true};
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const S={side:'top',mode:'select',auto:[],review:[],known:[],boxes:[],img:{},view:{s:10,x:0,y:0},views:{},sel:null,undo:[],hover:null,drag:null,space:false,version:'0',needFit:true};
const $=id=>document.getElementById(id);
function note(t,bad){ const e=$('saved'); e.textContent=t; e.classList.toggle('bad',!!bad); }

function take(d){
  S.auto=d.auto.map(a=>({id:a.id,side:a.side,kind:a.kind,x:+a.x_mm,y:+a.y_mm,w:+a.w_mm,h:+a.h_mm}));
  S.review=d.review; S.known=d.known; S.boxes=d.boxes; S.version=d.version;
}
async function load(){
  take(await (await fetch('/api/state')).json());
  for(const side of ['top','bottom']){ const im=new Image(); im.onload=()=>draw(); im.src='/map/'+side+'.jpg'; S.img[side]=im; }
  setSide('top'); setMode('select'); note('Loaded');
}

// The list as it stands after the review rows are applied.
function items(){
  const removed=new Set(), retype={};
  S.review.forEach(r=>{ if(r.action==='remove') removed.add(r.id); else if(r.action==='retype') retype[r.id]=r; });
  const out=[];
  for(const a of S.auto){
    const it={key:a.id,auto:true,side:a.side,kind:a.kind,x:a.x,y:a.y,w:a.w,h:a.h,removed:removed.has(a.id),note:''};
    const rt=retype[a.id];
    if(rt){ it.kind=rt.kind; if(rt.w_mm){it.w=+rt.w_mm; it.h=+rt.h_mm;} it.note=rt.note||''; }
    out.push(it);
  }
  S.review.forEach((r,i)=>{ if(r.action==='add') out.push({key:'add'+i,row:i,auto:false,side:r.side,kind:r.kind,x:+r.x_mm,y:+r.y_mm,w:+(r.w_mm||0.6),h:+(r.h_mm||0.6),removed:false,note:r.note||''}); });
  return out;
}
function sideItems(){ return items().filter(i=>i.side===S.side); }

function commit(mut){
  S.undo.push(JSON.stringify(S.review)); if(S.undo.length>200) S.undo.shift();
  mut(); save(); refresh();
}
let saveTimer=null, saving=false, dirty=false;
function save(){ dirty=true; note('Saving...'); clearTimeout(saveTimer); saveTimer=setTimeout(push,300); }
async function push(){
  if(saving){ saveTimer=setTimeout(push,300); return; }
  saving=true; dirty=false;
  try{
    const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base:S.version,rows:S.review})});
    if(r.status===409){
      take(await (await fetch('/api/state')).json()); S.undo=[]; S.sel=null; dirty=false; refresh();
      note('The list was changed outside the editor. It has been reloaded: please redo your last change.',true);
    } else if(r.ok){ S.version=(await r.json()).version; note('Saved '+new Date().toLocaleTimeString()); }
    else note('NOT SAVED (error '+r.status+')',true);
  }catch(e){ note('NOT SAVED: is the tool still running?',true); }
  saving=false; if(dirty) push();
}
function row(action,side,kind,id,x,y,w,h,note){
  const f=v=>v===''||v===undefined||v===null?'':(+v).toFixed(2);
  return {action,side,kind:kind||'',id:id||'',x_mm:f(x),y_mm:f(y),w_mm:f(w),h_mm:f(h),note:note||''};
}
function removeItem(it){
  commit(()=>{
    if(it.auto){ S.review=S.review.filter(r=>!(r.action==='retype'&&r.id===it.key)); S.review.push(row('remove',it.side,'',it.key,'','','','','removed by hand')); }
    else S.review.splice(it.row,1);
  });
  S.sel=null; refresh();
}
function restoreItem(it){ commit(()=>{ S.review=S.review.filter(r=>!(r.action==='remove'&&r.id===it.key)); }); }
function setKind(it,kind){
  const m=Math.max(0.8,Math.min(it.w,it.h));
  const sz = kind==='via'?[0.6,0.6] : kind==='testpad'?[m,m] : [it.w,it.h];
  commit(()=>{
    if(it.auto){ S.review=S.review.filter(r=>!(r.action==='retype'&&r.id===it.key)); S.review.push(row('retype',it.side,kind,it.key,'','',sz[0],sz[1],'changed by hand')); }
    else { const r=S.review[it.row]; r.kind=kind; r.w_mm=sz[0].toFixed(2); r.h_mm=sz[1].toFixed(2); }
  });
}
function addItem(kind,x,y,w,h){
  if(x<0||y<0||x>BW||y>BH){ note('Not added: that spot is off the board',true); return; }
  commit(()=>{ S.review.push(row('add',S.side,kind,'',x,y,w,h,'added by hand')); });
}
function undo(){ if(!S.undo.length) return; S.review=JSON.parse(S.undo.pop()); S.sel=null; save(); refresh(); }

// ---- view
function fit(){ const r=cv.getBoundingClientRect(); if(r.width<50||r.height<50){ S.needFit=true; return; } S.needFit=false;
  const s=Math.min(r.width/(BW+4),r.height/(BH+4)); S.view={s,x:(r.width-BW*s)/2,y:(r.height-BH*s)/2}; draw(); }
const toScr=(x,y)=>[S.view.x+x*S.view.s, S.view.y+y*S.view.s];
const toMm=(px,py)=>[(px-S.view.x)/S.view.s,(py-S.view.y)/S.view.s];
function resize(){ const r=cv.getBoundingClientRect(), d=window.devicePixelRatio||1; cv.width=Math.round(r.width*d); cv.height=Math.round(r.height*d); ctx.setTransform(d,0,0,d,0,0); draw(); }

function draw(){
  const r=cv.getBoundingClientRect(); ctx.clearRect(0,0,r.width,r.height);
  const v=S.view; let im=S.img[S.side];
  if($('showCopper').checked){ if(!S.cop) S.cop={}; if(!S.cop[S.side]){ const c=new Image(); c.onload=()=>draw(); c.src='/copper/'+S.side+'.jpg'; S.cop[S.side]=c; } const c=S.cop[S.side]; if(c.complete&&c.naturalWidth) im=c; }
  if(im&&im.complete&&im.naturalWidth){ const mmH=MAPW*im.naturalHeight/im.naturalWidth; ctx.imageSmoothingEnabled = v.s < im.naturalWidth/MAPW*1.5;
    ctx.drawImage(im, v.x-MARGIN*v.s, v.y-MARGIN*v.s, MAPW*v.s, mmH*v.s); }
  const ov=S.trace&&S.trace.img[S.side];
  if(ov&&ov.complete&&ov.naturalWidth){ ctx.imageSmoothingEnabled=false; ctx.drawImage(ov, v.x-MARGIN*v.s, v.y-MARGIN*v.s, MAPW*v.s, MAPW*ov.naturalHeight/ov.naturalWidth*v.s); }
  if($('showGrid').checked){ ctx.lineWidth=1; ctx.strokeStyle='rgba(255,255,255,.28)'; ctx.fillStyle='rgba(255,255,255,.75)'; ctx.font='11px system-ui';
    for(let x=0;x<=BW;x+=10){ const [a,b]=toScr(x,0), [,d]=toScr(x,BH); ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(a,d);ctx.stroke(); ctx.fillText(x,a+3,b-4); }
    for(let y=0;y<=BH;y+=10){ const [a,b]=toScr(0,y), [c]=toScr(BW,y); ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,b);ctx.stroke(); ctx.fillText(y,a-22,b+4); } }
  if($('showKnown').checked){ ctx.lineWidth=1.5; ctx.font='bold 12px system-ui';
    const bx=(k,label,col)=>{ const [a,b]=toScr(+k.x0_mm,+k.y0_mm),[c,d]=toScr(+k.x1_mm,+k.y1_mm); ctx.strokeStyle=col; ctx.strokeRect(a,b,c-a,d-b); ctx.fillStyle=col; ctx.fillText(label,a+3,b+13); };
    S.known.filter(k=>k.side===S.side).forEach(k=>bx(k,k.ref,'rgba(255,255,255,.8)'));
    if(S.side==='top') S.boxes.forEach(k=>bx(k,k.name,'rgba(255,255,255,.6)')); }
  const lab=$('showLabels').checked && v.s>14, showRem=$('showRemoved').checked;
  ctx.font='11px system-ui';
  for(const it of sideItems()){
    if(it.removed && !showRem) continue;
    const [cx,cy]=toScr(it.x,it.y), hw=it.w/2*v.s+2, hh=it.h/2*v.s+2;
    if(cx<-80||cy<-80||cx>r.width+80||cy>r.height+80) continue;
    const sel=S.sel===it.key, hov=S.hover===it.key;
    ctx.lineWidth=sel?3.5:(hov?2.8:1.8); ctx.setLineDash(it.removed?[5,4]:[]);
    ctx.strokeStyle=it.removed?'#ff3b30':(sel?'#ffffff':COL[it.kind]||'#fff');
    ctx.beginPath(); if(BOXY[it.kind]) ctx.rect(cx-hw,cy-hh,hw*2,hh*2); else ctx.ellipse(cx,cy,Math.max(hw,4),Math.max(hh,4),0,0,Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
    if(lab||sel||hov){ const t=it.auto?it.key:'added'; ctx.lineWidth=3; ctx.strokeStyle='#000'; ctx.strokeText(t,cx+hw+2,cy-hh+2); ctx.fillStyle=it.removed?'#ff8a80':(COL[it.kind]||'#fff'); ctx.fillText(t,cx+hw+2,cy-hh+2); }
  }
  if(S.drag&&S.drag.type==='box'){ const [a,b]=toScr(S.drag.x0,S.drag.y0),[c,d]=toScr(S.drag.x1,S.drag.y1); ctx.strokeStyle=COL[S.mode]||'#ffe600'; ctx.lineWidth=2; ctx.setLineDash([4,3]); ctx.strokeRect(a,b,c-a,d-b); ctx.setLineDash([]); }
}

function hit(mx,my){
  let best=null,bd=1e9; const showRem=$('showRemoved').checked;
  for(const it of sideItems()){
    if(it.removed&&!showRem) continue;
    const dx=Math.abs(mx-it.x), dy=Math.abs(my-it.y), pad=Math.max(0.25,6/S.view.s);
    if(dx<=it.w/2+pad && dy<=it.h/2+pad){ const d=Math.hypot(dx,dy)+(it.kind==='part'?0.3:0); if(d<bd){bd=d;best=it;} }
  }
  return best;
}

function refresh(){
  const all=sideItems(), its=all.filter(i=>!i.removed);
  const n=k=>its.filter(i=>i.kind===k).length;
  $('counts').innerHTML=`<div><span>Vias</span><b>${n('via')}</b></div><div><span>Test pads</span><b>${n('testpad')}</b></div><div><span>Solder pads</span><b>${n('pad')}</b></div><div><span>Parts</span><b>${n('part')}</b></div><div class="dim"><span>Added by hand</span><span>${its.filter(i=>!i.auto).length}</span></div><div class="dim"><span>Removed</span><span>${all.length-its.length}</span></div>`;
  const it=all.find(i=>i.key===S.sel);
  if(!it){ $('sel').className='dim'; $('sel').textContent='Nothing selected.'; }
  else{
    $('sel').className='';
    const others=['via','testpad','pad','part'].filter(k=>k!==it.kind);
    $('sel').innerHTML=`<p><b>${it.auto?it.key:'Added by hand'}</b>: ${NAME[it.kind]||it.kind}${it.removed?' <span style="color:#ff8a80">(removed)</span>':''}</p>
      <p class="dim">at ${it.x.toFixed(2)}, ${it.y.toFixed(2)} mm, ${it.w.toFixed(1)} x ${it.h.toFixed(1)} mm${it.note?'<br>'+it.note.replace(/</g,'&lt;'):''}</p>
      <div class="col">${it.removed?'<button id="aRestore">Bring it back</button>':
        '<button id="aDel">Remove it <kbd>Del</kbd></button>'+others.map(k=>`<button data-k="${k}">Change to a ${NAME[k]}</button>`).join('')}</div>`;
    if($('aRestore')) $('aRestore').onclick=()=>restoreItem(it);
    if($('aDel')) $('aDel').onclick=()=>removeItem(it);
    $('sel').querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>setKind(it,b.dataset.k));
  }
  draw();
}
function setSide(s){
  if(!S.needFit) S.views[S.side]={...S.view};
  S.side=s; S.sel=null; $('bTop').classList.toggle('on',s==='top'); $('bBottom').classList.toggle('on',s==='bottom');
  $('mirror').textContent = s==='bottom' ? 'The bottom side is shown mirrored, so it lines up with the top side.' : '';
  if(S.views[s]){ S.view={...S.views[s]}; } else fit();
  refresh();
}
function setMode(m){ S.mode=m; document.querySelectorAll('button[data-mode]').forEach(b=>b.classList.toggle('on',b.dataset.mode===m)); cv.style.cursor = m==='select'?'default':(m==='delete'?'not-allowed':(m==='trace'?'pointer':'crosshair')); }

// ---- mouse and keys
function pos(e){ const r=cv.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top]; }
function clickAt(mx,my){
  const it=hit(mx,my);
  if(S.mode==='select'){ S.sel=it?it.key:null; refresh(); }
  else if(S.mode==='delete'){ if(it){ it.removed?restoreItem(it):removeItem(it); } }
  else if(S.mode==='via'){ addItem('via',mx,my,0.6,0.6); }
  else if(S.mode==='testpad'){ addItem('testpad',mx,my,1.0,1.0); }
  else if(S.mode==='trace'){ followTrace(mx,my); }
}
function clearTrace(){ S.trace=null; $('trace').className='dim'; $('trace').textContent='Choose "Follow a trace", then click a piece of copper.'; draw(); }
async function followTrace(mx,my,rebuild){
  const t=$('trace'); t.className=''; t.textContent = S.traceReady ? 'Following...' : 'Preparing the trace data. The first time takes about 15 seconds...';
  S.lastTrace=[S.side,mx,my];
  let d;
  try{
    const r=await fetch(`/api/trace?side=${S.side}&x=${mx.toFixed(3)}&y=${my.toFixed(3)}&vias=${$('trVias').checked?1:0}&edges=${$('trEdges').checked?1:0}${rebuild?'&rebuild=1':''}`);
    d=await r.json();
  }catch(e){ t.textContent='Could not follow the trace: is the tool still running?'; return; }
  if(!d.ok){ S.trace=null; t.textContent=d.reason; draw(); return; }
  S.traceReady=true;
  const img={}; for(const s of ['top','bottom']){ if(d[s]){ const im=new Image(); im.onload=()=>draw(); im.src=d[s]; img[s]=im; } }
  S.trace={img};
  const other=S.side==='top'?'bottom':'top';
  const pads=d.pads.length? d.pads.map(p=>p.replace(/</g,'&lt;')).join('<br>') : 'no chip pin or test pad';
  t.innerHTML=`<p><b>${d.islands}</b> piece${d.islands===1?'':'s'} of copper, joined through <b>${d.vias}</b> via${d.vias===1?'':'s'}${d[other]?' (continues on the '+other+' side)':''}.</p>
    <p class="dim">Touches:</p><p style="max-height:170px;overflow:auto">${pads}</p>
    <p class="dim">This shows what the picture suggests, not a measurement. Trace data built at ${d.built}.</p>
    <div class="row"><button id="trClear">Clear <kbd>Esc</kbd></button><button id="trRebuild">Rebuild data</button></div>`;
  $('trClear').onclick=clearTrace;
  $('trRebuild').onclick=()=>{ S.traceReady=false; followTrace(mx,my,true); };
  draw();
}
document.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{
  const [px,py]=pos(e), [mx,my]=toMm(px,py);
  const pan={type:'pan',px,py,vx:S.view.x,vy:S.view.y,moved:false,click:false};
  if(e.button===2||e.button===1||S.space){ S.drag=pan; e.preventDefault(); return; }
  if(e.button!==0) return;
  if(BOXY[S.mode]){ S.drag={type:'box',x0:mx,y0:my,x1:mx,y1:my}; }
  else { pan.click=true; S.drag=pan; }      // a press that does not move is a click; one that moves drags the board
});
window.addEventListener('mousemove',e=>{
  const [px,py]=pos(e), [mx,my]=toMm(px,py);
  $('status').textContent = `x ${mx.toFixed(2)} mm   y ${my.toFixed(2)} mm   block c${Math.floor(mx/10)} r${Math.floor(my/10)}`;
  if(S.drag && e.buttons===0){ S.drag=null; draw(); return; }          // the button was let go somewhere we could not see
  if(S.drag&&S.drag.type==='pan'){ if(Math.hypot(px-S.drag.px,py-S.drag.py)>4) S.drag.moved=true;
    if(S.drag.moved){ S.view.x=S.drag.vx+px-S.drag.px; S.view.y=S.drag.vy+py-S.drag.py; draw(); } return; }
  if(S.drag&&S.drag.type==='box'){ S.drag.x1=mx; S.drag.y1=my; draw(); return; }
  const it=hit(mx,my), k=it?it.key:null; if(k!==S.hover){ S.hover=k; draw(); }
});
window.addEventListener('mouseup',e=>{
  const d=S.drag; S.drag=null; if(!d) return;
  if(d.type==='pan'&&d.click&&!d.moved){ const [mx,my]=toMm(d.px,d.py); clickAt(mx,my); }
  if(d.type==='box'){ const w=Math.abs(d.x1-d.x0), h=Math.abs(d.y1-d.y0);
    if(w>=0.2&&h>=0.2) addItem(S.mode,(d.x0+d.x1)/2,(d.y0+d.y1)/2,w,h); else draw(); }
});
window.addEventListener('blur',()=>{ S.drag=null; S.space=false; draw(); });
cv.addEventListener('wheel',e=>{ e.preventDefault(); const [px,py]=pos(e), [mx,my]=toMm(px,py); const f=Math.exp(-e.deltaY*0.0015);
  S.view.s=Math.max(3,Math.min(400,S.view.s*f)); S.view.x=px-mx*S.view.s; S.view.y=py-my*S.view.s; draw(); },{passive:false});
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.code==='Space'){ S.space=true; e.preventDefault(); return; }
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){ e.preventDefault(); undo(); return; }
  if(e.ctrlKey||e.metaKey||e.altKey) return;
  const k=e.key.toLowerCase();
  if(k==='s') setMode('select'); else if(k==='v') setMode('via'); else if(k==='t') setMode('testpad'); else if(k==='a') setMode('pad'); else if(k==='p') setMode('part'); else if(k==='d') setMode('delete'); else if(k==='w') setMode('trace');
  else if(k==='c'){ $('showCopper').checked=!$('showCopper').checked; draw(); }
  else if(k==='f') fit(); else if(k==='escape'){ S.sel=null; S.drag=null; if(S.trace) clearTrace(); setMode('select'); refresh(); }
  else if(k==='delete'||k==='backspace'){ const it=sideItems().find(i=>i.key===S.sel); if(it&&!it.removed) removeItem(it); }
});
window.addEventListener('keyup',e=>{ if(e.code==='Space'){ S.space=false; e.preventDefault(); } });
// buttons give up keyboard focus after a click, so Space and Enter never press them again by accident
document.addEventListener('click',e=>{ const b=e.target.closest('button'); if(b) b.blur(); });
document.querySelectorAll('button[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
$('bTop').onclick=()=>setSide('top'); $('bBottom').onclick=()=>setSide('bottom'); $('bFit').onclick=fit; $('bUndo').onclick=undo;
['showRemoved','showLabels','showKnown','showGrid','showCopper'].forEach(id=>$(id).onchange=draw);
['trVias','trEdges'].forEach(id=>$(id).onchange=()=>{ if(S.trace&&S.lastTrace&&S.lastTrace[0]===S.side) followTrace(S.lastTrace[1],S.lastTrace[2]); });
window.addEventListener('resize',resize);
new ResizeObserver(()=>{ resize(); if(S.needFit) fit(); }).observe($('view'));
window.addEventListener('beforeunload',e=>{ if(dirty||saving){ e.preventDefault(); e.returnValue=''; } });
resize(); load();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(PCB, "features-auto.csv")):
        sys.exit("data/features-auto.csv is missing: run  find_features.py --detect  first")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = "http://localhost:%d/" % a.port
    print("Board review tool running at %s   (Ctrl+C to stop)" % url, flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
