#!/usr/bin/env python3
"""Generate a self-contained index.html for GitHub Pages.

Shows, for the three confirmed binders:
  1. pLDDT-coloured structures of each binder vs its highest-PAE partner
  2. TCR-peptide and TCR-MHC binding interactions (bonds coloured by type)

All geometry is pre-computed here and baked into the page as 3Dmol.js calls,
so the page is fully static (no Python/Jupyter needed to view it).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
STRUCT = BASE / "structures"
PLDDT = BASE / "plddt"
EXPORT = BASE / "results" / "binder_structures_export"
RESULTS_TSV = BASE / "data" / "cancer_output_w_pae.tsv"
OLD_CSV = BASE / "data" / "combined_df_merged_stats_pyrosetta_previous_runApr2025.csv"

# ---------------------------------------------------------------- load binders
res = pd.read_table(RESULTS_TSV)
old = pd.read_csv(OLD_CSV)
key_cols = ["mhc", "peptide", "va", "ja", "cdr3a", "vb", "jb", "cdr3b"]
seqkey = lambda d: d[key_cols].astype(str).agg("|".join, axis=1)
res["seqkey"] = seqkey(res); old["seqkey"] = seqkey(old)
res = res.merge(old[["seqkey", "query", "motif"]].drop_duplicates("seqkey"),
                on="seqkey", how="left").rename(columns={"motif": "tcr_group"})
binders = res[res["query"] == 1].sort_values("pmhc_tcr_pae").reset_index(drop=True)


def find_pdb(targetid):
    return list(STRUCT.glob(f"*{targetid}*model_2_ptm_ft4.pdb"))[0]


def find_plddt(targetid):
    return list(PLDDT.glob(f"*{targetid}*_plddt.npy"))[0]


def chain_lengths(row):
    return [len(x) for x in row["target_chainseq"].split("/")]


# --------------------------------------------------- interaction detection code
POS_RES = {"ARG", "LYS", "HIS"}
NEG_RES = {"ASP", "GLU"}
POS_ATOMS = {"NZ", "NH1", "NH2", "NE", "ND1", "NE2"}
NEG_ATOMS = {"OD1", "OD2", "OE1", "OE2"}
DONOR_ACCEPTOR = lambda el: el in ("N", "O")
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS"}
SALT_CUT, HBOND_CUT, DISULF_CUT, VDW_CUT = 4.0, 3.5, 2.5, 4.0


def parse_atoms(targetid, seg_lengths):
    bounds = np.cumsum(seg_lengths)

    def seg_of(o):
        if o < bounds[0]: return "MHC"
        if o < bounds[1]: return "peptide"
        if o < bounds[2]: return "TCRa"
        return "TCRb"

    atoms, ridx, prev = [], -1, None
    for line in open(find_pdb(targetid)):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[21], line[22:27])
        if key != prev:
            ridx += 1; prev = key
        atoms.append({
            "name": line[12:16].strip(), "resn": line[17:20].strip(),
            "resi": line[22:26].strip(),
            "elem": (line[76:78].strip() or line[12:16].strip()[0]),
            "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            "seg": seg_of(ridx),
        })
    return atoms


def find_interactions(atoms, target):
    if isinstance(target, str):
        target = (target,)
    tgt = [a for a in atoms if a["seg"] in target]
    tcr = [a for a in atoms if a["seg"] in ("TCRa", "TCRb")]
    out = {"salt": [], "hbond": [], "disulfide": [], "hydrophobic": []}
    if not tgt:
        return out
    G = np.array([a["xyz"] for a in tgt]); T = np.array([a["xyz"] for a in tcr])
    hyd = {}
    for j, ta in enumerate(tcr):
        d = np.linalg.norm(G - T[j], axis=1)
        for i in np.where(d <= VDW_CUT + 0.1)[0]:
            pa = tgt[i]; dist = float(d[i])
            if ta["name"] == "SG" and pa["name"] == "SG" and dist <= DISULF_CUT:
                out["disulfide"].append((ta, pa, dist)); continue
            is_salt = ((ta["resn"] in POS_RES and ta["name"] in POS_ATOMS and
                        pa["resn"] in NEG_RES and pa["name"] in NEG_ATOMS) or
                       (ta["resn"] in NEG_RES and ta["name"] in NEG_ATOMS and
                        pa["resn"] in POS_RES and pa["name"] in POS_ATOMS))
            if is_salt and dist <= SALT_CUT:
                out["salt"].append((ta, pa, dist)); continue
            if DONOR_ACCEPTOR(ta["elem"]) and DONOR_ACCEPTOR(pa["elem"]) and dist <= HBOND_CUT:
                out["hbond"].append((ta, pa, dist)); continue
            if (ta["elem"] == "C" and pa["elem"] == "C" and dist <= VDW_CUT
                    and ta["resn"] in HYDROPHOBIC and pa["resn"] in HYDROPHOBIC):
                pk = (ta["resi"], pa["resi"])
                if pk not in hyd or dist < hyd[pk][2]:
                    hyd[pk] = (ta, pa, dist)
    out["hydrophobic"] = list(hyd.values())
    return out


def seg_ranges(atoms):
    out = {}
    for a in atoms:
        out.setdefault(a["seg"], set()).add(a["resi"])
    return out


SEG_COLORS = {"MHC": "#BDBDBD", "peptide": "#FF8C00", "TCRa": "#8CB3FF", "TCRb": "#9DDF9D"}
BOND_STYLE = {
    "salt": (0.14, "#E000E0"), "hbond": (0.06, "#E6C700"),
    "disulfide": (0.18, "#FFD000"), "hydrophobic": (0.05, "#9E9E9E"),
}

# --------------------------------------------------------- build partner panel
# For each binder pick its highest-PAE same-peptide partner, deduplicating so
# every partner is a genuinely different TCR (matches the notebook).
pairs = []
used_partners = set()
for _, b in binders.iterrows():
    pep = b["peptide"]
    same = (res[(res["peptide"] == pep) & (res["targetid"] != b["targetid"])]
            .sort_values("pmhc_tcr_pae", ascending=False))
    w = None
    for _, cand in same.iterrows():
        if cand["tcr_group"] not in used_partners:
            w = cand; break
    if w is None:
        w = same.iloc[0]
    used_partners.add(w["tcr_group"])
    pairs.append(("binder", b["targetid"], b["tcr_group"], pep, float(b["pmhc_tcr_pae"])))
    pairs.append(("high-PAE partner", w["targetid"], w["tcr_group"], pep, float(w["pmhc_tcr_pae"])))


def pdb_with_plddt(targetid):
    plddt = np.load(find_plddt(targetid))
    out, ridx, prev = [], -1, None
    for line in open(find_pdb(targetid)):
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:27])
            if key != prev:
                ridx += 1; prev = key
            val = float(plddt[ridx]) if ridx < len(plddt) else 0.0
            line = line[:60] + f"{val:6.2f}" + line[66:]
        out.append(line)
    return "".join(out)


# --------------------------------------------------------------- emit JS blocks
def js_pdb(text):
    return json.dumps(text)


plddt_panels = []
for role, tid, grp, pep, pae in pairs:
    plddt_panels.append({
        "pdb": pdb_with_plddt(tid),
        "title": f"{grp} · {pep} · {role} · PAE {pae:.2f}",
    })

inter_panels = []
for _, r in binders.iterrows():
    tid = r["targetid"]
    atoms = parse_atoms(tid, chain_lengths(r))
    segs = seg_ranges(atoms)
    pdb_text = open(find_pdb(tid)).read()
    seg_res = {s: sorted(v, key=int) for s, v in segs.items()}
    targets = {}
    for target in ("peptide", "MHC"):
        ix = find_interactions(atoms, target)
        allbonds = [(k, t) for k in ix for t in ix[k]]
        iface = sorted({a["resi"] for _, (ta, pa, d) in allbonds for a in (ta, pa)}, key=int)
        bonds = []
        for kind, (ta, pa, d) in allbonds:
            radius, color = BOND_STYLE[kind]
            bonds.append({
                "s": [float(x) for x in ta["xyz"]], "e": [float(x) for x in pa["xyz"]],
                "r": radius, "c": color, "dashed": kind != "disulfide",
            })
        targets[target] = {
            "iface": iface, "bonds": bonds,
            "counts": {k: len(v) for k, v in ix.items()},
        }
    inter_panels.append({
        "pdb": pdb_text, "seg_res": seg_res, "targets": targets,
        "title": f"{r['tcr_group']} · {r['peptide']}",
    })

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TCRdock cancer TCR:pMHC — structures & interactions</title>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js"></script>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#fafafa}
  header{background:#0d3b66;color:#fff;padding:22px 28px}
  header h1{margin:0;font-size:20px}
  header p{margin:6px 0 0;font-size:13px;opacity:.85}
  header p.authors{font-size:13.5px;font-style:italic;opacity:.95;margin-top:8px}
  main{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
  h2{font-size:17px;border-bottom:2px solid #0d3b66;padding-bottom:6px;margin-top:34px}
  .legend{font-size:12.5px;background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:10px 14px;margin:12px 0}
  .legend b{display:inline-block;margin-right:6px}
  .chip{display:inline-block;width:13px;height:13px;border:1px solid #999;vertical-align:middle;margin:0 4px 0 12px;border-radius:2px}
  .grad{display:inline-block;width:120px;height:13px;vertical-align:middle;margin:0 6px;border:1px solid #999;
        background:linear-gradient(90deg,#c0392b,#ffffff,#2657b0);border-radius:2px}
  .grid{display:grid;gap:16px}
  .grid.two{grid-template-columns:repeat(2,1fr)}
  .grid.three{grid-template-columns:repeat(3,1fr)}
  .cell{background:#fff;border:1px solid #e2e2e2;border-radius:8px;overflow:hidden}
  .cell .cap{font-size:12px;padding:7px 10px;border-bottom:1px solid #eee;background:#f6f8fb}
  .cell .cap small{color:#666}
  .viewer{position:relative;width:100%;height:330px}
  .toggle{display:flex;gap:6px;padding:7px 10px;border-bottom:1px solid #eee;background:#fff}
  .toggle button{flex:1;font-size:11.5px;padding:5px 4px;border:1px solid #cfd8e3;background:#fff;
                 color:#0d3b66;border-radius:5px;cursor:pointer;transition:all .12s}
  .toggle button:hover{background:#eef3fa}
  .toggle button.active{background:#0d3b66;color:#fff;border-color:#0d3b66}
  .availability{background:#eef4fb;border:1px solid #cfe0f2;border-left:5px solid #0d3b66;
    border-radius:8px;padding:18px 22px;margin:20px 0 8px;font-size:19px;line-height:1.5;color:#123}
  .availability a{color:#0d3b66;font-weight:600;word-break:break-all}
  @media(max-width:820px){.grid.two,.grid.three{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Development of anti-NPM1c-specific T-cell receptors using a humanized TCR mouse model</h1>
  <p class="authors">Xiaoxuan Liu, Qunwei Wang, Yizhou Yu&hellip;, Sarah A. Teichmann, Allan Bradley and George S. Vassiliou</p>
  <p>Predicted TCR:pMHC structures — three confirmed binders vs high-PAE partners. Drag to rotate · scroll to zoom.</p>
</header>
<main>

<div class="availability">
  Data, analysis scripts and predicted structures are available at
  <a href="https://github.com/izu0421/NPM1c_TCRpMHC" target="_blank" rel="noopener">github.com/izu0421/NPM1c_TCRpMHC</a>
</div>

<h2>1 · Model confidence (pLDDT)</h2>
<div class="legend">
  <b>pLDDT</b> low <span class="grad"></span> high
  &nbsp;(red 50 → white 70 → blue 90). Left column = binder, right column = highest-PAE partner for the same peptide.
</div>
<div class="grid two" id="plddtGrid"></div>

<h2>2 · TCR binding interactions</h2>
<div class="legend">
  <b>Chains</b>
  <span class="chip" style="background:#8CB3FF"></span>TCR&alpha;
  <span class="chip" style="background:#9DDF9D"></span>TCR&beta;
  <span class="chip" style="background:#FF8C00"></span>peptide
  <span class="chip" style="background:#BDBDBD"></span>MHC
  &nbsp;|&nbsp; <b>Bonds</b>
  <span class="chip" style="background:#E000E0"></span>salt bridge
  <span class="chip" style="background:#E6C700"></span>H-bond
  <span class="chip" style="background:#FFD000"></span>disulfide
  <span class="chip" style="background:#9E9E9E"></span>hydrophobic
  <br><span style="color:#555">Use the buttons on each panel to show TCR–peptide, TCR–MHC, or both sets of interactions.</span>
</div>
<div class="grid three" id="interGrid"></div>

</main>
<script>
const PLDDT = __PLDDT__;
const INTER = __INTER__;
const SEG_COLORS = __SEGCOLORS__;

function makeCell(parent, title, sub){
  const cell = document.createElement('div'); cell.className='cell';
  const cap = document.createElement('div'); cap.className='cap';
  cap.innerHTML = title + (sub ? ' <small>'+sub+'</small>' : '');
  const vd = document.createElement('div'); vd.className='viewer';
  cell.appendChild(cap); cell.appendChild(vd); parent.appendChild(cell);
  return vd;
}

// --- pLDDT panels
const pg = document.getElementById('plddtGrid');
PLDDT.forEach(p=>{
  const vd = makeCell(pg, p.title, '');
  const v = $3Dmol.createViewer(vd,{backgroundColor:'white'});
  v.addModel(p.pdb,'pdb');
  v.setStyle({},{cartoon:{colorscheme:{prop:'b',gradient:'rwb',min:50,max:90}}});
  v.zoomTo(); v.render();
});

// --- interaction panels (with TCR-peptide / TCR-MHC / both toggle)
const ig = document.getElementById('interGrid');
function fmtCounts(t){
  const c = t.counts;
  return 'SB '+c.salt+' · HB '+c.hbond+' · SS '+c.disulfide+' · H\u03a6 '+c.hydrophobic;
}
INTER.forEach(p=>{
  // build cell: caption + toggle buttons + viewer
  const cell = document.createElement('div'); cell.className='cell';
  const cap = document.createElement('div'); cap.className='cap';
  const sub = document.createElement('small');
  cap.innerHTML = p.title + ' ';
  cap.appendChild(sub);
  const bar = document.createElement('div'); bar.className='toggle';
  const vd = document.createElement('div'); vd.className='viewer';
  cell.appendChild(cap); cell.appendChild(bar); cell.appendChild(vd);
  ig.appendChild(cell);

  const v = $3Dmol.createViewer(vd,{backgroundColor:'white'});
  v.addModel(p.pdb,'pdb');

  function draw(mode){  // mode: 'peptide' | 'MHC' | 'both'
    const shown = mode==='both' ? ['peptide','MHC'] : [mode];
    v.removeAllShapes();
    v.setStyle({},{cartoon:{color:'white'}});
    for(const seg in p.seg_res){
      v.setStyle({resi:p.seg_res[seg]},{cartoon:{color:SEG_COLORS[seg]}});
    }
    let iface=[], bonds=[], counts={salt:0,hbond:0,disulfide:0,hydrophobic:0};
    shown.forEach(tg=>{
      const t=p.targets[tg];
      iface=iface.concat(t.iface); bonds=bonds.concat(t.bonds);
      for(const k in counts) counts[k]+=t.counts[k];
    });
    if(iface.length) v.setStyle({resi:iface},{cartoon:{},stick:{radius:0.15}});
    bonds.forEach(b=>{
      v.addCylinder({start:{x:b.s[0],y:b.s[1],z:b.s[2]},end:{x:b.e[0],y:b.e[1],z:b.e[2]},
        dashed:b.dashed,fromCap:1,toCap:1,radius:b.r,color:b.c});
    });
    // zoom on the shown target region(s)
    let zres=[]; shown.forEach(tg=>{ zres=zres.concat(p.seg_res[tg]||[]); });
    v.zoomTo(zres.length?{resi:zres}:{});
    v.render();
    sub.textContent = fmtCounts({counts});
  }

  const modes=[['peptide','TCR–peptide'],['MHC','TCR–MHC'],['both','Both']];
  const btns={};
  modes.forEach(([m,label])=>{
    const b=document.createElement('button'); b.textContent=label;
    b.onclick=()=>{ draw(m); for(const k in btns) btns[k].classList.toggle('active',k===m); };
    bar.appendChild(b); btns[m]=b;
  });
  btns.peptide.classList.add('active');
  draw('peptide');
});
</script>
</body>
</html>
"""

html = (HTML
        .replace("__PLDDT__", json.dumps(plddt_panels))
        .replace("__INTER__", json.dumps(inter_panels))
        .replace("__SEGCOLORS__", json.dumps(SEG_COLORS)))

(BASE / "index.html").write_text(html)
print("wrote", BASE / "index.html")
print("pLDDT panels:", len(plddt_panels), "| interaction panels:", len(inter_panels))
