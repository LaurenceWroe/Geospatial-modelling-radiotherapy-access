import sys, hashlib, json
from pathlib import Path
import numpy as np, pycountry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from data.regions import get_region
_C = Path(__file__).resolve().parent.parent / "travel_time_cache"
_AGG = {"all cancers","all cancers excl. nmsc","all cancers excl nmsc"}
def dem(cg):
    cols=[c for c in cg.columns if c.endswith("_optimal_rt") and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cg[cols].sum(axis=1).clip(lower=0).to_numpy(float) if cols else None
def nm(iso,locs):
    ll=[(a,b) for a,b,_ in locs]; h=hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p=_C/f"{iso}_res5_2h_{h}_driving.npz"
    if not p.exists(): return None
    m=np.load(p)["matrix"]; 
    if not np.isfinite(m).any(): return None
    v=m.min(axis=1); v[~np.isfinite(v)]=np.inf; return v
acc={t:dict(p=0,pa=0,d=0,da=0) for t in (30,120)}
allc=get_cancer_types()+DERIVED_CANCER_TYPES
for a2 in get_region("World").member_alpha2:
    o=pycountry.countries.get(alpha_2=a2)
    if not o: continue
    try:
        g=load_population_at_resolution(o.name,target_resolution=5)
        cg=apportion_cancer_to_h3(g,o.alpha_3,allc,use_actual_rt=False, rtu_set="barton_2014")
    except: continue
    d=dem(cg)
    if d is None: continue
    try: locs,_=load_linacs_from_dirac_db(o.name)
    except: locs=[]
    if not locs: continue
    v=nm(o.alpha_3,locs)
    if v is None: continue
    pop=g["population"].to_numpy(float)
    for t in (30,120):
        m=(v<=t)
        acc[t]["p"]+=pop.sum(); acc[t]["pa"]+=pop[m].sum()
        acc[t]["d"]+=d.sum(); acc[t]["da"]+=d[m].sum()
for t in (30,120):
    a=acc[t]
    print(f"{t} min: geo access POP-weighted (A_G) = {a['pa']/a['p']:.1%}   DEMAND-weighted = {a['da']/a['d']:.1%}")
