import sys, hashlib, json
from pathlib import Path
import numpy as np, pycountry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.regions import _AFRICA
_C = Path(__file__).resolve().parent.parent/"travel_time_cache"
def nm(iso,locs):
    ll=[(a,b) for a,b,_ in locs]; h=hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p=_C/f"{iso}_res5_2h_{h}_driving.npz"
    if not p.exists(): return None
    m=np.load(p)["matrix"]
    if not np.isfinite(m).any(): return None
    v=m.min(axis=1); v[~np.isfinite(v)]=np.inf; return v
def cov(name, thr):
    obj=pycountry.countries.lookup(name)
    g=load_population_at_resolution(obj.name,target_resolution=5); locs,_=load_linacs_from_dirac_db(obj.name)
    v=nm(obj.alpha_3,locs); pop=g["population"].to_numpy(float)
    if v is None: return None,pop.sum()
    return pop[v<=thr].sum(), pop.sum()
# Single countries
for name,thr in [("Canada",120),("United Kingdom",45),("United States",60)]:
    c,t=cov(name,thr); print(f"{name:16} {thr:>3} min: {c/t:.1%}")
# Sub-Saharan Africa = Africa minus North Africa (DZ,EG,LY,MA,TN)
north={"DZ","EG","LY","MA","TN"}
ssa=[a for a in _AFRICA if a not in north]
cc=tt=0.0; miss=[]
for a2 in ssa:
    obj=pycountry.countries.get(alpha_2=a2)
    if not obj: continue
    try:
        g=load_population_at_resolution(obj.name,target_resolution=5); locs,_=load_linacs_from_dirac_db(obj.name)
    except: 
        # no facilities -> counts as 0 access but still population
        try: g=load_population_at_resolution(obj.name,target_resolution=5); tt+=g["population"].sum()
        except: pass
        miss.append(a2); continue
    v=nm(obj.alpha_3,locs); pop=g["population"].to_numpy(float); tt+=pop.sum()
    if v is not None: cc+=pop[v<=120].sum()
    else: miss.append(a2)
print(f"Sub-Saharan Africa 120 min: {cc/tt:.1%}  (over {len(ssa)} countries, {len(miss)} no-facility/uncached counted as 0)")
