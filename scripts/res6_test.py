import sys, hashlib, json
from pathlib import Path
import numpy as np, pycountry
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt_loader import make_tt_loader
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
_C = Path(__file__).resolve().parent.parent/"travel_time_cache"

def cache_matrix(iso, res, locs):
    ll=[(a,b) for a,b,_ in locs]; h=hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p=_C/f"{iso}_res{res}_2h_{h}_driving.npz"
    if not p.exists(): return None
    m=np.load(p)["matrix"]
    return m if np.isfinite(m).any() else None

def cov60(gdf, mat):
    nm = mat.min(axis=1); nm[~np.isfinite(nm)]=np.inf
    pop = gdf["population"].to_numpy(float)
    return pop[nm<=60].sum()/pop.sum()

loader6 = make_tt_loader(6, max_minutes=120)
print(f"{'Country':13} {'res5 60min':>10} {'res6 60min':>10} {'diff':>6}")
for name in ["New Zealand","Kenya","Norway","Portugal"]:
    obj = pycountry.countries.lookup(name)
    g5 = load_population_at_resolution(obj.name, target_resolution=5)
    locs,_ = load_linacs_from_dirac_db(obj.name)
    m5 = cache_matrix(obj.alpha_3, 5, locs)
    g6 = load_population_at_resolution(obj.name, target_resolution=6)
    m6 = loader6(obj.name, obj.alpha_3, g6)   # fetches res-6 TT
    if m5 is None or m6 is None:
        print(f"{name:13} {'m5?' if m5 is None else '':>10} {'m6?' if m6 is None else '':>10}")
        continue
    c5, c6 = cov60(g5,m5), cov60(g6,m6)
    print(f"{name:13} {c5:>9.1%} {c6:>9.1%} {(c6-c5)*100:>+5.1f}", flush=True)
