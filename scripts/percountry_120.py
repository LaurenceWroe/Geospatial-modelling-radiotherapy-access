import sys, hashlib, json, math
from pathlib import Path
import numpy as np, pycountry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import h3
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
_C = Path(__file__).resolve().parent.parent/"travel_time_cache"; _R=6371.0088
PROXY_KM_120=64.4
def nk(gdf,locs):
    la=np.radians([h3.cell_to_latlng(h)[0] for h in gdf["h3"]]); lo=np.radians([h3.cell_to_latlng(h)[1] for h in gdf["h3"]]); cl=np.cos(la)
    d=np.full(len(gdf),np.inf)
    for lat,lon,_ in locs:
        p,l=math.radians(lat),math.radians(lon)
        a=np.sin((la-p)/2)**2+math.cos(p)*cl*np.sin((lo-l)/2)**2
        np.minimum(d,2*_R*np.arcsin(np.sqrt(np.clip(a,0,1))),out=d)
    return d
def nm(iso,locs):
    ll=[(a,b) for a,b,_ in locs]; h=hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p=_C/f"{iso}_res5_2h_{h}_driving.npz"
    if not p.exists(): return None
    m=np.load(p)["matrix"]
    if not np.isfinite(m).any(): return None
    v=m.min(axis=1); v[~np.isfinite(v)]=np.inf; return v
# biggest-population countries
names=['China','India','United States','Indonesia','Pakistan','Nigeria','Brazil','Bangladesh','Russian Federation','Japan','Mexico','Ethiopia','Philippines','Egypt','Viet Nam','Iran, Islamic Republic of','Turkey','Germany','Thailand','United Kingdom']
print(f"{'Country':22} {'pop(M)':>7} {'120min A_G':>11}  src")
for name in names:
    try: obj=pycountry.countries.lookup(name)
    except: continue
    try:
        g=load_population_at_resolution(obj.name,target_resolution=5); locs,_=load_linacs_from_dirac_db(obj.name)
    except: 
        print(f"{name:22} {'--':>7}  no data"); continue
    pop=g["population"].to_numpy(float); tot=pop.sum()
    v=nm(obj.alpha_3,locs)
    if v is not None:
        cov=pop[v<=120].sum()/tot; src="TravelTime"
    else:
        d=nk(g,locs); cov=pop[d<=PROXY_KM_120].sum()/tot; src="proxy 68km"
    print(f"{obj.name[:22]:22} {tot/1e6:>7.0f} {cov:>10.1%}  {src}")
