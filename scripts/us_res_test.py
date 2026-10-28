import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import h3
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
_R=6371.0088
def nk(gdf,locs):
    la=np.radians([h3.cell_to_latlng(h)[0] for h in gdf["h3"]]); lo=np.radians([h3.cell_to_latlng(h)[1] for h in gdf["h3"]]); cl=np.cos(la)
    d=np.full(len(gdf),np.inf)
    for lat,lon,_ in locs:
        p,l=math.radians(lat),math.radians(lon)
        a=np.sin((la-p)/2)**2+math.cos(p)*cl*np.sin((lo-l)/2)**2
        np.minimum(d,2*_R*np.arcsin(np.sqrt(np.clip(a,0,1))),out=d)
    return d
locs,_=load_linacs_from_dirac_db("United States")
print("USA population-weighted access within 30 km straight-line, by grid resolution")
print("(coarser = more centroid-like; if access falls, that's the paper's effect)")
for res in (3,4,5,6):
    g=load_population_at_resolution("United States",target_resolution=res)
    d=nk(g,locs); pop=g["population"].to_numpy(float)
    print(f"  res {res} ({h3.average_hexagon_area(res,unit='km^2'):>7.0f} km2/hex, {len(g):>7,} hexes):  {pop[d<=30].sum()/pop.sum():.1%}")
