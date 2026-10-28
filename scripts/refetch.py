"""Re-fetch travel-time matrices for specific countries (real TravelTime)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tt_loader import make_tt_loader
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db

TARGETS = ["Uzbekistan", "Macao", "Barbados", "South Korea", "Republic of Korea", "Korea, Republic of"]

loader = make_tt_loader(5, max_minutes=120)
import pycountry
seen = set()
for name in TARGETS:
    try:
        obj = pycountry.countries.lookup(name)
    except LookupError:
        continue
    if obj.alpha_2 in seen:
        continue
    seen.add(obj.alpha_2)
    try:
        gdf = load_population_at_resolution(obj.name, target_resolution=5)
        locs, _ = load_linacs_from_dirac_db(obj.name)
    except Exception as e:
        print(f"{obj.name}: setup failed ({e})"); continue
    tt = loader(obj.name, obj.alpha_3, gdf)
    if tt is None:
        print(f"{obj.name}: loader returned None"); continue
    finite = int(np.isfinite(tt).any(axis=1).sum())
    print(f"{obj.name:20} {len(locs)} facilities, {finite}/{len(gdf)} hexes reachable "
          f"({'CACHED OK' if np.isfinite(tt).any() else 'ALL INF -> failed'})")
