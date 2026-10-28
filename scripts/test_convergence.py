import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt_loader import _creds
from analysis.convergence import run_convergence
app_id, api_key = _creds()
t=time.time()
def prog(f,m): print(f"  [{f*100:3.0f}%] {m}", flush=True)
df = run_convergence("Norway","NOR",app_id,api_key,progress=prog)
print(f"\nElapsed {time.time()-t:.0f}s, errors={df.attrs.get('errors')}, facilities={df.attrs.get('n_facilities')}")
import pandas as pd
for metric in ("distance","time"):
    print(f"\n=== {metric} — A_G by resolution x threshold ===")
    piv=df[df.metric==metric].pivot(index="resolution",columns="threshold",values="A_G")
    print((piv*100).round(1).to_string())
