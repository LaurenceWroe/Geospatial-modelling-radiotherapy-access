import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt_loader import _creds
from analysis.convergence import run_convergence
app_id, api_key = _creds()
df = run_convergence("Norway","NOR",app_id,api_key)
print(f"A_C (flat, all resolutions) = {df['A_C'].iloc[0]*100:.1f}%   |   {df.attrs.get('n_facilities')} facilities\n")
for metric in ("distance","time"):
    unit = "km" if metric=="distance" else "min"
    for col in ("A_G","A_RM"):
        piv = df[df.metric==metric].pivot(index="resolution",columns="threshold",values=col)*100
        piv.columns = [f"{int(c)} {unit}" for c in piv.columns]
        print(f"### {metric.upper()} — {col}")
        print(piv.round(1).to_string()); print()
