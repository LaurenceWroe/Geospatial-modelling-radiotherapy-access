import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), default_timeout=3600)
at.run()
for b in at.button:
    if "Calculate" in (b.label or ""):
        b.click(); break
at.run()
print("exceptions:", len(at.exception))
for e in at.exception: print("EXC:", str(e)[:300])
print("cache exists:", os.path.exists("cache/world_default.pkl"),
      os.path.getsize("cache/world_default.pkl") if os.path.exists("cache/world_default.pkl") else 0)
