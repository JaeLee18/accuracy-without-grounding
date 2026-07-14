
# --- VDG portable paths (override via environment variables) ---
import os
VDG_VIDEOMME_SRC = os.environ.get("VDG_VIDEOMME_SRC", "data/raw/Video-MME")
# ------------------------------------------------------------
import pandas as pd
df = pd.read_parquet(VDG_VIDEOMME_SRC + '/videomme/test-00000-of-00001.parquet')
for tt in sorted(df['task_type'].unique()):
    sub = df[df.task_type==tt]
    s = len(sub[sub.duration=='short'])
    m = len(sub[sub.duration=='medium'])
    l = len(sub[sub.duration=='long'])
    print(f'{tt:<25} total={len(sub):>4}  short={s:>4}  medium={m:>4}  long={l:>4}  short+med={s+m:>4}')
