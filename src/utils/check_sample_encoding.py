
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
data = open(VDG_DATA_ROOT + '/mvbench/mvbench_sample.json', 'rb').read()
pos = 417081
print('bytes around pos:', data[pos-5:pos+10].hex())
print('context:', repr(data[pos-20:pos+20]))
# Try latin-1 decode to see what it looks like
try:
    text = data.decode('latin-1')
    print('latin-1 decode OK, length:', len(text))
except Exception as e:
    print('latin-1 failed:', e)
