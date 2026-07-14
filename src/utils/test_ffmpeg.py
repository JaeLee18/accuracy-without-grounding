import subprocess
import os
from imageio_ffmpeg import get_ffmpeg_exe

ffmpeg = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())
r = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
print(r.stdout[:500])
r2 = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True)
for line in r2.stdout.split("\n"):
    if "264" in line.lower():
        print(line)
