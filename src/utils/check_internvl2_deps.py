import torch, os
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
try:
    from transformers import AutoTokenizer, AutoModel
    print("transformers OK")
except ImportError as e:
    print("transformers MISSING:", e)
try:
    from torchvision import transforms
    print("torchvision OK")
except ImportError as e:
    print("torchvision MISSING:", e)
try:
    from decord import VideoReader
    print("decord OK")
except ImportError as e:
    print("decord MISSING:", e)
try:
    import imageio_ffmpeg
    print("imageio_ffmpeg OK")
except ImportError as e:
    print("imageio_ffmpeg MISSING:", e)
try:
    from PIL import Image
    print("Pillow OK")
except ImportError as e:
    print("Pillow MISSING:", e)
cache = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache):
    internvl = [d for d in os.listdir(cache) if "InternVL" in d or "internvl" in d.lower()]
    print("Cached InternVL:", internvl if internvl else "(not cached — will download ~16GB)")
else:
    print("HF cache not found")
