import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('VRAM free:', round(torch.cuda.mem_get_info()[0]/1e9, 1), 'GB')
import importlib
for pkg in ['cv2', 'PIL', 'clip', 'matplotlib', 'scipy', 'tqdm']:
    try:
        importlib.import_module(pkg)
        print(f'{pkg}: OK')
    except ImportError:
        print(f'{pkg}: MISSING')
