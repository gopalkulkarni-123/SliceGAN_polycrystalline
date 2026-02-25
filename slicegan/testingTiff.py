import torch
import tifffile as tiff
import numpy as np

# 1. Your tensor
noise = torch.randn(64, 64, 64)

# 2. Convert to NumPy 
# We move to CPU and ensure it's a float32 array
noise_np = noise.detach().cpu().numpy().astype(np.float32)

# 3. Save as a TIFF stack
# This will create a single file with 64 "pages" (slices)
tiff.imwrite('./polycrystalline_noise.tif', noise_np)

print("Saved as polycrystalline_noise.tif")