import os
import numpy as np
import tifffile
from PIL import Image

def extract_tiff_slices(tiff_path, output_folder, num_slices=100):
    # 1. Load the 3D TIFF
    # This returns a numpy array usually shaped (Depth, Height, Width)
    volume = tifffile.imread(tiff_path)
    
    print(f"Loaded volume shape: {volume.shape}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_depth = volume.shape[0]
    
    # 2. Determine which slices to take
    # If the volume has exactly 100 slices, we take them all.
    # If it has more, we spread the 100 slices evenly across the volume.
    indices = np.linspace(0, total_depth - 1, num=num_slices, dtype=int)

    for i, idx in enumerate(indices):
        slice_2d = volume[idx, :, :]

        # 3. Handle Data Scaling
        # JPEGs must be 8-bit (0-255). 
        # If your TIFF is 16-bit or float, we must normalize it.
        if slice_2d.dtype != np.uint8:
            s_min, s_max = slice_2d.min(), slice_2d.max()
            if s_max - s_min > 0:
                slice_2d = (slice_2d - s_min) / (s_max - s_min) * 255
            slice_2d = slice_2d.astype(np.uint8)

        # 4. Save as JPEG
        img = Image.fromarray(slice_2d)
        filename = f"slice_{i:03d}_idx{idx}.jpg"
        img.save(os.path.join(output_folder, filename), quality=95)

    print(f"Successfully saved {len(indices)} slices to '{output_folder}'.")

# --- Usage ---
tiff_file = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/Grain_Structure.tif'  # Your 3D TIFF file
output_dir = 'extracted_jpegs'
extract_tiff_slices(tiff_file, output_dir, num_slices=100)