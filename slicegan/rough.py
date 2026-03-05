import tifffile
import numpy as np

def check_volume_specs(file_path):
    # Use TiffFile to read metadata without loading the full pixels yet
    with tifffile.TiffFile(file_path) as tif:
        volume_shape = tif.asarray().shape
        volume_dtype = tif.series[0].dtype
        
    print(f"--- Volume Inspection: {file_path.split('/')[-1]} ---")
    print(f"Shape (Z, Y, X): {volume_shape}")
    print(f"Data Type:       {volume_dtype}")
    print(f"Total Voxels:    {np.prod(volume_shape):,}")
    
    # Check for cubic symmetry
    if len(set(volume_shape)) == 1:
        print("Symmetry:        Perfectly Cubic")
    else:
        print("Symmetry:        Anisotropic (Non-cubic)")
        
    # Calculate memory footprint in GB
    # (assuming 1 byte for uint8, 4 for float32, etc.)
    bytes_per_pixel = np.dtype(volume_dtype).itemsize
    gb_size = (np.prod(volume_shape) * bytes_per_pixel) / (1024**3)
    print(f"Estimated RAM:   {gb_size:.2f} GB")
    print("-" * 40)

# --- Usage ---
real_volume_path = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/extracted_jpegs/binarizedImages/real3Dvolume.tif'
#real_volume_path = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/Grain_Structure.tif'

check_volume_specs(real_volume_path)