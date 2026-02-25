import h5py
import numpy as np
import tifffile

file_path = '/home/kulkarni/Desktop/2D_3D_conversion/Input/Synthetic HCP 3D polycrystalline microstructures with grain-wise microstructural descriptors and stress fields under uniaxial tensile deformation  Part One/Equal CRSS/HCP_equal_CRSS_voxelwise/EqualCRSS_micro1_voxelwise/micro1_1_voxel.h5'

with h5py.File(file_path, 'r') as f:
    # 1. Read the data
    grains_raw = f['ngr'][:]
    print(f"Original shape from H5: {grains_raw.shape}")

    # 2. RESHAPE the data
    # Calculate the side length (cube root) if it's a perfect cube
    # For example, if grains_raw has 262144 elements, side is 64
    side_length = int(round(len(grains_raw)**(1/3)))
    
    # If the file is a cube, use this:
    grains = grains_raw.reshape((side_length, side_length, side_length))
    
    # Optional: Transpose for ImageJ orientation (Z, Y, X)
    grains = np.transpose(grains, (2, 1, 0))

    # 3. Save
    print(f"New shape for TIFF: {grains.shape}")
    tifffile.imwrite('Grain_Structure.tif', grains.astype('uint16'), imagej=True)