import matplotlib.pyplot as plt
import numpy as np
import tifffile

# Load your generated volume
data = tifffile.imread('/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/Examples/NMC.tif')

# Normalize or threshold to get a boolean mask (1 for grains, 0 for empty space)
# Adjust the threshold (e.g., 128) based on your grayscale values
binary_volume = data > 128 

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the voxels
# Note: plotting a 64x64x64 volume can be slow; you might want to slice it [::2, ::2, ::2] to speed up
ax.voxels(binary_volume, edgecolor='k', alpha=0.7)

ax.set_title("3D Voxel Reconstruction")
plt.show()