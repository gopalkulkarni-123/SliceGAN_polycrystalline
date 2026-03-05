import os
import numpy as np
import tifffile
from PIL import Image
from skimage import segmentation

def process_and_slice_boundaries(
    tiff_path,
    output_3d_path,
    output_2d_folder,
    num_slices=100,
    boundary_mode='subpixel',
    downsample_to_original=False,
):
    # 1. Load the 3D Grain IDs
    volume = tifffile.imread(tiff_path)
    print(f"Loaded volume shape: {volume.shape}")

    # 2. Binarize the entire 3D volume first
    # This ensures 2D slices and 3D volume are perfectly consistent
    print(f"Binarizing 3D volume (mode={boundary_mode})...")
    binary_3d = segmentation.find_boundaries(volume, mode=boundary_mode)

    # optionally shrink the upsampled result back to the original grid
    if downsample_to_original and boundary_mode == 'subpixel':
        binary_3d = binary_3d[::2, ::2, ::2]
        print(f"Downsampled boundary volume to original shape {binary_3d.shape}")

    # Convert to uint8 (0 and 255)
    binary_3d_uint8 = (binary_3d.astype(np.uint8) * 255)

    # 3. Save the 3D Binary Volume (Ground Truth)
    if not os.path.exists(os.path.dirname(output_3d_path)):
        os.makedirs(os.path.dirname(output_3d_path))
    tifffile.imwrite(output_3d_path, binary_3d_uint8, imagej=True)
    print(f"3D Binary Ground Truth saved to: {output_3d_path}")

    # 4. Extract and Save 2D Slices from the already binarized volume
    if not os.path.exists(output_2d_folder):
        os.makedirs(output_2d_folder)

    total_depth = binary_3d_uint8.shape[0]
    indices = np.linspace(0, total_depth - 1, num=num_slices, dtype=int)

    print(f"Extracting {num_slices} slices from binarized volume...")
    for i, idx in enumerate(indices):
        # Slice from the binarized uint8 volume
        binary_slice = binary_3d_uint8[idx, :, :]

        # Save as PNG
        img = Image.fromarray(binary_slice)
        filename = f"test_slice_{i:03d}_idx{idx}.jpg"
        #img.save(os.path.join(output_2d_folder, filename))

    print(f"Successfully saved {len(indices)} slices to '{output_2d_folder}'.")

# --- Usage ---
# specify the boundary mode that gives the best GAN results;
# when using 'subpixel' the produced volume will be (2*n-1)^3 in size
# but you can optionally downsample back to the original grid.
tiff_file = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/Grain_Structure.tif'  
output_3d = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/extracted_jpegs/binarizedImages/real3Dvolume.tif'
output_folder_2d = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/extracted_jpegs/binarizedImages/'

# use subpixel boundaries and keep the enlarged grid for training
process_and_slice_boundaries(
    tiff_file,
    output_3d,
    output_folder_2d,
    num_slices=10,
    boundary_mode='subpixel',
    downsample_to_original=False,
)

# if you later need a volume that matches the input shape, set
# downsample_to_original=True; this simply takes every other voxel
# along each axis after the subpixel result has been computed.