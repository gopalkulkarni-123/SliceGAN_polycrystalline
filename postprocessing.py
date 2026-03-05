import tifffile as tiff
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

def simple_3d_mask(volume):

    # light denoising
    v = ndi.gaussian_filter(volume, sigma=1.0)

    # global 3D threshold
    t = filters.threshold_otsu(v)

    mask = v > t

    # small cleanup
    mask = morphology.remove_small_objects(mask, min_size=100)
    mask = ndi.binary_fill_holes(mask)

    return mask

def label_3d(mask):

    # 6 connectivity
    structure = ndi.generate_binary_structure(3, 1)

    labels, n = ndi.label(mask, structure=structure)

    return labels, n

def component_z_extents(labels):

    extents = []

    for lab in range(1, labels.max() + 1):

        zz = np.where(labels == lab)[0]

        if zz.size == 0:
            continue

        zmin = zz.min()
        zmax = zz.max()

        extents.append(zmax - zmin + 1)

    return np.array(extents)

def component_volumes(labels):

    counts = np.bincount(labels.ravel())
    return counts[1:]

def percolation_flags(labels):

    Z, Y, X = labels.shape

    flags = []

    for lab in range(1, labels.max() + 1):

        mask = (labels == lab)

        touch_z0 = mask[0,:,:].any()
        touch_z1 = mask[-1,:,:].any()

        touch_y0 = mask[:,0,:].any()
        touch_y1 = mask[:,-1,:].any()

        touch_x0 = mask[:,:,0].any()
        touch_x1 = mask[:,:,-1].any()

        flags.append({
            "z": touch_z0 and touch_z1,
            "y": touch_y0 and touch_y1,
            "x": touch_x0 and touch_x1
        })

    return flags

#real_vol = tiff.imread("real_volume.tif")
ai_vol   = tiff.imread(r"D:\TU_Darmstadt\SliceGAN_polycrystalline\Trained_Generators\NMCTrained\NMCTrained_50_epoch_50_crops.tif")

#print(real_vol.shape, real_vol.dtype)
print(ai_vol.shape, ai_vol.dtype)

#real_vol = np.moveaxis(real_vol, -1, 0)
ai_vol   = np.moveaxis(ai_vol,   -1, 0)

#real_vol = real_vol.astype(np.float32)
ai_vol   = ai_vol.astype(np.float32)

#real_mask = simple_3d_mask(real_vol)
ai_mask   = simple_3d_mask(ai_vol)