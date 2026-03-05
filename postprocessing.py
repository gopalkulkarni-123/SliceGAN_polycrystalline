import numpy as np
import tifffile
import porespy as ps
import matplotlib.pyplot as plt
from skimage import measure

def get_volume_fraction(vol):
    """Calculates the volume fraction of the 1-phase."""
    return np.mean(vol)

def get_surface_area_density(vol):
    """Calculates Sv (Surface Area / Total Volume) using Marching Cubes."""
    if np.unique(vol).size == 1: # Handle case where GAN output is all one value
        return 0.0
    verts, faces, _, _ = measure.marching_cubes(vol)
    surface_area = measure.mesh_surface_area(verts, faces)
    return surface_area / vol.size

def get_two_point_correlation(vol):
    """Computes the S2 probability function using PoreSpy."""
    return ps.metrics.two_point_correlation(vol)

def get_chord_distribution(vol, bins=25):
    """Computes chord length distribution on the grain phase (0-phase)."""
    # 1 - vol flips boundaries to grains so we measure grain sizes
    return ps.metrics.chord_length_distribution(1 - vol, bins=bins)

def plot_comparison(real_data, gan_data, title, xlabel, ylabel, plot_type='plot'):
    """Helper to standardize plotting."""
    plt.figure(figsize=(6, 4))
    if plot_type == 'plot':
        plt.plot(real_data.distance, real_data.probability, 'k-', label='Real')
        plt.plot(gan_data.distance, gan_data.probability, 'r--', label='SliceGAN')
    elif plot_type == 'bar':
        plt.bar(real_data.L, real_data.pdf, alpha=0.5, color='gray', label='Real')
        plt.step(gan_data.L, gan_data.pdf, color='red', where='mid', label='SliceGAN')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def run_microstructure_report(real_path, gan_path):
    # 1. Load and Binarize
    real_vol = (tifffile.imread(real_path) > 0).astype(int)
    gan_vol = (tifffile.imread(gan_path) > 0).astype(int)
    
    # 2. Compute Metrics
    metrics = {
        "Boundary Vol Fraction": (get_volume_fraction(real_vol), get_volume_fraction(gan_vol)),
        "Surface Area Density (Sv)": (get_surface_area_density(real_vol), get_surface_area_density(gan_vol))
    }
    
    # 3. Print Tabular Report
    print(f"\n{'Metric':<25} | {'Real':<12} | {'SliceGAN':<12}")
    print("-" * 55)
    for name, values in metrics.items():
        print(f"{name:<25} | {values[0]:<12.4f} | {values[1]:<12.4f}")

    # 4. Generate Distributions
    s2_real = get_two_point_correlation(real_vol)
    s2_gan = get_two_point_correlation(gan_vol)
    
    chords_real = get_chord_distribution(real_vol)
    chords_gan = get_chord_distribution(gan_vol)

    # 5. Visualize
    plot_comparison(s2_real, s2_gan, 'Two-Point Correlation ($S_2$)', 'Distance (voxels)', 'Probability')
    plot_comparison(chords_real, chords_gan, 'Grain Size (Chord Length)', 'Length (voxels)', 'PDF', plot_type='bar')

# --- RUN ---
real_bin_path = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/extracted_jpegs/binarizedImages/real3Dvolume.tif'
gan_bin_path = '/home/kulkarni/Desktop/2D_3D_conversion/SliceGAN_polycrystalline/Trained_Generators/NMCTrained/HPC_output/NMC_Trained_binarized_100_epochs_900_crops_thin.tif'

run_microstructure_report(real_bin_path, gan_bin_path)