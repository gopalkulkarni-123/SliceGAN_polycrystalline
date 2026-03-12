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
    if np.unique(vol).size == 1:
        return 0.0
    verts, faces, _, _ = measure.marching_cubes(vol)
    surface_area = measure.mesh_surface_area(verts, faces)
    return surface_area / vol.size


def get_two_point_correlation(vol):
    """Computes the S2 probability function using PoreSpy."""
    return ps.metrics.two_point_correlation(vol)


def get_chord_distribution(vol, bins=25):
    """Computes chord length distribution on the grain phase (0-phase)."""
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

def plot_chord_distribution_smooth(vol, bins=25, title='Chord Length Distribution'):
    """
    Plots a smooth, readable chord length distribution for a 3D binary volume.
    
    Parameters
    ----------
    vol : ndarray
        3D binary volume (0 = background, 1 = phase of interest)
    bins : int
        Number of bins for the histogram
    title : str
        Plot title
    """
    # Step 1: Compute chords (flip phase to measure grains)
    chords = ps.metrics.chord_length_distribution(1 - vol, bins=bins)
    """
    # Step 2: Create a histogram manually
    L = chords.L
    pdf = chords.pdf
    
    # Step 3: Make a smooth step plot
    plt.figure(figsize=(7,4))
    plt.step(L, pdf, where='mid', linewidth=2, color='blue', label='PDF')
    plt.fill_between(L, 0, pdf, step='mid', alpha=0.3, color='blue')
    """
    pdf = chords.pdf / chords.pdf.max()
    plt.step(chords.L, pdf, where='mid')
    #plt.xlim(0, 4000)
    plt.xscale('log')
    plt.fill_between(chords.L, 0, pdf, step='mid', alpha=0.3)

    plt.xlabel('Chord Length (voxels)')
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_chord_distribution_smooth_compare(vol_real, vol_gan, bins=25, title='Chord Length Distribution'):
    """
    Plots a smooth, readable chord length distribution comparison for 3D binary volumes.
    
    Parameters
    ----------
    vol_real, vol_gan : ndarray
        3D binary volumes (0 = background, 1 = phase of interest)
    bins : int
        Number of bins for the histogram
    title : str
        Plot title
    """
    # Step 1: Compute chords using PoreSpy (flip phase: 1-vol to measure the phase of interest)
    chords_real = ps.metrics.chord_length_distribution(1 - vol_real, bins=bins)
    chords_gan = ps.metrics.chord_length_distribution(1 - vol_gan, bins=bins)

    # Step 2: Normalize PDF for relative comparison visibility
    pdf_real = chords_real.pdf / chords_real.pdf.max()
    pdf_gan = chords_gan.pdf / chords_gan.pdf.max()
    
    plt.figure(figsize=(7, 4))

    # Step 3: Plot Real Data (Gray base)
    plt.step(chords_real.L, pdf_real, where='mid', linewidth=2.5, color='gray', label='Real')
    plt.fill_between(chords_real.L, 0, pdf_real, step='mid', alpha=0.3, color='gray')

    # Step 4: Plot SliceGAN Data (Red overlay)
    plt.step(chords_gan.L, pdf_gan, where='mid', linewidth=2.5, color='red', label='SliceGAN')
    plt.fill_between(chords_gan.L, 0, pdf_gan, step='mid', alpha=0.2, color='red')

    # Step 5: Formatting and Log Scale
    plt.xscale('log')
    plt.xlabel('Chord Length (voxels)', fontweight='bold')
    plt.ylabel('Normalized Probability Density', fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold')
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()

def main():
    real_bin_path = r'D:\TU_Darmstadt\SliceGAN_polycrystalline\Trained_Generators\NMCTrained\HPC_output\real3DvolumeBinarized.tif'
    gan_bin_path = r'D:\TU_Darmstadt\SliceGAN_polycrystalline\Trained_Generators\NMCTrained\HPC_output\NMC_Trained_binarized__100_epochs_200_crops_binarized.tif'

    # 1. Load and binarize
    real_vol = (tifffile.imread(real_bin_path) > 0).astype(int)
    gan_vol = (tifffile.imread(gan_bin_path) > 0).astype(int)

    # 2. Compute metrics
    metrics = {
        "Boundary Vol Fraction": (
            get_volume_fraction(real_vol),
            get_volume_fraction(gan_vol)
        ),
        "Surface Area Density (Sv)": (
            get_surface_area_density(real_vol),
            get_surface_area_density(gan_vol)
        )
    }

    # 3. Print tabular report
    print(f"\n{'Metric':<25} | {'Real':<12} | {'SliceGAN':<12}")
    print("-" * 55)

    for name, values in metrics.items():
        print(f"{name:<25} | {values[0]:<12.4f} | {values[1]:<12.4f}")

    # 4. Generate distributions
    s2_real = get_two_point_correlation(real_vol)
    s2_gan = get_two_point_correlation(gan_vol)

    chords_real = get_chord_distribution(real_vol)
    chords_gan = get_chord_distribution(gan_vol)



    # 5. Visualize
    """plot_comparison(
        s2_real,
        s2_gan,
        'Two-Point Correlation ($S_2$)',
        'Distance (voxels)',
        'Probability'
    )

    plot_comparison(
        chords_real,
        chords_gan,
        'Grain Size (Chord Length)',
        'Length (voxels)',
        'PDF',
        plot_type='bar'
    )"""

    plot_chord_distribution_smooth(gan_vol)
    plot_chord_distribution_smooth(real_vol)

    plot_chord_distribution_smooth_compare(real_vol, gan_vol)


if __name__ == "__main__":
    main()