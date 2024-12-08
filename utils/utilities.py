
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def make_pastel_colormap(base_cmap_name, blend_factor=0.5):
    """
    Create a pastel version of a given base colormap by blending it with white.

    Parameters:
        base_cmap_name (str): Name of the base colormap (e.g., 'jet').
        blend_factor (float): Blending factor with white (0 = no change, 1 = fully white).

    Returns:
        LinearSegmentedColormap: A pastel colormap.
    """
    base_cmap = plt.cm.get_cmap(base_cmap_name)
    colors = base_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])  # RGBA for white
    pastel_colors = (1 - blend_factor) * colors + blend_factor * white
    pastel_cmap = LinearSegmentedColormap.from_list(
        f"{base_cmap_name}_pastel", pastel_colors
    )
    return pastel_cmap