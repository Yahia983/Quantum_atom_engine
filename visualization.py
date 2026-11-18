"""
visualization.py
=================
Rendering layer: static matplotlib plots (radial density, 2D cross-section)
and a fully self-contained, OFFLINE-capable interactive 3D viewer built with
Plotly. "Offline-capable" means the Plotly.js library itself is embedded
directly inside the output .html file (include_plotlyjs=True) -- once
downloaded, the file needs no internet connection ever again, including on
first open.
"""

import numpy as np
import matplotlib.pyplot as plt

from .wavefunctions import radial_probability_density, probability_density
from .sampling import sample_electron_cloud
from .constants import energy_level_eV, ORBITAL_LABELS

# Shared dark colorscale: black/near-black (low probability) -> purple ->
# magenta -> orange -> bright yellow (high probability)
PROBABILITY_COLORSCALE = [
    [0.0, "#050014"],
    [0.15, "#1b0c3f"],
    [0.35, "#5b1a8c"],
    [0.55, "#b3245c"],
    [0.75, "#f2622e"],
    [1.0, "#fde725"],
]
PROBABILITY_CMAP = "inferno"  # matplotlib equivalent for static plots


def plot_radial_density(n: int, l: int, save_path: str = None, r_max: float = None, ax=None):
    if r_max is None:
        r_max = 3 * n * n + 5
    r = np.linspace(0.001, r_max, 2000)
    P = radial_probability_density(n, l, r)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, P, color="#3b82f6", lw=2)
    ax.fill_between(r, P, alpha=0.2, color="#3b82f6")
    ax.set_xlabel("r (Bohr radii)")
    ax.set_ylabel(r"$r^2 |R_{nl}(r)|^2$")
    ax.set_title(f"Radial probability density: n={n}, l={l} ({ORBITAL_LABELS.get(l,'?')})")
    if save_path:
        ax.figure.savefig(save_path, dpi=150)
    return ax


def plot_cross_section(n: int, l: int, m: int, save_path: str = None, grid: int = 250, extent: float = None):
    if extent is None:
        extent = 3 * n * n + 8

    x = np.linspace(-extent, extent, grid)
    z = np.linspace(-extent, extent, grid)
    X, Z = np.meshgrid(x, z)
    r = np.sqrt(X**2 + Z**2) + 1e-9
    theta = np.arccos(np.clip(Z / r, -1, 1))
    phi = np.zeros_like(r)

    density = probability_density(n, l, m, r, theta, phi)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    im = ax.imshow(density, extent=[-extent, extent, -extent, extent],
                    origin="lower", cmap=PROBABILITY_CMAP)
    ax.set_facecolor("black")
    ax.set_xlabel("x (Bohr radii)", color="white")
    ax.set_ylabel("z (Bohr radii)", color="white")
    orbital = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    ax.set_title(f"Cross-section |ψ|² — orbital {orbital} (m={m})", color="white")
    ax.tick_params(colors="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability density", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="black")
    return fig, ax


def render_interactive_offline_html(
    n: int, l: int, m: int,
    out_path: str,
    n_samples: int = 60000,
    point_size: float = 2.2,
):
    """
    Builds a fully offline-capable interactive 3D electron cloud: orbit,
    zoom, and pan with the mouse, no internet connection required at any
    point (Plotly.js is embedded directly in the HTML file).
    """
    import plotly.graph_objects as go

    x, y, z, dens = sample_electron_cloud(n, l, m, n_samples=n_samples, return_density=True)
    dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-30)

    orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    energy = energy_level_eV(n)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=dens_norm,
                    colorscale=PROBABILITY_COLORSCALE,
                    opacity=0.55,
                    colorbar=dict(
                        title=dict(text="Relative<br>probability", font=dict(color="white")),
                        tickfont=dict(color="white"),
                    ),
                ),
                hovertemplate="x=%{x:.2f} a₀<br>y=%{y:.2f} a₀<br>z=%{z:.2f} a₀<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="black",
        scene=dict(
            xaxis=dict(title="x (Bohr radii)", backgroundcolor="black", gridcolor="#222"),
            yaxis=dict(title="y (Bohr radii)", backgroundcolor="black", gridcolor="#222"),
            zaxis=dict(title="z (Bohr radii)", backgroundcolor="black", gridcolor="#222"),
            aspectmode="data",
        ),
        title=dict(
            text=(
                f"Hydrogen electron cloud — orbital {orbital_label} "
                f"(n={n}, l={l}, m={m})   |   E_{n} = {energy:.3f} eV<br>"
                f"<sup>Drag to orbit · scroll to zoom · right-click-drag to pan "
                f"· works fully offline</sup>"
            ),
            font=dict(color="white"),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # include_plotlyjs=True embeds the full library -- zero internet needed,
    # ever, including the very first time the file is opened.
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return out_path
