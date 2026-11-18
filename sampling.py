"""
sampling.py
===========
Monte Carlo rejection sampling of the electron's position from the true
quantum probability distribution |psi_nlm(r,theta,phi)|^2 * r^2 (the r^2
comes from the spherical volume element). This is what actually produces
the "electron cloud" point cloud used for visualization -- every accepted
point is a physically valid location the electron could be found at, in
exactly the right relative frequency dictated by quantum mechanics.
"""

import numpy as np
from .wavefunctions import radial_wavefunction, probability_density


def sample_electron_cloud(
    n: int, l: int, m: int,
    n_samples: int = 50000,
    r_max: float = None,
    seed: int = 0,
    return_density: bool = False,
):
    """
    Rejection-sample `n_samples` electron positions (Cartesian x,y,z, in
    Bohr radii) from |psi_nlm|^2. If return_density is True, also returns
    the (unnormalized) probability weight at each accepted point, useful
    for continuous color-mapping in visualizations.
    """
    rng = np.random.default_rng(seed)
    if r_max is None:
        r_max = 3 * n * n + 8

    # Bound on the target density for rejection sampling, estimated from
    # the radial part's peak (angular part |Y|^2 is O(1) after normalization)
    r_grid = np.linspace(1e-3, r_max, 800)
    R_grid = radial_wavefunction(n, l, r_grid)
    max_radial_density = np.max(r_grid**2 * R_grid**2) * 1.25
    density_bound = max_radial_density

    xs, ys, zs, dens_vals = [], [], [], []
    batch = max(n_samples * 4, 4000)
    tries = 0
    while len(xs) < n_samples and tries < 80:
        tries += 1
        r = rng.uniform(0, r_max, batch)
        theta = np.arccos(rng.uniform(-1, 1, batch))
        phi = rng.uniform(0, 2 * np.pi, batch)

        dens = probability_density(n, l, m, r, theta, phi)
        weight = dens * r**2
        u = rng.uniform(0, 1, batch) * density_bound
        keep = u < weight

        need = n_samples - len(xs)
        r_k, th_k, ph_k, w_k = r[keep][:need], theta[keep][:need], phi[keep][:need], weight[keep][:need]

        x = r_k * np.sin(th_k) * np.cos(ph_k)
        y = r_k * np.sin(th_k) * np.sin(ph_k)
        z = r_k * np.cos(th_k)

        xs.extend(x.tolist())
        ys.extend(y.tolist())
        zs.extend(z.tolist())
        dens_vals.extend(w_k.tolist())

    x, y, z, dens = map(np.array, (xs, ys, zs, dens_vals))
    if return_density:
        return x, y, z, dens
    return x, y, z
