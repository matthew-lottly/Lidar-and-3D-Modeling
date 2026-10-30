"""Rooftop solar irradiance from roof planes and a surface model: sun geometry, terrain and building shading by horizon angles, and hourly beam and diffuse irradiance on every roof cell.

Per cell (a 1 m square on a roof plane with slope/aspect from the RANSAC fit):
  * horizon elevation in 72 azimuth directions, found by marching along each direction over the surface model (own roof and neighbours, buildings and trees);
  * beam: DNI x max(cos of the angle between sun and plane normal, 0) x (sun above the horizon in the sun's azimuth), summed over the hours of a typical meteorological year;
  * diffuse: sky treated as isotropic, so the annual diffuse is sum(DHI) x f, where f is the cosine-weighted share of the sky dome that the plane sees above its horizon (a fixed factor per cell);
  * no ground-reflected light, no circumsolar boost, no soiling or snow. Values are energy arriving at the roof surface, not electricity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

AZ_BINS = 72


def sun_positions(times, lat, lon, altitude=0.0):
    """Apparent solar elevation and azimuth (deg, azimuth clockwise from north) at UTC `times` via pvlib's NREL SPA."""
    import pvlib
    sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(times), lat, lon, altitude=altitude)
    return sp.apparent_elevation.to_numpy(), sp.azimuth.to_numpy()


def plane_normals(a, b):
    """Unit normals (east, north, up) of planes z = a x + b y + c."""
    n = np.c_[-np.asarray(a, float), -np.asarray(b, float), np.ones(np.size(a))]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def sun_vectors(el, az):
    e, a = np.radians(el), np.radians(az)
    return np.c_[np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)]


def horizon_angles(xy, z0, dsm, half, res=1.0, tol=0.15, max_range=250.0, n_az=AZ_BINS):
    """Horizon elevation (deg) for each point and each of `n_az` azimuths (0 = north, clockwise). The surface model is marched outward in 1 m steps to 60 m, then 2 m steps to `max_range`;
    an obstacle counts when it rises more than `tol` m above the point's own height (so noise on the roof itself does not shade it). Returns an array (n_points, n_az)."""
    d = np.r_[np.arange(1.0, 60.0, 1.0), np.arange(60.0, max_range + 1e-9, 2.0)]
    az = np.radians(np.arange(n_az) * 360.0 / n_az)
    H = np.zeros((len(xy), n_az), np.float32)
    fz = np.nan_to_num(dsm, nan=-1e3)
    n = dsm.shape[0]
    for k in range(0, len(xy), 4000):
        P, z = xy[k:k + 4000], z0[k:k + 4000]
        for j, a in enumerate(az):
            X = P[:, 0][:, None] + d[None] * np.sin(a)
            Y = P[:, 1][:, None] + d[None] * np.cos(a)
            col = (X + half) / res - 0.5
            row = (Y + half) / res - 0.5
            inside = (col >= 0) & (col <= n - 1) & (row >= 0) & (row <= n - 1)
            zs = ndimage.map_coordinates(fz, [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(X.shape)
            t = np.where(inside, (zs - z[:, None] - tol) / d[None], 0.0)
            H[k:k + 4000, j] = np.degrees(np.arctan(np.maximum(t.max(1), 0.0)))
    return H


def sunlit(H, el, az):
    """Boolean (n_points, n_times): the sun is above the horizon of each point in the sun's azimuth (linear interpolation between azimuth bins)."""
    n_az = H.shape[1]
    f = (az % 360.0) / (360.0 / n_az)
    i0 = np.floor(f).astype(int) % n_az
    i1 = (i0 + 1) % n_az
    w = f - np.floor(f)
    h = H[:, i0] * (1 - w)[None] + H[:, i1] * w[None]
    return el[None] > h


def diffuse_factor(H, normals, n_el=18):
    """Cosine-weighted share of the sky dome seen by each plane (1 = a horizontal plane with a flat horizon sees pi sr of weight): isotropic sky patches above each point's horizon, weighted by the
    cosine of the angle to the plane normal (clipped at 0) and by solid angle, divided by pi."""
    n_az = H.shape[1]
    az = np.radians((np.arange(n_az) + 0.5) * 360.0 / n_az)
    el = np.radians((np.arange(n_el) + 0.5) * 90.0 / n_el)
    dOm = (np.sin(np.radians((np.arange(n_el) + 1) * 90.0 / n_el)) - np.sin(np.radians(np.arange(n_el) * 90.0 / n_el))) * (2 * np.pi / n_az)
    A, E = np.meshgrid(az, el, indexing="ij")
    S = np.stack([np.sin(A) * np.cos(E), np.cos(A) * np.cos(E), np.sin(E)], -1).reshape(-1, 3)       # (n_az*n_el, 3)
    w = np.tile(dOm, n_az)
    cos = np.clip(normals @ S.T, 0, None)                                                           # (n_pts, n_patch)
    Hh = np.repeat(H, n_el, axis=1)                                                                 # horizon at each patch's azimuth
    vis = np.degrees(np.tile(E[0], n_az))[None] > Hh                                                # patch elevation above horizon
    return (cos * vis * w[None]).sum(1) / np.pi


def annual_irradiance(H, normals, el, az, dni, dhi, chunk=800):
    """Annual kWh/m2 on the plane surface of each point: (beam, diffuse). `el`, `az`, `dni`, `dhi` are hourly (one per TMY hour; hours with the sun below the horizon contribute no beam)."""
    up = el > 0
    S = sun_vectors(el[up], az[up])
    D = dni[up]
    beam = np.zeros(len(H))
    for k in range(0, len(H), chunk):
        cos = np.clip(normals[k:k + chunk] @ S.T, 0, None)
        beam[k:k + chunk] = (cos * D[None] * sunlit(H[k:k + chunk], el[up], az[up])).sum(1) / 1000.0
    diff = diffuse_factor(H, normals) * dhi.sum() / 1000.0
    return beam, diff


def unobstructed_beam(normals, el, az, dni):
    """Annual beam kWh/m2 on planes with a flat horizon (no shading), for the shading-loss comparison and for checks against pvlib."""
    up = el > 0
    S = sun_vectors(el[up], az[up])
    return (np.clip(normals @ S.T, 0, None) * dni[up][None]).sum(1) / 1000.0
