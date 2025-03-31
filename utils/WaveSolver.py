# -*- coding: utf-8 -*-
from devito import (
    Eq,
    solve,
    Operator,
    Function,
    TimeFunction,
    Constant,
    Max,
    Min,
    SpaceDimension,
    Grid,
    Inc,
)
from devito.symbolics.extended_sympy import INT, LONG
from devito.types import Scalar
import numpy as np
from sympy import floor, Piecewise

__all__ = ["wavesolver_waveform", "wavesolver_rev"]


def wavesolver_waveform(
    model,
    u,
    source,
    waveform,
    source_mask,
    rec,
    time_range,
    direction=None,
    pml=False,
    isRiver=False,
):
    x, y, z = model.grid.dimensions
    dx, dy, dz = model.grid.spacing_symbols
    t = model.grid.time_dim
    t_step = model.grid.stepping_dim
    dt = model.grid.time_dim.spacing
    # The x-, y- and z-components of direction must be positive!
    u.data[:] = 0.0
    rec.data[:] = 0.0

    if pml:
        pml = Function(name="pml", grid=model.grid)
        # The default value for pml_alpha refers to the manual of k-Wave
        pml.data[:] = _pml_absorption(
            pml_alpha=0.2 * model.vp.data.max() / min(model.spacing),
            # pml_alpha=0.1,
            pml_size=model.nbl,
            interior_shape=model.shape,
            degree=2,  # refer to the choice in k-Wave
            spacing=model.spacing,
            time_range=time_range,
        )
        pde = (
            (model.m * (pml * u.forward - 2.0 * u + u.backward / pml) / dt**2)
            - u.laplace
            - source
        )
    else:
        pde = model.m * u.dt2 - u.laplace + model.damp * u.dt - source

    stencil = [
        Eq(
            u.forward,
            solve(pde, u.forward),
            subdomain=model.grid.subdomains["interior"],
        )
    ]
    rec_term = rec.interpolate(expr=u.forward)

    if isRiver:
        nbl = model.nbl
        bc = [
            Eq(u[t_step + 1, x, y, nbl], 0.0),
            Eq(
                u[t_step + 1, x, y, model.grid.shape[2] - nbl - 1],
                u[t_step + 1, x, y, model.grid.shape[2] - nbl - 2],
            ),
            Eq(u[t_step + 1, nbl, y, z], u[t_step + 1, 1 + nbl, y, z]),
            Eq(
                u[t_step + 1, model.grid.shape[0] - nbl - 1, y, z],
                u[t_step + 1, model.grid.shape[0] - nbl - 2, y, z],
            ),
        ]
        stencil += bc

    if direction is None:
        stencil_source = Eq(
            source, waveform * source_mask, subdomain=model.grid.subdomains["interior"]
        )
        return Operator([stencil_source] + stencil + rec_term)
    else:
        assert direction.ndim == 2
        sum_plane_waveform = TimeFunction(name="sum_plane_waveform", grid=model.grid)
        sum_plane_waveform.data[:] = 0

        stencil_index_waveform = sum_waveform_stencil(
            direction, waveform, sum_plane_waveform, model
        )

        stencil_source = Eq(
            source,
            sum_plane_waveform * source_mask,
        )
        return Operator(stencil_index_waveform + [stencil_source] + stencil + rec_term)


def wavesolver_rev(
    model,
    u,
    source_mask,
    waveform_rev,
    src,
    time_range,
    direction=None,
    pml=False,
    isRiver=False,
):
    x, y, z = model.grid.dimensions
    dx, dy, dz = model.grid.spacing_symbols
    t = model.grid.time_dim
    t_step = model.grid.stepping_dim
    dt = model.grid.time_dim.spacing
    u.data[:] = 0.0
    source_mask.data[:] = 0.0

    if pml:
        pml = Function(name="pml", grid=model.grid)
        # The default value for pml_alpha refers to the manual of k-Wave
        pml.data[:] = _pml_absorption(
            # pml_alpha=2 * model.vp.data.max() / min(model.spacing),
            pml_alpha=0.2,
            pml_size=model.nbl,
            interior_shape=model.shape,
            degree=4,  # refer to the choice in k-Wave
            spacing=model.spacing,
            time_range=time_range,
        )
        pde = (
            model.m * (pml * u.forward - 2.0 * u + u.backward / pml) / dt**2
        ) - u.laplace
    else:
        pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

    stencil = [
        Eq(
            u.forward,
            solve(pde, u.forward),
            subdomain=model.grid.subdomains["interior"],
        )
    ]
    src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
    if isRiver:
        nbl = model.nbl
        bc = [
            Eq(u[t_step + 1, x, y, nbl], 0.0),
            Eq(
                u[t_step + 1, x, y, model.grid.shape[2] - nbl - 1],
                u[t_step + 1, x, y, model.grid.shape[2] - nbl - 2],
            ),
            Eq(u[t_step + 1, nbl, y, z], u[t_step + 1, 1 + nbl, y, z]),
            Eq(
                u[t_step + 1, model.grid.shape[0] - nbl - 1, y, z],
                u[t_step + 1, model.grid.shape[0] - nbl - 2, y, z],
            ),
        ]
        stencil += bc

    if direction is None:
        eq = Eq(
            source_mask,
            source_mask + waveform_rev * u.forward,
            subdomain=model.grid.subdomains["interior"],
        )

        return Operator(stencil + src_term + [eq])
    else:
        assert direction.ndim == 2
        sum_plane_waveform_rev = TimeFunction(
            name="sum_plane_waveform_rev", grid=model.grid
        )
        sum_plane_waveform_rev.data[:] = 0

        stencil_index_waveform_rev = sum_waveform_stencil(
            direction, waveform_rev, sum_plane_waveform_rev, model, forward=False
        )

        sum_eq = Eq(
            source_mask,
            source_mask + sum_plane_waveform_rev * u.forward,
            subdomain=model.grid.subdomains["interior"],
        )

        return Operator(stencil_index_waveform_rev + stencil + src_term + [sum_eq])


def sum_waveform_stencil(direction, waveform, sum, model, forward=True):
    x, y, z = model.grid.dimensions
    dx, dy, dz = model.grid.spacing_symbols
    t = model.grid.time_dim
    stencil = []
    dt_const = Constant(name="dt_const")
    dt_const.data = model.critical_dt
    d1 = SpaceDimension(name="d1")
    d2 = SpaceDimension(name="d2")
    grid_d = Grid(shape=direction.shape, dimensions=(d1, d2))
    func_d = Function(name="func_d", dimensions=(d1, d2), grid=grid_d)
    func_d.data[:] = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    stencil.append(Eq(sum, 0))
    flag = -1.0 if forward else 1.0
    stencil.append(
        Inc(
            sum,
            waveform[
                LONG(
                    Min(
                        Max(
                            t
                            + flag
                            * (
                                func_d[d1, 0] * x * dx
                                + func_d[d1, 1] * y * dy
                                + func_d[d1, 2] * z * dz
                            )
                            / (model.vp * dt_const),
                            0,
                        ),
                        waveform.shape[0] - 1,
                    )
                )
            ],
        )
    )
    return stencil


def _pml_absorption(pml_alpha, pml_size, interior_shape, degree, spacing, time_range):
    """
    Refer to the PML used in k-Wave and see details in section 2.6 of the k-Wave manual

    pml_size: int, the thickness of the pml layer
    interior_shape: (Nx,Ny) or (Nx,Ny,Nz)
    """

    pml_mat = np.zeros([i + 2 * pml_size for i in interior_shape])

    if len(interior_shape) == 2:
        # x-direction pml setting
        pml_x = pml_alpha * (
            (np.linspace(spacing[0], pml_size * spacing[0], pml_size)) ** degree
            / (pml_size * spacing[0]) ** degree
        ).reshape(pml_size, 1)
        pml_mat[0:pml_size, :] = np.tile(pml_x[::-1], (1, pml_mat.shape[1]))
        pml_mat[-pml_size:, :] = np.tile(pml_x, (1, pml_mat.shape[1]))

        # y-direction pml setting
        pml_y = pml_alpha * (
            (np.linspace(spacing[1], pml_size * spacing[1], pml_size)) ** degree
            / (pml_size * spacing[1]) ** degree
        )
        pml_mat[:, 0:pml_size] += np.tile(pml_y[::-1], (pml_mat.shape[0], 1))
        pml_mat[:, -pml_size:] += np.tile(pml_y, (pml_mat.shape[0], 1))

        pml_mat = np.exp(pml_mat * time_range.step)
    elif len(interior_shape) == 3:
        # x-direction pml setting
        pml_x = pml_alpha * (
            (np.linspace(spacing[0], pml_size * spacing[0], pml_size)) ** degree
            / (pml_size * spacing[0]) ** degree
        ).reshape(pml_size, 1, 1)
        pml_mat[0:pml_size, :, :] = np.tile(
            pml_x[::-1, :, :], (1, pml_mat.shape[1], pml_mat.shape[2])
        )
        pml_mat[-pml_size:, :, :] = np.tile(
            pml_x, (1, pml_mat.shape[1], pml_mat.shape[2])
        )

        # y-direction pml setting
        pml_y = pml_alpha * (
            (np.linspace(spacing[1], pml_size * spacing[1], pml_size)) ** degree
            / (pml_size * spacing[1]) ** degree
        ).reshape(1, pml_size, 1)
        pml_mat[:, 0:pml_size, :] += np.tile(
            pml_y[:, ::-1, :], (pml_mat.shape[0], 1, pml_mat.shape[2])
        )
        pml_mat[:, -pml_size:, :] += np.tile(
            pml_y, (pml_mat.shape[0], 1, pml_mat.shape[2])
        )

        # z-direction pml setting
        pml_z = pml_alpha * (
            (np.linspace(spacing[2], pml_size * spacing[2], pml_size)) ** degree
            / (pml_size * spacing[2]) ** degree
        ).reshape(1, 1, pml_size)
        pml_mat[:, :, 0:pml_size] += np.tile(
            pml_z[:, :, ::-1], (pml_mat.shape[0], pml_mat.shape[1], 1)
        )
        pml_mat[:, :, -pml_size:] += np.tile(
            pml_z, (pml_mat.shape[0], pml_mat.shape[1], 1)
        )

        pml_mat = np.exp(pml_mat * time_range.step)
        # pml_mat = np.exp(pml_mat)
    else:
        ValueError("Invalid shape of interior_shape.")

    return pml_mat
