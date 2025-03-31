import numpy as np
from examples.seismic import Receiver

__all__ = ["recConfig"]


def recConfig(model, time_range, dist, layout, nums=None):
    """
    dist:   distance (in meter) between receivers and the boundary of the domain in meter
    layout: {'all','xOy and yOz','xOy','one xOy','yOz','open channel','singel'}
    nums:   (nx,ny,nz), numbers of receivers on axis
    """
    if nums is None:
        nx = model.shape[0] - 2 * int(dist / model.spacing[0])
        ny = model.shape[1] - 2 * int(dist / model.spacing[1])
        nz = model.shape[2] - 2 * int(dist / model.spacing[2])
    else:
        if type(nums) is tuple:
            (nx, ny, nz) = nums
        else:
            nx = ny = nz = nums

    if layout == "all":
        npoint = 2 * nx * ny + 2 * ny * nz + 2 * nz * nx
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )

        # set the coordinates of receivers
        xrec, yrec = np.meshgrid(
            np.linspace(dist, model.domain_size[0] - dist, nx),
            np.linspace(dist, model.domain_size[1] - dist, ny),
            indexing="ij",
        )
        rec.coordinates.data[0 : nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[0 : nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[0 : nx * ny, 2] = dist

        rec.coordinates.data[nx * ny : 2 * nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[nx * ny : 2 * nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[nx * ny : 2 * nx * ny, 2] = model.domain_size[2] - dist

        yrec, zrec = np.meshgrid(
            np.linspace(dist, model.domain_size[1] - dist, ny),
            np.linspace(dist, model.domain_size[2] - dist, nz),
            indexing="ij",
        )
        rec.coordinates.data[2 * nx * ny : (2 * nx * ny + ny * nz), 0] = dist
        rec.coordinates.data[2 * nx * ny : (2 * nx * ny + ny * nz), 1] = yrec.flatten()
        rec.coordinates.data[2 * nx * ny : (2 * nx * ny + ny * nz), 2] = zrec.flatten()

        rec.coordinates.data[
            (2 * nx * ny + ny * nz) : (2 * nx * ny + 2 * ny * nz), 0
        ] = (model.domain_size[2] - dist)
        rec.coordinates.data[
            (2 * nx * ny + ny * nz) : (2 * nx * ny + 2 * ny * nz), 1
        ] = yrec.flatten()
        rec.coordinates.data[
            (2 * nx * ny + ny * nz) : (2 * nx * ny + 2 * ny * nz), 2
        ] = zrec.flatten()

        zrec, xrec = np.meshgrid(
            np.linspace(dist, model.domain_size[2] - dist, nz),
            np.linspace(dist, model.domain_size[0] - dist, nx),
            indexing="ij",
        )
        rec.coordinates.data[
            (2 * nx * ny + 2 * ny * nz) : (2 * nx * ny + 2 * ny * nz + nz * nx), 0
        ] = xrec.flatten()
        rec.coordinates.data[
            (2 * nx * ny + 2 * ny * nz) : (2 * nx * ny + 2 * ny * nz + nz * nx), 1
        ] = dist
        rec.coordinates.data[
            (2 * nx * ny + 2 * ny * nz) : (2 * nx * ny + 2 * ny * nz + nz * nx), 2
        ] = zrec.flatten()

        rec.coordinates.data[
            (2 * nx * ny + 2 * ny * nz + nz * nx) : npoint, 0
        ] = xrec.flatten()
        rec.coordinates.data[(2 * nx * ny + 2 * ny * nz + nz * nx) : npoint, 1] = (
            model.domain_size[2] - dist
        )
        rec.coordinates.data[
            (2 * nx * ny + 2 * ny * nz + nz * nx) : npoint, 2
        ] = zrec.flatten()
    elif layout == "xOy and yOz":
        npoint = 2 * nx * ny + 2 * ny * nz
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )

        # set the coordinates of receivers
        xrec, yrec = np.meshgrid(
            np.linspace(dist, model.domain_size[0] - dist, nx),
            np.linspace(dist, model.domain_size[1] - dist, ny),
            indexing="ij",
        )
        rec.coordinates.data[0 : nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[0 : nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[0 : nx * ny, 2] = dist

        rec.coordinates.data[nx * ny : 2 * nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[nx * ny : 2 * nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[nx * ny : 2 * nx * ny, 2] = model.domain_size[2] - dist

        yrec, zrec = np.meshgrid(
            np.linspace(dist, model.domain_size[1] - dist, ny),
            np.linspace(dist, model.domain_size[2] - dist, nz),
            indexing="ij",
        )
        rec.coordinates.data[2 * nx * ny : (2 * nx * ny + ny * nz), 0] = dist
        rec.coordinates.data[2 * nx * ny : (2 * nx * ny + ny * nz), 1] = yrec.flatten()
        rec.coordinates.data[2 * nx * ny : (2 * nx * ny + ny * nz), 2] = zrec.flatten()

        rec.coordinates.data[(2 * nx * ny + ny * nz) : npoint, 0] = (
            model.domain_size[2] - dist
        )
        rec.coordinates.data[(2 * nx * ny + ny * nz) : npoint, 1] = yrec.flatten()
        rec.coordinates.data[(2 * nx * ny + ny * nz) : npoint, 2] = zrec.flatten()
    elif layout == "xOy":
        npoint = 2 * nx * ny
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )

        # set the coordinates of receivers
        xrec, yrec = np.meshgrid(
            np.linspace(dist, model.domain_size[0] - dist, nx),
            np.linspace(dist, model.domain_size[1] - dist, ny),
            indexing="ij",
        )
        rec.coordinates.data[0 : nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[0 : nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[0 : nx * ny, 2] = dist

        rec.coordinates.data[nx * ny :, 0] = xrec.flatten()
        rec.coordinates.data[nx * ny :, 1] = yrec.flatten()
        rec.coordinates.data[nx * ny :, 2] = model.domain_size[2] - dist
    elif layout == "one xOy":
        npoint = nx * ny
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )

        # set the coordinates of receivers
        xrec, yrec = np.meshgrid(
            np.linspace(dist, model.domain_size[0] - dist, nx),
            np.linspace(dist, model.domain_size[1] - dist, ny),
            indexing="ij",
        )
        rec.coordinates.data[0 : nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[0 : nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[0 : nx * ny, 2] = dist
    elif layout == "yOz":
        npoint = 2 * ny * nz
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )

        # set the coordinates of receivers
        yrec, zrec = np.meshgrid(
            np.linspace(dist, model.domain_size[1] - dist, ny),
            np.linspace(dist, model.domain_size[2] - dist, nz),
            indexing="ij",
        )
        rec.coordinates.data[0 : ny * nz, 0] = dist
        rec.coordinates.data[0 : ny * nz, 1] = yrec.flatten()
        rec.coordinates.data[0 : ny * nz, 2] = zrec.flatten()

        rec.coordinates.data[ny * nz : npoint, 0] = model.domain_size[2] - dist
        rec.coordinates.data[ny * nz : npoint, 1] = yrec.flatten()
        rec.coordinates.data[ny * nz : npoint, 2] = zrec.flatten()
    elif layout == "open channel":
        npoint = nx * ny + 2 * ny * nz
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )

        # set the coordinates of receivers
        xrec, yrec = np.meshgrid(
            np.linspace(dist, model.domain_size[0] - dist, nx),
            np.linspace(dist, model.domain_size[1] - dist, ny),
        )
        rec.coordinates.data[0 : nx * ny, 0] = xrec.flatten()
        rec.coordinates.data[0 : nx * ny, 1] = yrec.flatten()
        rec.coordinates.data[0 : nx * ny, 2] = dist

        yrec, zrec = np.meshgrid(
            np.linspace(dist, model.domain_size[1] - dist, ny),
            np.linspace(dist, model.domain_size[2] - dist, nz),
        )
        rec.coordinates.data[nx * ny : (nx * ny + ny * nz), 0] = dist
        rec.coordinates.data[nx * ny : (nx * ny + ny * nz), 1] = yrec.flatten()
        rec.coordinates.data[nx * ny : (nx * ny + ny * nz), 2] = zrec.flatten()

        rec.coordinates.data[(nx * ny + ny * nz) : npoint, 0] = (
            model.domain_size[2] - dist
        )
        rec.coordinates.data[(nx * ny + ny * nz) : npoint, 1] = yrec.flatten()
        rec.coordinates.data[(nx * ny + ny * nz) : npoint, 2] = zrec.flatten()
    elif layout == "single":
        npoint = 1
        rec = Receiver(
            name="rec", grid=model.grid, npoint=npoint, time_range=time_range
        )
        rec.coordinates.data[0, :] = np.array([d / 2.0 for d in model.domain_size])
    else:
        print("Invaild receiver layout!")

    rec.data[:] = 0.0
    return rec
