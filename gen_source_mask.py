import numpy as np
import os
import scipy.io as scio


def gen_ball_mask(shape, ball_info):
    mask = np.zeros(shape)
    xx, yy, zz = np.meshgrid(
        np.linspace(0, shape[0] - 1, shape[0]),
        np.linspace(0, shape[1] - 1, shape[1]),
        np.linspace(0, shape[2] - 1, shape[2]),
        indexing="ij",
    )
    for i in range(0, ball_info.shape[0]):
        mask[
            (xx - ball_info[i, 0]) ** 2
            + (yy - ball_info[i, 1]) ** 2
            + (zz - ball_info[i, 2]) ** 2
            <= ball_info[i, 3] ** 2
        ] = 1.0
    return mask


def gen_mask(param):
    npoints = param["npoints"]

    center_from_bdy = param["dist from bdy"] + param["radius range"][1]

    ball_info = np.zeros((npoints, 4))
    for i in range(3):
        ball_info[:, i] = (
            np.random.randint(0, param["shape"][i] - 2 * center_from_bdy, npoints)
            + center_from_bdy
        )
    ball_info[:, 3] = np.linspace(*param["radius range"], npoints)
    mask = gen_ball_mask(param["shape"], ball_info)

    from scipy.ndimage.filters import gaussian_filter
    mask=gaussian_filter(mask,param['mollify sigma'])

    if not os.path.exists(param["model dir"]):
        os.makedirs(param["model dir"])
    np.save(join(param["model dir"], param["model file"]), {"sourceMask": mask})
    scio.savemat(join(param["model dir"], param["model file"])+'.mat', {"sourceMask": mask})


if __name__ == "__main__":
    from os.path import join

    dir_pre = "/home1/lijiwei/flow-measurement3d/data/stationary/400points_2m"

    param = {
        "model dir": join(dir_pre, "mask_models"),
        "model file": "true_source_mask",
        "domain size": (2.0, 1.0, 1.0),  # [m]
        "shape": (201, 101, 101), # [grid point]
        "npoints": 400,
        "radius range": (3,4),  # [grid points]
        "dist from bdy": 12,  # [grid points]
        "mollify sigma": 1,
    }

    gen_mask(param)
