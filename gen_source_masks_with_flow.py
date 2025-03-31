import numpy as np
import scipy.io as scio
import os


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

    mask = gaussian_filter(mask, param["mollify sigma"])

    return mask


def save_mat_npy(param, mask, index=0):
    if not os.path.isdir(param["mask dir"]):
        os.makedirs(param["mask dir"])
    np.save(
        join(param["mask dir"], param["mask pre"]) + str(index), {"sourceMask": mask}
    )
    scio.savemat(
        join(param["mask dir"], param["mask pre"]) + str(index) + ".mat",
        {"sourceMask": mask},
    )


def gen_mask_with_flow(param):
    import scipy.io as scio
    from opticalflow3D.helpers import generate_inverse_image

    flow_data = np.load(param["flow file"])
    if "ylim" in param and param["ylim"] is not None:
        flow_data = np.take(flow_data, range(*param["ylim"]), axis=-3)
    if param["flow state"] == "stable":
        time_nums = param["flow time nums"]
    else:
        time_nums = min(param["flow time nums"], flow_data.shape[0])

    if param["initial mask"] == "random":
        mask = gen_mask(param)
    elif param["initial mask"] == "from file":
        mask = np.load(param["initial mask file"], allow_pickle=True).item()[
            "sourceMask"
        ]

    save_mat_npy(param, mask, 0)

    dt=param["Delta T"]
    for i in range(1, time_nums):
        if "deg" in param and "inlet" in param:
            print(
                "Generating " + str(i) + " th mask, degree ",
                str(param["deg"]) + " inlet " + str(param["inlet"]),
                " and ylim ",
                param["ylim"],
                "...",
            )
        else:
            print("Generating " + str(i) + " th mask...")
        if param["flow state"] == "stable":
            mask = generate_inverse_image(
                mask,
                dt * flow_data[:, :, :, 2],
                dt * flow_data[:, :, :, 1],
                dt * flow_data[:, :, :, 0],
                use_gpu=False,
            )
        else:
            mask = generate_inverse_image(
                mask,
                dt * flow_data[i - 1, :, :, :, 2],
                dt * flow_data[i - 1, :, :, :, 1],
                dt * flow_data[i - 1, :, :, :, 0],
                use_gpu=False,
            )
        dist = param["dist from bdy"]
        mask = np.pad(mask[dist:-dist, dist:-dist, dist:-dist], dist)
        save_mat_npy(param, mask, i)


if __name__ == "__main__":
    from os.path import join

    dir_pre = "/home1/lijiwei/flow-measurement3d/data/moving_T_junction_2m/400points"

    param = {
        "mask dir": join(dir_pre, "mask_models"),
        "mask pre": "true_source_mask_",
        "flow file": "/home1/lijiwei/flow-measurement3d/data/moving_T_junction_2m/flow_data/T_junction_",
        # "flow file": "/home1/lijiwei/flow-measurement3d/data/moving/flow_data/Taylor_Green_stable_3x.npy",
        "flow state": "stable",  # 'stable' with shape (Nx,Ny,Nz,3) or 'unstable' with shape (Nt,Nx,Ny,Nz,3)
        "flow time nums": 11,
        # "degrees": None,  # [[degree, inlet_velocity],...] degrees of T-junction, or None
        "degrees": [
            # [30, 20],
            [60, 20],
            # [90, 20],
            # [120, 20],
            # [150, 20],
        ],
        "ylims": np.vstack(
            (np.arange(0, 1001, 100), np.arange(101, 1102, 100))
        ).T,  # ((start, stop),...) only for flow in the T-junction, or None
        "Delta T": 0.5,
        "domain size": (2.0, 1.0, 1.0),  # [m]
        "shape": (201, 101, 101),
        "npoints": 10,
        "radius range": (
            5,
            6,
        ),  # [grid points], it's better choosing (5,8) for 10 points and (3,5) for 50 points
        "dist from bdy": 12,  # [grid points]
        "mollify sigma": 2,
        "initial mask": "from file",  # 'random' or 'from file'
        "initial mask file": "/home1/lijiwei/flow-measurement3d/data/stationary/400points_2m/mask_models/true_source_mask.npy",
    }

    if param["ylims"] is None or param["degrees"] is None:
        gen_mask_with_flow(param)
    else:
        mask_dir = param["mask dir"]
        flow_pre = param["flow file"]
        for deg, inlet in param["degrees"]:
            param["flow file"] = flow_pre + str(deg) + "deg_" + str(inlet) + "inlet.npy"
            for ylim in param["ylims"]:
                param["ylim"] = ylim
                param["deg"] = deg
                param["inlet"] = inlet
                param["mask dir"] = join(
                    mask_dir,
                    "T_junction_degree" + str(deg) + "_inlet" + str(inlet)+"_smallDeltaT",
                    "ylim" + str(ylim[0]) + "-" + str(ylim[1]),
                )
                gen_mask_with_flow(param)
