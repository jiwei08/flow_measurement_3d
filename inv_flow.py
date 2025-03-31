import numpy as np
import os
from scipy.ndimage import gaussian_filter

# switch to cuda-11.8
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["CUDA_ROOT"] = "/usr/local/cuda-11.8"
os.environ["PATH"] = "/usr/local/cuda-11.8/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:"
    + os.environ["LD_LIBRARY_PATH"]
)


def load_mat_mask(param, i):
    from os.path import join
    from scipy.io import loadmat

    data = loadmat(join(param["mask dir"], param["mask pre"] + str(i) + ".mat"))
    if "sourceMask" in data:
        mask = data["sourceMask"]
    elif "rst" in data:
        mask = data["rst"]
    else:
        raise KeyError("sourceMask or rst are not key of data")
    
    # if param['shape'][0]==51:
    #     mask=np.repeat(mask,repeats=2,axis=0)[:-1,:,:]
    # elif param['shape'][0]==201:
    #     mask=mask[::2,:,:]
    return mask


def calc_flow(param):
    from utils.OpticalFlow import optical_flow

    pre_mask = load_mat_mask(param, 0)

    print("Mask shape: ",pre_mask.shape)

    if param["flow state"] == "stable":
        flow = np.zeros((*pre_mask.shape, 3))
    else:
        flow = np.zeros((param["nmasks"] - 1, *pre_mask.shape, 3))

    if not os.path.exists(param["output folder"]):
        os.makedirs(param["output folder"])

    for i in range(1, param["nmasks"]):
        if "deg" in param and "inlet" in param and "ylim" in param:
            print(
                "Flow Inversion with {:d}, {:d} th mask, degree {:d}, inlet {:d} and ylim [{:d}, {:d}]...".format(
                    i - 1,
                    i,
                    param["deg"],
                    param["inlet"],
                    param["ylim"][0],
                    param["ylim"][1],
                )
            )
        else:
            print("Flow Inversion with {:d}, {:d} th mask...".format(i - 1, i))
        cur_mask = load_mat_mask(param, i)

        pre_mask[pre_mask < 0.1] = 0
        cur_mask[cur_mask < 0.1] = 0
        pre_mask = gaussian_filter(pre_mask, 0.5)
        cur_mask = gaussian_filter(cur_mask, 0.5)
        ux, uy, uz = optical_flow(pre_mask, cur_mask, method=param["method"])
        pre_mask = cur_mask

        if param["flow state"] == "stable":
            flow[:, :, :, 0] = ux / param["Delta T"]
            flow[:, :, :, 1] = uy / param["Delta T"]
            flow[:, :, :, 2] = uz / param["Delta T"]
        else:
            flow[i - 1, :, :, :, 0] = ux / param["Delta T"]
            flow[i - 1, :, :, :, 1] = uy / param["Delta T"]
            flow[i - 1, :, :, :, 2] = uz / param["Delta T"]
    
        # if param['shape'][0]==201:
        #     flow_ori=np.repeat(flow,repeats=2,axis=-4)[...,:-1,:,:,:]
        # elif param['shape'][0]==51:
        #     flow_ori=flow[...,::2,:,:,:]

        if param["flow state"] == "stable":
            np.save(
                join(param["output folder"], param["output pre"] + str(i - 1) + ".npy"),
                flow,
            )

    if param["flow state"] == "unstable":
        np.save(join(param["output folder"], param["output pre"] + ".npy"), flow)

    # if param["flow state"] == "stable":
    #     flow /= param["nmasks"] - 1

    # np.save(join(param["output folder"], param["output filename"]), flow)


if __name__ == "__main__":
    import time, sys, json
    from os.path import join
    from datetime import datetime

    dir_pre = "/home1/lijiwei/flow-measurement3d/data/moving_T_junction_2m/400points"

    param = {
        "mask dir": join(dir_pre, "inversion/mask_inversions"),
        "mask pre": "inv_Ricker20k_recAll_dirs10_river_",
        "nmasks": 11,
        "shape": (201, 101, 101), # [grid point]
        # "degrees": None,  # [[degree, inlet],...] degrees of T-junction, or None
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
        # "ylims": ((0, 101), ),
        "Delta T": 0.5,
        "method": "farneback3d",  #'LK','HS','farneback3d'
        "flow state": "stable",  #'stable' or 'unstable'
        "output folder": join(dir_pre, "inversion/flow_inversions"),
        "output pre": "inv_flow_farneback3d_Ricker20k_recAll_dirs10_river_",
    }

    if param["degrees"] is None or param["ylims"] is None:
        calc_flow(param)
    else:
        mask_dir = param["mask dir"]
        output_folder = param["output folder"]
        for deg, inlet in param["degrees"]:
            for ylim in param["ylims"]:
                param["deg"] = deg
                param["inlet"] = inlet
                param["ylim"] = ylim
                param["mask dir"] = join(
                    mask_dir,
                    "T_junction_degree" + str(deg) + "_inlet" + str(inlet) + '_smallDeltaT',
                    "ylim" + str(ylim[0]) + "-" + str(ylim[1]),
                )
                param["output folder"] = join(
                    output_folder,
                    "T_junction_degree" + str(deg) + "_inlet" + str(inlet) + '_smallDeltaT',
                    "ylim" + str(ylim[0]) + "-" + str(ylim[1]),
                )

                calc_flow(param)
                # print(param["mask dir"],param['output folder'])
