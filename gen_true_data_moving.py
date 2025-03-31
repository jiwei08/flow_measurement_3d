# successfully run with devito 4.7.1 but fail with devito 4.8
from difflib import context_diff
import os

# set devito enviroments (Must have been done before 'import devito' !)
# Please run with devito version 4.7.1
# os.environ["DEVITO_LANGUAGE"] = "openmp"
# os.environ["DEVITO_ARCH"] = "gcc"
# os.environ["DEVITO_AUTOTUNING"] = "max"

os.environ["DEVITO_PLATFORM"] = "nvidiaX"
os.environ["DEVITO_LANGUAGE"] = "openacc"
os.environ["DEVITO_ARCH"] = "nvc"

from devito import configuration

configuration["deviceid"] = 1  # Only supported by Devito>=4.8.2

import numpy as np
import time
import logging

logging.disable(logging.INFO)  # disable devito logger

from devito import TimeFunction, Function

from examples.seismic import Model, TimeAxis, Receiver

from utils.WaveSolver import wavesolver_waveform
from utils.IO import (
    write_npy_shotdata,
    load_mat_maskdata,
    write_npy_config,
    load_npy_maskdata,
)
from utils.GenWaveform import Gaussian_source, Ricker_source
from utils.RecConfig import recConfig
from utils.plots import plot_waveform


def gen_direction(ntheta_nphi=(3, 3)):
    if isinstance(ntheta_nphi, tuple):
        phi, theta = np.meshgrid(
            np.linspace(0, np.pi, ntheta_nphi[1] + 2)[1:-1],
            np.linspace(0, 2 * np.pi, ntheta_nphi[0]),
            indexing="ij",
        )
        phi, theta = phi.flatten(), theta.flatten()
        return np.column_stack(
            (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
        )
    elif isinstance(ntheta_nphi, int):
        # Fibonacci grid samples
        alpha = (np.sqrt(5) - 1) / 2.0
        arr = np.arange(1, ntheta_nphi + 1)
        zz = (2 * arr - 1) / ntheta_nphi - 1
        xx = np.sqrt(1 - zz**2) * np.cos(2 * np.pi * arr * alpha)
        yy = np.sqrt(1 - zz**2) * np.sin(2 * np.pi * arr * alpha)
        return np.column_stack((xx, yy, zz))


def modelling(param):
    print("Source mask file: " + param["mask file"])

    domain_size = tuple(param["domain size"])  # domain size in meter
    shape = tuple(param["shape"])  # number of grid points
    spacing = (
        domain_size[0] / (shape[0] - 1),
        domain_size[1] / (shape[1] - 1),
        domain_size[2] / (shape[2] - 1),
    )  # grid spacing in meter
    origin = (0.0, 0.0, 0.0)
    nbl = 50

    # define a velocity profile
    vp = param["vp"] * np.ones(shape, dtype=np.float32)  # [m/s]
    vp = vp + param["vp"] * np.random.normal(
        0, param["coefficient noise level"], vp.shape
    )

    # vp[:, :, 51:] = 2500.0

    t0 = param["t0"]  # [s]
    tn = param["t1"]  # [s]
    # if direction_of_translation is not None:
    #     tn = 2 * np.linalg.norm(domain_size) / vp.min() + 6.0 / (np.pi * f0)  # [s]
    # else:
    #     tn = np.linalg.norm(domain_size) / vp.min() + 6.0 / (np.pi * f0)  # [s]

    # if isRiver:
    #     tn += np.linalg.norm(domain_size) / vp.min()

    # define a model
    model = Model(
        vp=vp,
        origin=origin,
        shape=shape,
        spacing=spacing,
        space_order=2,
        nbl=nbl,
        bcs="damp",
        # fs=True,  # Ignore the top absorbing boundary or not
    )

    # To reduce the reflected wave
    model.damp.data[:] = model.damp.data[:] / vp.max()

    dt = model.critical_dt * 0.7
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    nt = time_range.num

    # symbol t, do not set t = model.grid.stepping_dim when using buffered Function u!!!
    t = model.grid.time_dim
    waveform = Ricker_source(
        f0=param["central frequency"], time_range=time_range, t=t
    )  # central frequency in Hz

    start_time = time.time()
    source_mask = Function(name="source_mask", grid=model.grid, dtype=np.float32)
    mask_file = join(param["mask dir"], param["mask file"])
    if mask_file == "single point":
        source_mask.data[
            nbl + 24 : nbl + 26, nbl + 48 : nbl + 50, nbl + 73 : nbl + 75
        ] = 1.0
    else:
        if mask_file[-4:] == ".mat":
            source_mask = load_mat_maskdata(mat_file=mask_file, model=model)
        elif mask_file[-4:] == ".npy":
            source_mask = load_npy_maskdata(npy_file=mask_file, model=model)

    rec = recConfig(
        model=model,
        time_range=time_range,
        dist=0.1,
        layout=param["receivers layout"],
        nums=param["receivers num"],
    )

    print("Start to define Function...")
    u = TimeFunction(name="u", grid=model.grid, space_order=2, time_order=2)
    u.data[:] = 0
    source = Function(name="source", grid=model.grid, space_order=1)
    source.data[:] = 0

    # rec_y = Receiver(
    #     name="rec_y",
    #     grid=model.grid,
    #     time_range=time_range,
    #     npoint=shape[0] * shape[2],
    # )
    # xrec, zrec = np.meshgrid(
    #     np.linspace(0, 1.0, shape[0]),
    #     np.linspace(0, 1.0, shape[2]),
    #     indexing="ij",
    # )
    # rec_y.coordinates.data[:, 0] = xrec.flatten()
    # rec_y.coordinates.data[:, 1] = zrec.flatten()
    # rec_y.coordinates.data[:, 2] = 0.75

    # generate operator
    print("Start to define Operator...")
    print("Start forward model...")
    recdata = np.zeros(rec.data.shape)
    op = wavesolver_waveform(
        model,
        u,
        source,
        waveform,
        source_mask,
        rec,
        time_range=time_range,
        direction=param["direction of translation"],
        isRiver=param["is river"],
        pml=False,
    )
    op.apply(time=nt - 2, dt=dt)

    print(
        "Total running time:",
        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
    )
    # add noise to observation
    rec.data[:] = rec.data[:] + np.max(np.abs(rec.data)) * np.random.normal(
        0, param["observation noise level"], rec.data.shape
    )

    # save observation data
    print("Saving observation data...")
    if not os.path.exists(param["output folder"]):
        os.makedirs(param["output folder"])
    write_npy_shotdata(join(param["output folder"], param["data filename"]), rec)

    # # save configuration data for inv.py
    # print("Saving configuration data...")
    # config_fname = output_folder + "/config.npy"
    # write_npy_config(
    #     config_fname,
    #     model=model,
    #     waveform=waveform,
    #     direction=param["direction of translation"],
    #     isRiver=param["is river"],
    # )
    print(
        "Shot data are saved in " + join(param["output folder"], param["data filename"])
    )


if __name__ == "__main__":
    import time, sys, json
    from os.path import join
    from datetime import datetime

    dir_pre = "/home1/lijiwei/flow-measurement3d/data/moving_T_junction_2m/400points"

    if len(sys.argv) == 1:
        param = {
            "central frequency": 2e4,
            "mask dir": join(
                dir_pre, "mask_models"
            ),  # source mask filename ,or 'single point' (for DEBUG)
            "mask pre": "true_source_mask_",
            "nmasks": 11,
            # "degrees": None, # [[degree, inlet_velocity],...] degrees of T-junction, or None
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
            "output folder": join(dir_pre, "data"),
            "data pre": "data_true_Ricker20k_recAll_dirs10_river_",
            "domain size": (2.0, 1.0, 1.0),  # [m]
            "shape": (201, 101, 101),
            "vp": 1500.0,  # [m/s], constant velocity profiler
            "t0": 0.0,
            "t1": 1.0e-2,  # [s]
            "receivers layout": "all",  # 'all','xOy and yOz','xOy','one xOy','yOz', or 'open channel'
            "receivers num": None,  # None: set receivers at every mesh point
            "num of directions": 10,  # (ntheta, nphi)
            "is river": True,
            "observation noise level": 0.0,
            "coefficient noise level": 0.0,
        }
    elif len(sys.argv) == 2:
        with open(sys.argv[1], "r") as f:
            param = json.load(f)
    else:
        raise ValueError("Invalid number of argments!")

    param["direction of translation"] = gen_direction(param["num of directions"])

    print("Now is {}".format(datetime.now()))

    if param["ylims"] is None or param["degrees"] is None:
        for m in range(param["nmasks"]):
            print("Generating data with " + str(m) + " th mask...")
            param["mask file"] = join(
                param["mask dir"], param["mask pre"] + str(m) + ".npy"
            )
            param["data filename"] = param["data pre"] + str(m) + ".npy"
            modelling(param)
    else:
        mask_dir = param["mask dir"]
        output_folder = param["output folder"]
        for deg, inlet in param["degrees"]:
            for ylim in param["ylims"]:
                param["mask dir"] = join(
                    mask_dir,
                    "T_junction_degree" + str(deg) + "_inlet" + str(inlet)+"_smallDeltaT",
                    "ylim" + str(ylim[0]) + "-" + str(ylim[1]),
                )
                param["output folder"] = join(
                    output_folder,
                    "T_junction_degree" + str(deg) + "_inlet" + str(inlet)+"_smallDeltaT",
                    "ylim" + str(ylim[0]) + "-" + str(ylim[1]),
                )
                for m in range(param["nmasks"]):
                    print(
                        "Generating data with " + str(m) + " th mask, degree ",
                        str(deg) + ", inlet " + str(inlet),
                        " and ylim ",
                        ylim,
                        "...",
                    )
                    param["mask file"] = param["mask pre"] + str(m) + ".npy"
                    param["data filename"] = param["data pre"] + str(m) + ".npy"
                    modelling(param)
                    # print(param["mask file"], param["data filename"])
