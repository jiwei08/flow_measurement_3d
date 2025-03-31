import os
import inspect

# set devito enviroments
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

from scipy.sparse.linalg import LinearOperator, lsmr, cg
from scipy.stats import wasserstein_distance

from pylops.optimization.basic import lsqr
from pylops.optimization.sparsity import SplitBregman
from pylops import FirstDerivative

from devito import TimeFunction, Function
from examples.seismic import Model, Receiver

from utils.WaveSolver import wavesolver_waveform, wavesolver_rev
from utils.GenWaveform import Gaussian_source, Ricker_source
from utils.IO import (
    load_npy_shotdata,
    write_mat_inversion,
    load_mat_maskdata,
    load_npy_config,
    load_npy_maskdata,
)


def gen_direction(ntheta_nphi=(3,3)):
    if isinstance(ntheta_nphi,tuple):
        phi, theta = np.meshgrid(
            np.linspace(0, np.pi, ntheta_nphi[1] + 2)[1:-1],
            np.linspace(0, 2 * np.pi, ntheta_nphi[0]),
            indexing="ij",
        )
        phi, theta = phi.flatten(), theta.flatten()
        return np.column_stack(
            (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
        )
    elif isinstance(ntheta_nphi,int):
        # Fibonacci grid samples
        alpha=(np.sqrt(5)-1)/2.0
        arr=np.arange(1,ntheta_nphi+1)
        zz=(2*arr-1)/ntheta_nphi-1
        xx=np.sqrt(1-zz**2)*np.cos(2*np.pi*arr*alpha)
        yy=np.sqrt(1-zz**2)*np.sin(2*np.pi*arr*alpha)
        return np.column_stack((xx,yy,zz))


def inv(param, maxiter):

    domain_size = tuple(param["domain size"])  # domain size in meter
    shape = tuple(param["shape"])  # number of grid points
    spacing = (
        domain_size[0] / (shape[0] - 1),
        domain_size[1] / (shape[1] - 1),
        domain_size[2] / (shape[2] - 1),
    )  # grid spacing in meter
    origin = (0.0, 0.0, 0.0)
    nbl = 50

    # define a velocity profile used for inversion
    vp = 1500.0 * np.ones(shape, dtype=np.float32)  # [m/s]

    # define a model
    model = Model(
        vp=vp,
        origin=origin,
        shape=shape,
        spacing=spacing,
        space_order=2,
        nbl=nbl,
        bcs="damp",
    )
    model.damp.data[:] = model.damp.data[:] / vp.max()

    print("Start to load obs data...")
    rec = load_npy_shotdata(join(param['data dir'],param["data filename"]), model.grid)
    obs_data = np.copy(rec.data).reshape(-1)
    time_range = rec.time_range

    nt = time_range.num
    dt = time_range.step

    t = model.grid.time_dim  # symbol t, do not use stepping_dim!!!

    waveform = Ricker_source(
        f0=param["central frequency"], time_range=time_range, t=t
    )  # central frequency in Hz
    waveform_rev = Function(
        name="waveform_rev", shape=(nt,), dimensions=(t,), dtype=np.float32
    )
    waveform_rev.data[:] = waveform.data[::-1]

    print("Start to define Function...")
    u = TimeFunction(name="u", grid=model.grid, space_order=2, time_order=2)
    u.data[:] = 0
    source_mask = Function(name="source_mask", grid=model.grid, dtype=np.float32)
    source_mask.data[:] = 0
    source = Function(name="source", grid=model.grid, space_order=1)
    source.data[:] = 0

    def forward(x):
        rec.data[:] = 0
        u.data[:] = 0
        source_mask.data[:] = 0
        source_mask.data[nbl:-nbl, nbl:-nbl, nbl:-nbl] = np.reshape(x, shape)

        op = wavesolver_waveform(
            model=model,
            u=u,
            source=source,
            waveform=waveform,
            source_mask=source_mask,
            rec=rec,
            time_range=time_range,
            direction=param["direction of translation"],
            isRiver=param["is river"],
        )
        op.apply(time=nt - 2, dt=dt)

        return rec.data.flatten()  # row-majority order

    def adjoint(y):
        source_mask.data[:] = 0
        u.data[:] = 0
        rec.data[:] = np.flip(
            np.reshape(y, rec.data.shape), -2
        )  # flip the obs data along time

        op = wavesolver_rev(
            model=model,
            u=u,
            source_mask=source_mask,
            waveform_rev=waveform_rev,
            src=rec,
            time_range=time_range,
            direction=param["direction of translation"],
            isRiver=param["is river"],
        )
        op.apply(time=nt - 2, dt=dt)

        return np.reshape(
            np.pad(
                source_mask.data[
                    nbl + 10 : -nbl - 10, nbl + 10 : -nbl - 10, nbl + 10 : -nbl - 10
                ],
                (10, 10),
                "constant",
                constant_values=(0, 0),
            ),
            -1,
        )

    # Initialize the linear operator
    mm = np.prod(shape)
    nn = np.prod(obs_data.shape)

    print("Start to define LinearOperator...")
    A = LinearOperator((nn, mm), matvec=forward, rmatvec=adjoint)

    # # adjoint test
    # xx = np.random.rand(mm)
    # yy = np.random.rand(nn)

    # v0 = np.dot(A.matvec(xx), yy)
    # v1 = np.dot(xx, A.rmatvec(yy))
    # print(v0, v1, (v0 - v1) / ((v0 + v1) / 2.0))


    def callback(xk):
        mask_file = param["mask file"]
        if mask_file[-4:] == ".mat":
            true_source_mask = load_mat_maskdata(mat_file=mask_file, model=model)
        elif mask_file[-4:] == ".npy":
            true_source_mask = load_npy_maskdata(npy_file=mask_file, model=model)

        global cur_iter, res_model, res

        frame = inspect.currentframe().f_back
        if "resid" in frame.f_locals:
            res[cur_iter] = frame.f_locals["resid"]
        elif "r1norm" in frame.f_locals:
            res[cur_iter] = frame.f_locals["r1norm"]
        elif "self" in frame.f_locals:
            res[cur_iter] = frame.f_locals["self"].r1norm

        res_model[cur_iter] = np.linalg.norm(
            xk - true_source_mask.data[nbl:-nbl, nbl:-nbl, nbl:-nbl].flatten()
        )
        # res_ot[cur_iter] = wasserstein_distance(
        #     xk, true_source_mask.data[nbl:-nbl, nbl:-nbl, nbl:-nbl].flatten()
        # )
        cur_iter += 1
        print("Running ", cur_iter, "th iteration...")

    # run inversion
    print("Running inversion...")
    start_time = time.time() 

    # LSQR
    result = lsqr(
        A,
        obs_data,
        x0=np.zeros(mm),
        niter=maxiter,
        show=True,
        callback=callback,
    )

    # # split Bregman with TV regularization
    # Dop = [
    #     FirstDerivative(mm, dims=shape, dir=0, edge=False, kind="backward"),
    #     FirstDerivative(mm, dims=shape, dir=1, edge=False, kind="backward"),
    #     FirstDerivative(mm, dims=shape, dir=2, edge=False, kind="backward"),
    # ]
    # result = SplitBregman(
    #     A,
    #     Dop,
    #     obs_data,
    #     niter_outer=20,
    #     niter_inner=1,
    #     mu=2,
    #     epsRL1s=[5e-10, 5e-10, 5e-10],
    #     tol=1e-4,
    #     tau=1.0,
    #     show=True,
    #     **dict(iter_lim=5, damp=1e-4)
    # )
    # # # print(np.max(result[0]))

    # # LSMR
    # result = lsmr(A, obs_data, maxiter=max_iters, show=True)

    # # CG
    # def compose_adjoint_with_forward(x):
    #     return adjoint(forward(x))

    # B = LinearOperator(
    #     shape=(mm, mm),
    #     matvec=compose_adjoint_with_forward,
    #     rmatvec=compose_adjoint_with_forward,
    # )
    # result = cg(B, adjoint(obs_data), maxiter=max_iters, callback=callback)

    print(
        "Solving the least-squares in ",
        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
    )

    print("Saving result...")
    if not os.path.exists(param["output folder"]):
        os.makedirs(param["output folder"])
    output_file = join(param['output folder'],param['output filename'])
    write_mat_inversion(
        output_file, result, model, res=res, res_model=res_model
    )
    print("Inversion results are saved in " + output_file)


if __name__ == "__main__":
    import time, sys, json
    from os.path import join
    from datetime import datetime

    dir_pre = "/home1/lijiwei/flow-measurement3d/data/stationary/400points_2m"

    max_iters = 100

    if len(sys.argv) == 1:
        param = {
            "central frequency": 2e4,
            "mask file": join(
                dir_pre, "mask_models/true_source_mask.npy"
            ),  # source mask filename ,or 'single point' (for DEBUG)
            "data dir": join(dir_pre, "data"),
            "data filename": "data_true_Ricker20k_recAll_dirs10_river.npy",
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
            "output folder": join(dir_pre, "inversion"),
            "output filename":"inv_Ricker20k_recAll_dirs10_river.mat"
        }
    elif len(sys.argv) == 2:
        with open(sys.argv[1], "r") as f:
            param = json.load(f)
    else:
        raise ValueError("Invalid number of argments!")

    param["direction of translation"] = gen_direction(param["num of directions"])

    print("Now is {}".format(datetime.now()))

    res_model = np.zeros(max_iters)
    res = np.zeros(max_iters)
    # res_ot = np.zeros(max_iters)

    cur_iter=0

    inv(param, max_iters)
