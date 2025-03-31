import numpy as np
import scipy.io as scio
import os

__all__ = [
    "write_npy_shotdata",
    "write_npy_seq_shotdata",
    "load_npy_shotdata",
    "load_npy_seq_shotdata",
    "write_mat_inversion",
    "write_mat_seq_inversion",
    "load_mat_maskdata",
    "load_mat_mask_discinfo",
    "load_npy_config",
    "write_npy_config",
    "load_npy_flowdata",
    "read_csv_fluent",
    "write_npy_seq_inversion",
]


def write_npy_shotdata(npy_file, rec):
    data_dict = {
        "time_range": rec.time_range,
        "npoint": rec.npoint,
        "data": np.asarray(rec.data),
        "coordinates_data": np.asarray(rec.coordinates.data),
    }
    np.save(npy_file, data_dict)


def write_npy_seq_shotdata(npy_file, data, rec):
    data_dict = {
        "npoint": rec.npoint,
        "time_range": rec.time_range,
        "rec_data": data,
        "coordinates_data": np.asarray(rec.coordinates.data),
    }
    np.save(npy_file, data_dict)


def load_npy_shotdata(npy_file, grid, name="rec", **kwargs):
    from examples.seismic import Receiver

    if os.path.splitext(npy_file)[1]=='.npy':
        data_dict = np.load(npy_file, allow_pickle=True).item()
    elif os.path.splitext(npy_file)[1]=='.npz':
        data_dict = np.load(npy_file, allow_pickle=True)['arr_0'].item()
    else:
        raise ValueError('Invalid file format.')

    rec = Receiver(
        name=name,
        grid=grid,
        time_range=data_dict["time_range"],
        npoint=data_dict["npoint"],
        coordinates=data_dict["coordinates_data"],
        data=data_dict['data']
    )

    return rec


def load_npy_seq_shotdata(npy_file, grid, name="rec"):
    from examples.seismic import Receiver

    data_dict = np.load(npy_file, allow_pickle=True).item()
    rec_data = data_dict["rec_data"]

    rec = Receiver(
        name=name,
        grid=grid,
        time_range=data_dict["time_range"],
        npoint=data_dict["npoint"],
        coordinates=data_dict["coordinates_data"],
    )

    return rec, rec_data


def load_npy_maskdata(npy_file, model, name="source_mask"):
    from devito import Function

    nbl = model.nbl

    maskdata = np.load(npy_file, allow_pickle=True).item()
    maskdata["sourceMask"] = maskdata["sourceMask"].astype(np.float32)
    if "discInfo" in maskdata:
        maskdata["discInfo"] = maskdata["discInfo"].astype(np.float32)

    if maskdata["sourceMask"].shape != model.shape:
        raise TypeError("Loading npy data fails: shapes dismatch! (data shape:{},model shape:{})".format(maskdata['sourceMask'].shape,model.shape))
    else:
        source_mask = Function(name=name, grid=model.grid)
        source_mask.data[nbl:-nbl, nbl:-nbl, nbl:-nbl] = maskdata["sourceMask"][:]

        if "discInfo" in maskdata:
            return source_mask, maskdata["discInfo"]
        else:
            return source_mask


def load_mat_maskdata(mat_file, model, name="source_mask"):
    from devito import Function

    nbl = model.nbl

    maskdata = scio.loadmat(mat_file)
    maskdata["sourceMask"] = maskdata["sourceMask"].astype(np.float32)

    if maskdata["sourceMask"].shape != model.shape:
        raise TypeError("Loading mat data fails: shapes dismatch!")
    else:
        source_mask = Function(name=name, grid=model.grid)
        source_mask.data[nbl:-nbl, nbl:-nbl, nbl:-nbl] = maskdata["sourceMask"][:]

        return source_mask


def load_mat_mask_discinfo(mat_file, model, name="source_mask"):
    from devito import Function

    nbl = model.nbl

    maskdata = scio.loadmat(mat_file)
    maskdata["sourceMask"] = maskdata["sourceMask"].astype(np.float32)
    maskdata["discInfo"] = maskdata["discInfo"].astype(np.float32)

    if maskdata["sourceMask"].shape != model.shape:
        print("Loading mat data fails: shapes dismatch!")
    else:
        source_mask = Function(name=name, grid=model.grid)
        source_mask.data[nbl:-nbl, nbl:-nbl, nbl:-nbl] = maskdata["sourceMask"][:]

        return source_mask, maskdata["discInfo"]


def write_mat_inversion(mat_file, result, model, **kwargs):
    rst = np.reshape(result[0], model.shape)
    scio.savemat(mat_file, dict({"rst": rst}, **kwargs))


def write_mat_seq_inversion(mat_file, result):
    scio.savemat(mat_file, {"seq_inv": result})


def write_npy_seq_inversion(npy_file, result):
    np.save(npy_file, result)


def write_npy_config(
    # file, model, waveform, direction, isRiver, measure_freq=None, dt_flow=None
    file,
    **kwargs,
):
    # if measure_freq is None and dt_flow is None:
    #     data_dict = {
    #         "model": model,
    #         "waveform": waveform,
    #         "direction": direction,
    #         "isRiver": isRiver,
    #     }
    # else:
    #     data_dict = {
    #         "model": model,
    #         "waveform": waveform,
    #         "direction": direction,
    #         "isRiver": isRiver,
    #         "measure_freq": measure_freq,
    #         "dt_flow": dt_flow,
    #     }

    np.save(file, kwargs)


def load_npy_config(file):
    data_dict = np.load(file, allow_pickle=True).item()

    model = data_dict["model"]
    waveform = data_dict["waveform"]
    direction = data_dict["direction"]
    isRiver = data_dict["isRiver"]
    if "measure_freq" in data_dict and "dt_flow" in data_dict:
        measure_freq = data_dict["measure_freq"]
        dt_flow = data_dict["dt_flow"]
        return model, waveform, direction, isRiver, measure_freq, dt_flow
    else:
        return model, waveform, direction, isRiver


def load_npy_flowdata(npy_file, model):
    flow = np.load(npy_file, allow_pickle=True)

    if flow.shape[1:4] != model.shape:
        print("Loading npy data fails: shapes dismatch!")
    else:
        return flow


def write_npy_flowdata(npy_file, flowdata):
    np.save(npy_file, flowdata)


def read_csv_fluent(filename, model=None, ylim=(0.0, 1.0), xlim=(0.0,1.0), nperiod=None):
    import csv

    coord_vel = np.empty((0, 6))
    data = list(csv.reader(open(filename)))
    print('Number of lines: {:d}'.format(len(data)))

    # select nodes whose y-coordinates belong to ylim and x-coordinate > 0
    for idx, line in enumerate(data):
        if idx % 10000==1:
            print('Line {:d} ...'.format(idx))
        if idx > 5:
            if (
                ylim[0] <= float(line[1])
                and float(line[1]) <= ylim[1]
                and float(line[0]) >= xlim[0]
                and float(line[0]) <= xlim[1]
            ):
                coord_vel = np.append(
                    coord_vel, [[float(x) for x in line[0:6]]], axis=0
                )

    print('Number of nodes: {:d}'.format(len(coord_vel)))
    print('x_min: {:.4f} m, x_max: {:.4f} m'.format(np.min(coord_vel[:,0]),np.max(coord_vel[:,0])))
    print('y_min: {:.4f} m, y_max: {:.4f} m'.format(np.min(coord_vel[:,1]),np.max(coord_vel[:,1])))
    print('z_min: {:.4f} m, z_max: {:.4f} m'.format(np.min(coord_vel[:,2]),np.max(coord_vel[:,2])))
    # # sort on 'x', 'y', 'z' columns
    # coord_vel[:, 0:3] = np.round(coord_vel[:, 0:3], 2)
    # coord_vel = coord_vel[coord_vel[:, 2].argsort()]
    # coord_vel = coord_vel[coord_vel[:, 1].argsort(kind="mergesort")]
    # coord_vel = coord_vel[coord_vel[:, 0].argsort(kind="mergesort")]

    # N = coord_vel.shape[0]
    # n = int(np.cbrt(N))
    # u = np.reshape(coord_vel[:, 3], (n, n, n))
    # v = np.reshape(coord_vel[:, 4], (n, n, n))
    # w = np.reshape(coord_vel[:, 5], (n, n, n))

    # interpolation on unstructed mesh
    if model is None:
        shape=(101, 101, 101)
        domain_size=(1.0, 1.0, 1.0)
    else:
        shape=model.shape
        domain_size=model.domain_size
    flow = np.zeros(np.concatenate((shape, [3])))

    xx, yy, zz = np.meshgrid(
        np.linspace(0.0, domain_size[0], shape[0]),
        np.linspace(*ylim, shape[1]),
        np.linspace(0.0, domain_size[2], shape[2]),
        indexing="ij",
    )

    from scipy.interpolate import LinearNDInterpolator

    interp = LinearNDInterpolator(coord_vel[:, 0:3], coord_vel[:, 3], fill_value=0.0)
    flow[:, :, :, 0] = interp(
        xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)
    ).reshape(shape)
    interp = LinearNDInterpolator(coord_vel[:, 0:3], coord_vel[:, 4], fill_value=0.0)
    flow[:, :, :, 1] = interp(
        xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)
    ).reshape(shape)
    interp = LinearNDInterpolator(coord_vel[:, 0:3], coord_vel[:, 5], fill_value=0.0)
    flow[:, :, :, 2] = interp(
        xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)
    ).reshape(shape)

    if nperiod is None:
        return flow
    else:
        flow = np.array([flow])
        return np.repeat(flow, nperiod, axis=0)
