import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.ticker as mticker

import matplotlib

# matplotlib.rcParams["font.size"] = 20

__all__ = ["plot_waveform", "plot_shot_data", "plot_mat_by_matlab", "plot_flow"]


def plot_waveform(waveform, time_range=None, filename=None):
    fig, ax = plt.subplots()
    plt.plot(waveform.data)
    plt.xlabel(
        "time step"
        if time_range is None
        else "time step (" + str(time_range.step) + ")"
    )
    plt.ylabel("magnitude")
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def plot_shot_data(recdata, model, time_range, filename=None):
    # from examples.seismic import plot_shotrecord
    extent = [0, recdata.shape[1], time_range.stop * 1000, time_range.start * 1000]
    plt.figure()
    plt.imshow(
        recdata,
        vmin=-np.abs(recdata).max(),
        vmax=np.abs(recdata).max(),
        cmap=cm.seismic,
        extent=extent,
    )

    plt.xlabel("rec")
    plt.ylabel("time(ms)")

    # y_ticks, _ = plt.yticks()
    # plt.yticks(
    #     ticks=y_ticks,
    #     labels=[
    #         str(i) + " ms"
    #         for i in np.around(
    #             np.linspace(
    #                 time_range.start * 1000, time_range.stop * 1000, len(y_ticks)
    #             ),
    #             2,
    #         )
    #     ],
    # )
    # plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f km"))
    # plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f ms"))
    plt.colorbar()
    plt.axis("auto")

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def plot_mat_by_matlab(filename):
    import os

    # os.system(
    #     "nohup matlab -nodesktop -nosplash -r filename="
    #     + filename
    #     + ",</data/lijiwei/flow-measurement3d/utils/plot_mat3d.m >output.log 2>&1 &"
    # )


def plot_flow(flow, title=None, f=10, saved_file=None,reversed_yaxis=False):
    # flow : (ndims,3) array or (3,ndims) array
    ax = plt.figure().add_subplot(projection="3d")

    if flow.shape[-1] == 3:
        x, y, z = np.meshgrid(
            np.arange(0, flow.shape[0], f),
            np.arange(0, flow.shape[1], f),
            np.arange(0, flow.shape[2], f),
            indexing='ij',
        )
        color=np.sqrt(np.sum(flow[::f,::f,::f,:]**2,axis=-1))
        ax.quiver(
            x,
            y,
            z,
            flow[::f, ::f, ::f, 0],
            -flow[::f, ::f, ::f, 1] if reversed_yaxis else flow[::f, ::f, ::f, 1],
            flow[::f, ::f, ::f, 2],
            length=3,
            arrow_length_ratio=0.5,
            # normalize=True,
            linewidths=0.7,
        )
        ax.set_xlim(0, flow.shape[0] - 1)
        if reversed_yaxis:
            ax.set_ylim(flow.shape[1] - 1, 0)
        else:
            ax.set_ylim(0, flow.shape[1] - 1)
        ax.set_zlim(0, flow.shape[2] - 1)
        ax.set_box_aspect(flow.shape[:3])
    elif flow.shape[0] == 3:
        x, y, z = np.meshgrid(
            np.arange(0, flow.shape[1], f),
            np.arange(0, flow.shape[2], f),
            np.arange(0, flow.shape[3], f),
            indexing='ij',
        )
        ax.quiver(
            x,
            y,
            z,
            flow[0, ::f, ::f, ::f],
            -flow[1, ::f, ::f, ::f] if reversed_yaxis else flow[::f, ::f, ::f, 1],
            flow[2, ::f, ::f, ::f],
            length=3,
            arrow_length_ratio=0.5,
            # normalize=True,
            linewidths=0.7,
        )
        ax.set_xlim(0, flow.shape[1] - 1)
        if reversed_yaxis:
            ax.set_ylim(flow.shape[1] - 1, 0)
        else:
            ax.set_ylim(0, flow.shape[1] - 1)
        ax.set_zlim(0, flow.shape[3] - 1)
        ax.set_box_aspect(flow.shape[1:])
    ax.set_xlabel("Transverse")
    ax.set_ylabel("Flow direction")
    ax.set_zlabel("Depth")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(30, -120)
    if title:
        plt.title(title)
    if saved_file is not None:
        plt.savefig(saved_file, dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.close()
