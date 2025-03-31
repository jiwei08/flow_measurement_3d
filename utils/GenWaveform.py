import numpy as np
from devito import Function

__all__ = [
    "Gaussian_source",
    "Gaussian_derivative_source",
    "Ricker_source",
    "sinusoidal_source",
]


def Gaussian_source(f0, time_range, t):
    waveform = Function(
        name="waveform", shape=(time_range.num,), dimensions=(t,), dtype=np.float32
    )
    waveform.data[:] = 0

    supp = 6.0 / (np.pi * f0)
    nsupp = int(supp / time_range.step)
    waveform.data[0:nsupp] = np.exp(
        -np.pi**2 * f0**2 * (np.linspace(0, supp, nsupp) - supp / 2.0) ** 2
    )

    return waveform


def Gaussian_derivative_source(f0, time_range, t):
    waveform = Function(
        name="waveform", shape=(time_range.num,), dimensions=(t,), dtype=np.float32
    )
    waveform.data[:] = 0

    supp = 6.0 / (np.pi * f0)
    nsupp = int(supp / time_range.step)
    waveform.data[0:nsupp] = (
        2 * np.pi**2 * f0**2 * (np.linspace(0, supp, nsupp) - supp / 2.0)
    ) * np.exp(-np.pi**2 * f0**2 * (np.linspace(0, supp, nsupp) - supp / 2.0) ** 2)

    return waveform


def Ricker_source(f0, time_range, t):
    waveform = Function(
        name="waveform", shape=(time_range.num,), dimensions=(t,), dtype=np.float32
    )
    waveform.data[:] = 0

    supp = 6.0 / (np.pi * f0)
    nsupp = int(supp / time_range.step)
    waveform.data[0:nsupp] = (
        1 - 2 * np.pi**2 * f0**2 * (np.linspace(0, supp, nsupp) - supp / 2.0) ** 2
    ) * np.exp(-np.pi**2 * f0**2 * (np.linspace(0, supp, nsupp) - supp / 2.0) ** 2)

    return waveform


def sinusoidal_source(time_range, t):
    waveform = Function(
        name="waveform", shape=(time_range.num,), dimensions=(t,), dtype=np.float32
    )
    waveform.data[:] = 0

    supp = 100.0
    nsupp = int(supp / time_range.step)
    waveform.data[0:supp] = np.sin(np.linspace(0, np.pi, nsupp))

    return waveform
