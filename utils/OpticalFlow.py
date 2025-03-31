import numpy as np
import matlab

# import numba as nb

__all__ = ["optical_flow", "imageDerivative3d"]


# @nb.jit(nopython=True)
def optical_flow(image1, image2, r=2, matlab_eng=None, method="farneback3d"):
    print("Optical flow method: " + method)
    image1 = image1 * (image1 > 1.0e-16)
    image2 = image2 * (image2 > 1.0e-16)
    if method in ["LK", "HS", "HS_seq", "LKW", "LKPR"]:
        if matlab_eng is not None:
            matlab_eng.addpath(matlab_eng.genpath("./utils"))
            image1m = matlab.double(image1.tolist())
            image2m = matlab.double(image2.tolist())
            if method == "LK":
                ux, uy, uz = matlab_eng.LK3D(image1m, image2m, nargout=3)
            elif method == "HS":
                ux, uy, uz = matlab_eng.HS3D(image1m, image2m, nargout=3)
            elif method == "HS_seq":
                ux, uy, uz = matlab_eng.HS3D_seq(image1m, image2m, nargout=3)
            elif method == "LKW":
                ux, uy, uz = matlab_eng.LKW3D(image1m, image2m, nargout=3)
            elif method == "LKPR":
                ux, uy, uz = matlab_eng.LKPR3D(image1m, image2m, nargout=3)
            ux = np.array(ux)
            uy = np.array(uy)
            uz = np.array(uz)
        else:
            height, width, depth = image1.shape

            ux = np.zeros(image1.shape)
            uy = np.zeros(image1.shape)
            uz = np.zeros(image1.shape)

            Ix, Iy, Iz, It = imageDerivative3d(image1, image2)

            for i in range(r, height - r):
                for j in range(r, width - r):
                    for k in range(r, depth - r):
                        blockOfIx = Ix[
                            i - r : i + r + 1, j - r : j + r + 1, k - r : k + r + 1
                        ]
                        blockOfIy = Iy[
                            i - r : i + r + 1, j - r : j + r + 1, k - r : k + r + 1
                        ]
                        blockOfIz = Iz[
                            i - r : i + r + 1, j - r : j + r + 1, k - r : k + r + 1
                        ]
                        blockOfIt = It[
                            i - r : i + r + 1, j - r : j + r + 1, k - r : k + r + 1
                        ]

                        A = np.zeros((3, 3))
                        B = np.zeros((3, 1))

                        A[0, 0] = np.sum(blockOfIx**2)
                        A[0, 1] = np.sum(blockOfIx * blockOfIy)
                        A[0, 2] = np.sum(blockOfIx * blockOfIz)

                        A[1, 0] = np.sum(blockOfIy * blockOfIx)
                        A[1, 1] = np.sum(blockOfIy**2)
                        A[1, 2] = np.sum(blockOfIy * blockOfIz)

                        A[2, 0] = np.sum(blockOfIz * blockOfIx)
                        A[2, 1] = np.sum(blockOfIz * blockOfIy)
                        A[2, 2] = np.sum(blockOfIz**2)

                        B[0, 0] = np.sum(blockOfIx * blockOfIt)
                        B[1, 0] = np.sum(blockOfIy * blockOfIt)
                        B[2, 0] = np.sum(blockOfIz * blockOfIt)

                        A = A * (A > 1.0e-16)
                        V = np.linalg.pinv(A) @ (-B)

                        ux[i, j, k] = V[0]
                        uy[i, j, k] = V[1]
                        uz[i, j, k] = V[2]
    elif method == "farneback3d":
        import farneback3d

        optflow = farneback3d.Farneback(
            levels=10,
            num_iterations=1,
            winsize=20,
            poly_n=5,
            quit_at_level=-1,
            use_gpu=True,
            fast_gpu_scaling=False,
        )
        uz, uy, ux = optflow.calc_flow(
            1000 * image2.astype(np.float32), 1000 * image1.astype(np.float32)
        )
    elif method == "of3d_farneback3d":
        import opticalflow3D

        farneback = opticalflow3D.Farneback3D(
            iters=5,
            num_levels=5,
            scale=0.5,
            spatial_size=7,
            presmoothing=7,
            filter_type="box",
            filter_size=21,
        )
        uz, uy, ux, _ = farneback.calculate_flow(
            1000 * image2.astype(np.float32),
            1000 * image1.astype(np.float32),
            total_vol=image1.shape,
        )
    elif method == "proj_optical_flow":
        if matlab_eng is None:
            raise ValueError("Must use matlab to calculate!")
        else:
            img1_x = matlab.double(np.sum(image1, axis=0).tolist())
            img1_y = matlab.double(np.sum(image1, axis=1).tolist())
            img1_z = matlab.double(np.sum(image1, axis=2).tolist())

            img2_x = matlab.double(np.sum(image2, axis=0).tolist())
            img2_y = matlab.double(np.sum(image2, axis=1).tolist())
            img2_z = matlab.double(np.sum(image2, axis=2).tolist())

            matlab_eng.addpath(matlab_eng.genpath("./utils"))

            # alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations
            para = matlab.double([0.012, 0.75, 10, 7, 1, 30])
            v1, w1, _ = matlab_eng.Coarse2FineTwoFrames(img1_x, img2_x, para, nargout=3)
            u1, w2, _ = matlab_eng.Coarse2FineTwoFrames(img1_y, img2_y, para, nargout=3)
            u2, v2, _ = matlab_eng.Coarse2FineTwoFrames(img1_z, img2_z, para, nargout=3)

            ux = (
                np.tile(u1, [image1.shape[1], 1, 1]).transpose(1, 0, 2)
                + np.tile(u2, [image1.shape[2], 1, 1]).transpose(1, 2, 0)
            ) / 2.0
            uy = (
                np.tile(v1, [image1.shape[0], 1, 1])
                + np.tile(v2, [image1.shape[2], 1, 1]).transpose(1, 2, 0)
            ) / 2.0
            uz = (
                np.tile(w1, [image1.shape[0], 1, 1])
                + np.tile(w2, [image1.shape[1], 1, 1]).transpose(1, 0, 2)
            ) / 2.0

    return ux, uy, uz


def imageDerivative3d(image1, image2):
    dx = np.zeros((2, 2, 2))
    dx[:, :, 0] = np.array([[-1, -1], [1, 1]])
    dx[:, :, 1] = np.array([[-1, -1], [1, 1]])
    dx = 0.25 * dx

    dy = np.zeros((2, 2, 2))
    dy[:, :, 0] = np.array([[-1, 1], [-1, 1]])
    dy[:, :, 1] = np.array([[-1, 1], [-1, 1]])
    dy = 0.25 * dy

    dz = np.zeros((2, 2, 2))
    dz[:, :, 0] = np.array([[-1, -1], [-1, -1]])
    dz[:, :, 1] = np.array([[1, 1], [1, 1]])
    dz = 0.25 * dz

    dt = np.ones((2, 2, 2))
    dt = 0.25 * dt

    from scipy.signal import convolve

    Ix = 0.5 * (convolve(image1, dx) + convolve(image2, dx))
    Iy = 0.5 * (convolve(image1, dy) + convolve(image2, dy))
    Iz = 0.5 * (convolve(image1, dz) + convolve(image2, dz))
    It = 0.5 * (convolve(image1, dt) - convolve(image2, dt))

    Ix = Ix[0:-1, 0:-1, 0:-1]
    Iy = Iy[0:-1, 0:-1, 0:-1]
    Iz = Iz[0:-1, 0:-1, 0:-1]
    It = It[0:-1, 0:-1, 0:-1]

    return Ix, Iy, Iz, It
