import numpy as np
import sys, os

__all__ = [
    # "calc_mask_seq",
    "calc_mask_one_step",
    "gen_disc_mask",
    "l2_error_between_flow",
    "calc_aver_flow_vel",
    "hough_circles_3d",
    "block_print",
    "enable_print",
]


# def calc_mask_seq(init_source_mask, init_disc_info, flow):
#     # flow: nperiod*Nx*Ny*Nz*3 array
#     from devito import Function

#     source_mask_seq = [init_source_mask]
#     cur_disc_info = init_disc_info
#     for i in range(0, flow.shape[0]):
#         source_mask_seq.append(one_step(cur_disc_info, flow[i]))

#     return source_mask_seq


def calc_mask_one_step(disc_info, one_flow, dt):
    cur_disc_info = np.zeros(disc_info.shape)
    cur_disc_info[:, -1] = disc_info[:, -1]
    shape = one_flow.shape[0:3]
    # for i in range(0, disc_info.shape[0]):
    #     ind = tuple(disc_info[i, 0:3].astype(int))
    #     cur_disc_info[i, 0:3] = disc_info[i, 0:3] + dt * one_flow[ind][:]
    cur_disc_info[:, 0:3] = (
        disc_info[:, 0:3]
        + dt
        * one_flow[
            disc_info[:, 0].astype(int),
            disc_info[:, 1].astype(int),
            disc_info[:, 2].astype(int),
            :,
        ]
    )

    # delete the info of particles outside the domain
    del_ind = np.where(
        ~(
            (cur_disc_info[:, 0] < shape[0] - 1)
            * (cur_disc_info[:, 1] < shape[1] - 1)
            * (cur_disc_info[:, 2] < shape[2] - 1)
            * (cur_disc_info[:, 0] > 0)
            * (cur_disc_info[:, 1] > 0)
            * (cur_disc_info[:, 2] > 0)
        )
    )
    cur_disc_info = np.delete(cur_disc_info, del_ind, axis=0)
    print("Current number of particles: ", cur_disc_info.shape[0])
    return cur_disc_info


def gen_disc_mask(disc_info, model):
    mask = np.zeros(model.shape)
    xx, yy, zz = np.meshgrid(
        np.linspace(0, model.shape[0] - 1, model.shape[0]),
        np.linspace(0, model.shape[1] - 1, model.shape[1]),
        np.linspace(0, model.shape[2] - 1, model.shape[2]),
        indexing="ij",
    )
    for i in range(0, disc_info.shape[0]):
        mask[
            (xx - disc_info[i, 0]) ** 2
            + (yy - disc_info[i, 1]) ** 2
            + (zz - disc_info[i, 2]) ** 2
            <= disc_info[i, 3] ** 2
        ] = 1.0
    return mask


def l2_error_between_flow(flow_inv, flow_true, measure_freq, mask=None):
    if mask is None:
        mask = flow_inv > 1.0e-16
    elif mask.shape != flow_inv.shape[0:-1]:
        raise ValueError(
            "mask shape"
            + str(mask.shape)
            + " is not same as flow_inv shape"
            + str(flow_inv.shape)
            + "!"
        )
    else:
        mask = np.expand_dims(mask, axis=mask.ndim)
        mask = np.repeat(mask, 3, axis=mask.ndim - 1)
    err = (flow_inv - flow_true[0::measure_freq, :]) ** 2 * mask
    # print(np.sum((flow_true[0::measure_freq, :]) ** 2 * mask), np.sum(err))
    return np.sqrt(np.sum(err) / np.sum((flow_true[0::measure_freq, :]) ** 2 * mask))


def calc_aver_flow_vel(
    flow, xRange=(11, 91), yRange=(11, 91), zRange=(11, 91), mask=None
):
    # xRange, yRange, zRange: (int, int)

    if mask is None:
        return np.sum(
            flow[
                :,
                xRange[0] : xRange[1],
                yRange[0] : yRange[1],
                zRange[0] : zRange[1],
                :,
            ],
            axis=(0, 1, 2, 3),
        ) / (
            flow.shape[0]
            * (xRange[1] - xRange[0])
            * (yRange[1] - yRange[0])
            * (zRange[1] - zRange[0])
        )
    else:
        mask_expand = np.expand_dims(mask, axis=mask.ndim)
        mask_expand = np.repeat(mask_expand, 3, axis=mask_expand.ndim - 1)
        return np.sum(
            flow[
                :,
                xRange[0] : xRange[1],
                yRange[0] : yRange[1],
                zRange[0] : zRange[1],
                :,
            ]
            * mask_expand[
                :,
                xRange[0] : xRange[1],
                yRange[0] : yRange[1],
                zRange[0] : zRange[1],
                :,
            ],
            axis=(0, 1, 2, 3),
        ) / np.count_nonzero(
            mask[
                :,
                xRange[0] : xRange[1],
                yRange[0] : yRange[1],
                zRange[0] : zRange[1],
            ]
        )


# def imfindballs(img, radmin, radmax, detect_time=20):
#     for r in np.linspace(radmin, radmax, detect_time):
#         xx, yy, zz = np.meshgrid(
#             np.linspace(0, 2 * r, 2 * r + 1),
#             np.linspace(0, 2 * r, 2 * r + 1),
#             np.linspace(0, 2 * r, 2 * r + 1),
#             indexing="ij",
#         )

#         # a spherical mask with r radius
#         mask = ((xx - r) ** 2 + (yy - r) ** 2 + (zz - r) ** 2 <= r**2).astype(int)

#         from scipy.signal import convolve

#         score = convolve(img, mask, mode="same")
#     return


def hough_circles_blocks_3d(img, radius_range, threshold=0.1, blocks=1):
    circles = np.empty((0, 4))
    ix, iy, iz = (
        np.linspace(0, img.shape[0] - 1, blocks + 1).astype(np.int8),
        np.linspace(0, img.shape[1] - 1, blocks + 1).astype(np.int8),
        np.linspace(0, img.shape[2] - 1, blocks + 1).astype(np.int8),
    )

    max_rad = radius_range[1]

    for i in range(0, blocks):
        for j in range(0, blocks):
            for k in range(0, blocks):
                x_tmp = max(ix[i] - max_rad, 0)
                y_tmp = max(iy[j] - max_rad, 0)
                z_tmp = max(iz[k] - max_rad, 0)
                # print("Current shape: ", x_tmp, y_tmp, z_tmp)
                circles_tmp = hough_circles_3d(
                    img[
                        x_tmp : ix[i + 1] + max_rad,
                        y_tmp : iy[j + 1] + max_rad,
                        z_tmp : iz[k + 1] + max_rad,
                    ],
                    radius_range,
                    threshold=threshold,
                )
                circles_tmp[:, 0:3] = circles_tmp[:, 0:3] + [x_tmp, y_tmp, z_tmp]
                circles = np.vstack((circles, circles_tmp))
    return np.unique(circles, axis=0)


def hough_circles_3d(img, radius_range, threshold=0.1):
    # 输入的参数img是三维数组，其中每个元素是一个像素值。
    # radius_range是一个包含最小和最大半径的元组，表示要检测的圆的半径范围。

    # 1. 计算图像的梯度
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dz = np.zeros_like(img)

    dx[:, :-1, :] = img[:, 1:, :] - img[:, :-1, :]
    dy[:-1, :, :] = img[1:, :, :] - img[:-1, :, :]
    dz[:, :, :-1] = img[:, :, 1:] - img[:, :, :-1]

    gradient_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    gradient_magnitude /= np.max(gradient_magnitude)

    # 2. 构建半径空间
    min_radius, max_radius = radius_range
    radius_space = np.arange(min_radius, max_radius + 1)

    # 3. 构建累加器
    accumulator_shape = (img.shape[0], img.shape[1], img.shape[2], len(radius_space))
    accumulator = np.zeros(accumulator_shape, dtype=int)

    # 4. 遍历图像，并将所有边缘像素投票到累加器中
    edge_pixels = np.argwhere(gradient_magnitude > 0.1)  # 找到所有边缘像素的坐标
    for x, y, z in edge_pixels:
        for r_idx, r in enumerate(radius_space):
            # 计算圆心的可能位置
            x_c = np.arange(max(x - r, 0), min(x + r + 1, img.shape[0]))
            y_c = np.arange(max(y - r, 0), min(y + r + 1, img.shape[1]))
            z_c = np.arange(max(z - r, 0), min(z + r + 1, img.shape[2]))

            # 计算每个可能位置的投票值
            xx, yy, zz = np.meshgrid(x_c, y_c, z_c, indexing="ij")
            distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2)
            votes = np.where(np.abs(distance - r) <= 0.5, 1, 0)

            # 将投票值添加到累加器中
            accumulator[xx, yy, zz, r_idx] += votes

    # 5. 找到累加器中的峰值，并将每个峰值对应的圆加入结果列表
    circles = []
    while True:
        max_votes = np.max(accumulator)
        if max_votes < threshold * len(edge_pixels):
            break  # 如果没有峰值超过阈值，则退出循环

        # 找到当前最强的峰值
        x, y, z, r_idx = np.argwhere(accumulator == max_votes)[0]
        r = radius_space[r_idx]
        circles.append([x, y, z, r])

        # 将当前峰值对应的圆从累加器中删除
        x_c = np.arange(max(x - r, 0), min(x + r + 1, img.shape[0]))
        y_c = np.arange(max(y - r, 0), min(y + r + 1, img.shape[1]))
        z_c = np.arange(max(z - r, 0), min(z + r + 1, img.shape[2]))

        xx, yy, zz = np.meshgrid(x_c, y_c, z_c, indexing="ij")
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2)
        votes = np.where(distance <= r, 0, 1)
        # for r_idx in range(len(radius_space)):
        #     accumulator[xx, yy, zz, r_idx] *= votes

        # 找到所有边缘像素的坐标
        edge_pixels_cur = np.argwhere(gradient_magnitude[xx, yy, zz] > 0.1)
        for x, y, z in edge_pixels_cur:
            x += x_c[0]
            y += y_c[0]
            z += z_c[0]
            for r_idx, r in enumerate(radius_space):
                # 计算圆心的可能位置
                x_c_cur = np.arange(max(x - r, 0), min(x + r + 1, img.shape[0]))
                y_c_cur = np.arange(max(y - r, 0), min(y + r + 1, img.shape[1]))
                z_c_cur = np.arange(max(z - r, 0), min(z + r + 1, img.shape[2]))

                # 计算每个可能位置的投票值
                xx, yy, zz = np.meshgrid(x_c_cur, y_c_cur, z_c_cur, indexing="ij")
                distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2)
                votes = np.where(np.abs(distance - r) <= 0.5, 1, 0)

                # 将投票值添加到累加器中
                accumulator[xx, yy, zz, r_idx] -= votes

    return np.array(circles)


def nearest_point_search(
    img1,
    img2,
    radius_range=(5, 10),
    threshold=0.1,
    blocks=1,
    output_disc_info_mask=False,
):
    u = np.zeros(np.concatenate((img1.shape, [3])))
    disc_info_inv_cur = hough_circles_blocks_3d(
        img=img1, radius_range=radius_range, threshold=threshold, blocks=blocks
    )
    disc_info_inv_next = hough_circles_blocks_3d(
        img=img2, radius_range=radius_range, threshold=threshold, blocks=blocks
    )
    print("Number of detected particles: ", disc_info_inv_next.shape[0])
    for j in range(disc_info_inv_cur.shape[0]):
        point_cur = disc_info_inv_cur[j, 0:3]
        ind = np.argmin(
            np.linalg.norm(np.array([point_cur]) - disc_info_inv_next[:, 0:3], axis=1)
        )
        u[
            disc_info_inv_cur[j, 0].astype(np.int8),
            disc_info_inv_cur[j, 1].astype(np.int8),
            disc_info_inv_cur[j, 2].astype(np.int8),
            :,
        ] = (
            disc_info_inv_next[ind, 0:3] - point_cur
        )
    if not output_disc_info_mask:
        return u[:, :, :, 0], u[:, :, :, 1], u[:, :, :, 2]
    else:
        disc_info_mask = np.zeros(img1.shape)
        disc_info_inv_cur = disc_info_inv_cur.astype(np.int8)
        disc_info_mask[
            disc_info_inv_cur[:, 0], disc_info_inv_cur[:, 1], disc_info_inv_cur[:, 2]
        ] = 1
        return u[:, :, :, 0], u[:, :, :, 1], u[:, :, :, 2], disc_info_mask == 1


def block_print():
    sys.stdout = open(os.devnull, "w")


def enable_print():
    sys.stdout = sys.__stdout__
