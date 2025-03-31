import numpy as np

from numba import jit


# @jit(nopython=True)
def meshgrid(x, y, z, indexing="ij"):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                xx[i, j, k] = x[i]  # change to x[k] if indexing xy
                yy[i, j, k] = y[j]  # change to y[j] if indexing xy
                zz[i, j, k] = z[k]  # change to z[i] if indexing xy
    return zz, yy, xx


# @jit(nopython=True)
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

            # for ix in x_c:
            #     for iy in y_c:
            #         for iz in z_c:
            #             if (
            #                 np.abs(
            #                     np.sqrt((ix - x) ** 2 + (iy - y) ** 2 + (iz - z) ** 2)
            #                     - r
            #                 )
            #                 <= 0.5
            #             ):
            #                 accumulator[ix, iy, iz, r_idx] += 1

    # plt.imshow(accumulator[25, :, :, -2])
    # plt.colorbar()
    # plt.savefig("accumulator.png")
    # plt.close()

    # 5. 找到累加器中的峰值，并将每个峰值对应的圆加入结果列表
    circles = []
    while True:
        max_votes = np.max(accumulator)
        if max_votes < threshold * len(edge_pixels) or max_votes < 0:
            break  # 如果没有峰值超过阈值，则退出循环

        # 找到当前最强的峰值
        x, y, z, r_idx = np.argwhere(accumulator == max_votes)[0]
        r = radius_space[r_idx]
        circles.append((x, y, z, r))

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

                # for ix in x_c_cur:
                #     for iy in y_c_cur:
                #         for iz in z_c_cur:
                #             if (
                #                 np.abs(
                #                     np.sqrt(
                #                         (ix - x) ** 2
                #                         + (iy - y) ** 2
                #                         + (iz - z) ** 2
                #                     )
                #                     - r
                #                 )
                #                 <= 0.5
                #             ):
                #                 accumulator[ix, iy, iz, r_idx] -= 1

    # plt.imshow(accumulator[15, :, :, 1])
    # plt.colorbar()
    # plt.savefig("accumulator_minus.png")
    # plt.close()

    return np.array(circles)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 创建一个包含三个圆的三维数组
    shape = (100, 100, 100)
    img = np.zeros(shape)
    x, y, z = np.indices(shape)

    # r=2
    # for i in range(18, 83, 8):
    #     for j in range(18, 83, 8):
    #         for k in range(18, 83, 8):
    #             img[(x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2 < r**2] = 1.0

    circle1 = (x - 25.5) ** 2 + (y - 25) ** 2 + (z - 25) ** 2 < 9**2
    circle2 = (x - 15) ** 2 + (y - 35) ** 2 + (z - 20) ** 2 < 6**2
    circle3 = (x - 35) ** 2 + (y - 35) ** 2 + (z - 30) ** 2 < 7**2
    img[circle1] = 1
    img[circle2] = 1
    img[circle3] = 1

    # # 创建一个包含一个圆的三维数组
    # img = np.zeros((10, 10, 10))
    # x, y, z = np.indices((10, 10, 10))
    # circle1 = (x - 5) ** 2 + (y - 5) ** 2 + (z - 5) ** 2 < 3**2
    # img[circle1] = 1

    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(img, sigma=1)

    # 检测圆并打印结果
    print("开始检测...")
    circles = hough_circles_3d(img, (5, 10), threshold=0.001)
    print("检测到的圆数量：", circles.shape[0])
    print("检测到的圆：\n", circles)

    # # 在三维图形中可视化图像
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # x, y, z = np.nonzero(img)
    # ax.scatter(x, y, z, marker="o")
    # plt.savefig("hough.png")
    # plt.close()
