import numpy as np
import torch

# 参考量
L_ref = 0.001  # mm
T_ref = 273.15  # K
k_ref = 398  #

def getDFactor(single_point_data, heat_sources, dx=1/1000,dy=1/1000, dz=0.01/1000):
    """
    根据单个点的坐标计算热源系数。

    参数：
    - single_point_data: 形状为 (batch_size, 3)，每个点的坐标 (x, y, z)。
    - heat_sources: 热源列表 [(x1, y1, z1), ...]。
    - dx, dy, dz: 热源范围的偏移量。

    返回：
    - D: 热源系数数组，形状为 (batch_size,)。
    """

    # dx，dy，dz是1，1，0.1 还是 3.2，3.2，0.1
    
    batch_size = single_point_data.shape[0]
    D = np.ones(batch_size)  # 初始化 D 为 1
    x, y, z = single_point_data[:, 0], single_point_data[:, 1], single_point_data[:, 2]

    # 规则 1: z < -0.02
    D[z < -0.02] = 131

    # 规则 2: 指定 z 区间
    cond2_mask = ((-0.02 < z) & (z < 0)) | \
                 ((0.1 < z) & (z < 0.12)) | \
                 ((0.22 < z) & (z < 0.24)) | \
                 ((0.34 < z) & (z < 0.36))
    D[cond2_mask] = 1

    # 规则 3: z > 0.36
    D[z > 0.36] = 384

    # 规则 4: 热源范围内 D = 131
    for x1, y1, z1 in heat_sources:
        in_heat_source = (
            (x <= x1 + dx) &
            (y <= y1 + dy) &
            (z <= z1 + dz)
        )

        # Note：这里需要确认符号 < 还是<=
        z_cond = ((0 < z) & (z < 0.1)) | \
                 ((0.12 < z) & (z < 0.22)) | \
                 ((0.24 < z) & (z < 0.34))

        D[in_heat_source & z_cond] = 131

    return D


def getDFactorV2(single_point_data, heat_sources, boundary_points, dx=1, dy=1, dz=0.01,device=None):
    """
    根据单个点的坐标计算热源系数。

    参数：
    - single_point_data: 形状为 (batch_size, 3)，每个点的坐标 (x, y, z)。
    - heat_sources: 热源列表，形状为 (size, n_sources*3)，其中每行表示 n_sources 个小热源的 (x0, y0, z0) 坐标。
    - dx, dy, dz: 热源范围的偏移量。

    返回：
    - D: 热源系数数组，形状为 (size, batch_size)，其中每一列表示对应批次每个点的热源系数。
    """
    if 'device' not in locals():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(single_point_data, torch.Tensor):
        single_point_data = single_point_data.detach().cpu().numpy()
    # 转换为float32避免精度问题
    single_point_data = single_point_data.astype(np.float32)
    x, y, z = single_point_data.T
    
    batch_size = single_point_data.shape[0]
    size = heat_sources.shape[0]  # 获取热源的数量（即 size）
    D = np.ones((size, batch_size))  # 初始化 D 为 (size, batch_size)，全为 1
    D.fill(1.5/k_ref)  # 初始化 D 为 1
    # x, y, z = single_point_data[:, 0], single_point_data[:, 1], single_point_data[:, 2]

    # 规则 1: z < -0.02 base line
    D[:, z < -0.02] = 130/k_ref

    # 规则 2: 指定 z 区间 C4Bump+Bondinglayer
    cond2_mask = ((-0.02 < z) & (z < 0)) | \
                 ((0.1 < z) & (z < 0.12)) | \
                 ((0.22 < z) & (z < 0.24)) | \
                 ((0.34 < z) & (z < 0.36))
    D[:, cond2_mask] = 1.5/k_ref

    # 规则 3: z > 0.36
    D[:, z > 0.36] = 398/k_ref

    # 规则 4: 热源范围内 D = 131
    # 遍历每个热源，heat_sources 为 (size, n_sources * 3)
    n_sources = heat_sources.shape[1] // 3  # 每个热源有3个维度
    for i in range(n_sources):
        x1 = heat_sources[:, 3*i]   # 热源的 x 坐标
        y1 = heat_sources[:, 3*i+1] # 热源的 y 坐标
        z1 = heat_sources[:, 3*i+2] # 热源的 z 坐标

        # # 判断当前热源范围内的条件
        # in_heat_source = (
        #     (x <= x1 + dx) & (x >= x1 - dx) &
        #     (y <= y1 + dy) & (y >= y1 - dy) &
        #     (z <= z1 + dz) & (z >= z1 - dz)
        # )

        # # 对应的 z 条件判断
        # z_cond = ((0 < z) & (z < 0.1)) | \
        #          ((0.12 < z) & (z < 0.22)) | \
        #          ((0.24 < z) & (z < 0.34))
                # 使用广播扩展 x, y, z 的维度，以便与 x1, y1, z1 对比
        in_heat_source = (
            (x[:, np.newaxis] >= x1) & (x[:, np.newaxis] <= x1 + dx) &
            (y[:, np.newaxis] >= y1) & (y[:, np.newaxis] <= y1 + dy) &
            (z[:, np.newaxis] >= z1) & (z[:, np.newaxis] <= z1 + dz)
        )

        # 对应的 z 条件判断
        z_cond = ((0 < z[:, np.newaxis]) & (z[:, np.newaxis] < 0.1)) | \
                 ((0.12 < z[:, np.newaxis]) & (z[:, np.newaxis] < 0.22)) | \
                 ((0.24 < z[:, np.newaxis]) & (z[:, np.newaxis] < 0.34))
        

        in_heat_source = in_heat_source.T  # 转置，变为 (size, batch_size)
        z_cond = z_cond.T  # 转置，变为 (size, batch_size)
        # 更新 D 的值
        D[in_heat_source & z_cond] = 131/k_ref


        #边界点的值
    D = np.repeat(D, 2, axis=0)


    # # 将行转换为字符串进行比较
    # x_as_str = np.array(['_'.join(map(str, row)) for row in x])
    # boundary_points_as_str = np.array(['_'.join(map(str, row)) for row in boundary_points])

    # # 找到交集
    # _, indices = np.intersect1d(x_as_str, boundary_points_as_str, return_indices=True)

    indices = []
    for point in boundary_points:
        index = np.where((single_point_data == point).all(axis=1))[0]  # 找到匹配点的索引
        indices.extend(index)  # 将索引加入结果列表

    indices = np.array(indices)  # 转换为 NumPy 数组
    # print(indices)  # indices 是 x 中匹配的行索引

    # 根据 z 的值进行赋值
    z_values = single_point_data[indices, 2]  # 获取对应索引的 z 值
    
    # D[:, indices[np.isclose(z_values, 0.36)]] = [398, 1.5]
    # D[:, indices[np.isclose(z_values, -0.02)]] = [1.5, 130]
    # D[:, indices[~(np.isclose(z_values, 0.36) | np.isclose(z_values, -0.02))]] = [131, 1.5]
    # 条件赋值
    # 使用容差比较代替精确相等
    mask_036 = np.isclose(z_values, 0.36, atol=1e-4, rtol=0)
    mask_neg002 = np.isclose(z_values, -0.02, atol=1e-4, rtol=0)
    
    if indices[mask_036].shape[0] !=0:
        D[:, indices[mask_036]] = np.array([[398/k_ref], [1.5/k_ref]]) 
    if indices[mask_neg002].shape[0] !=0:
        D[:, indices[mask_neg002]] = np.array([[1.5/k_ref], [130/k_ref]])
    if indices[~(mask_036 | mask_neg002)].shape[0] != 0:
        D[:, indices[~(mask_036 | mask_neg002)]] =  np.array([[131/k_ref], [1.5/k_ref]])
    
    # 将返回的D类型转换为tensor
    # D_tensor = torch.from_numpy(D).float().T
    
    # # 移动到主设备（将与模型的设备自动同步）
    # return D_tensor.to(device)  # 需要保证device变量全局可见，可在文件顶部声明
    device = single_point_data.device if hasattr(single_point_data, 'device') else torch.device("cpu")
    D_tensor = torch.from_numpy(D).float().T.to(device)
    # D_tensor = torch.from_numpy(D.T).float().to(device)
    # return D_tensor
    # D_tensor = torch.as_tensor(D.T, dtype=torch.float32, device=device)
    # D_tensor = torch.from_numpy(D.T).float().to(device)
    return D_tensor

def getDFactorV3(single_point_data, heat_sources, dx=1, dy=1, dz=0.01):
    """
    根据单个点的坐标计算热源系数。

    参数：
    - single_point_data: 形状为 (batch_size, 3)，每个点的坐标 (x, y, z)。
    - heat_sources: 热源列表，形状为 (size, n_sources*3)，其中每行表示 n_sources 个小热源的 (x0, y0, z0) 坐标。
    - dx, dy, dz: 热源范围的偏移量。

    返回：
    - D: 热源系数数组，形状为 (size, batch_size)，其中每一列表示对应批次每个点的热源系数。
    """
    batch_size = single_point_data.shape[0]
    size = heat_sources.shape[0]  # 获取热源的数量（即 size）
    D = np.ones((size, batch_size))  # 初始化 D 为 (size, batch_size)，全为 1
    D.fill(1.5/k_ref)
    x, y, z = single_point_data[:, 0], single_point_data[:, 1], single_point_data[:, 2]

    # 规则 1: z < -0.02 base line
    D[:, z < -0.02] = 130/k_ref

    # 规则 2: 指定 z 区间 C4Bump+Bondinglayer
    cond2_mask = ((-0.02 < z) & (z < 0)) | \
                 ((0.1 < z) & (z < 0.12)) | \
                 ((0.22 < z) & (z < 0.24)) | \
                 ((0.34 < z) & (z < 0.36))
    D[:, cond2_mask] = 1.5/k_ref

    # 规则 3: z > 0.36
    D[:, z > 0.36/1000] = 398/k_ref

    # 规则 4: 热源范围内 D = 131
    # 遍历每个热源，heat_sources 为 (size, n_sources * 3)
    n_sources = heat_sources.shape[1] // 3  # 每个热源有3个维度
    for i in range(n_sources):
        x1 = heat_sources[:, 3*i]   # 热源的 x 坐标
        y1 = heat_sources[:, 3*i+1] # 热源的 y 坐标
        z1 = heat_sources[:, 3*i+2] # 热源的 z 坐标

        # # 判断当前热源范围内的条件
        # in_heat_source = (
        #     (x <= x1 + dx) & (x >= x1 - dx) &
        #     (y <= y1 + dy) & (y >= y1 - dy) &
        #     (z <= z1 + dz) & (z >= z1 - dz)
        # )

        # # 对应的 z 条件判断
        # z_cond = ((0 < z) & (z < 0.1)) | \
        #          ((0.12 < z) & (z < 0.22)) | \
        #          ((0.24 < z) & (z < 0.34))
                # 使用广播扩展 x, y, z 的维度，以便与 x1, y1, z1 对比
        in_heat_source = (
            (x[:, np.newaxis] >= x1) & (x[:, np.newaxis] <= x1 + dx) &
            (y[:, np.newaxis] >= y1) & (y[:, np.newaxis] <= y1 + dy) &
            (z[:, np.newaxis] >= z1) & (z[:, np.newaxis] <= z1 + dz)
        )

        # 对应的 z 条件判断
        z_cond = ((0 < z[:, np.newaxis]) & (z[:, np.newaxis] < 0.1)) | \
                 ((0.12 < z[:, np.newaxis]) & (z[:, np.newaxis] < 0.22)) | \
                 ((0.24 < z[:, np.newaxis]) & (z[:, np.newaxis] < 0.34))
        

        in_heat_source = in_heat_source.T  # 转置，变为 (size, batch_size)
        z_cond = z_cond.T  # 转置，变为 (size, batch_size)
        # 更新 D 的值
        D[in_heat_source & z_cond] = 131/k_ref


        #边界点的值
    D = np.repeat(D, 2, axis=0)

    return D

if __name__ == "__main__":

    single_point_data = np.random.uniform(0,1,(119, 3))
    heat_sources = np.random.uniform(0, 1, (3, 27))
    # heat_sources = [(0.5, 0.5, 0.15)]  # 一个热源
    dx, dy, dz = 0.1, 0.1, 0.1  # 偏移量

    D = getDFactorV2(single_point_data, heat_sources, dx, dy, dz)
    print(D)  # 输出热源系数
