import numpy as np
from deepxde.data.function_spaces import FunctionSpace

# 用于生成3层*每层3个的热源空间
# 参考量
L_ref = 0.001  # mm
T_ref = 273.15  # K
k_ref = 398  #

class RandomCuboidHeatSourceMultiLayer(FunctionSpace):
    """
    在主长方体的若干层上表面附近放置小长方体热源 (体热源)。
    
    - 每层有 n_sources_per_layer 个热源 (默认 3)。
    - 一共 n_layers 层 (默认 3)，总共 n_layers * n_sources_per_layer 个热源。
    - 在 x,y 方向随机，z 上表面由 layers_top_z 指定 (热源顶面 = layers_top_z[i])。
    - 不允许热源之间重叠。
    - 热源强度可配置 (intensity)。
    
    主长方体尺寸:
        (Lx, Ly, Lz)
    小长方体尺寸:
        (dx, dy, dz)
    
    示例:
        space = RandomCuboidHeatSourceMultiLayer(
            Lx=3.2/1000,
            Ly=3.2/1000,
            Lz=0.1/1000,
            dx=1/1000,
            dy=1/1000,
            dz=0.01/1000,
            n_layers=3,
            n_sources_per_layer=3,
            layers_top_z=[0.1, 0.22, 0.34],  # 每层热源的上表面坐标
            intensity=1.0,
        )
    """

    def __init__(
        self,
        Lx=3.18,
        Ly=3.18,
        Lz=0.1,
        dx=1,
        dy=1,
        dz=0.01,
        n_layers=3,
        n_sources_per_layer=3,
        layers_top_z=None,
        intensity=1e11 * L_ref**2 / (k_ref * T_ref),
        max_attempts=100,
    ):
        """
        Args:
            Lx, Ly, Lz: 主长方体尺寸 (mm)
            dx, dy, dz: 小热源长宽高
            n_layers (int): 层数 (默认 3)
            n_sources_per_layer (int): 每层热源数 (默认 3)
            layers_top_z (list[float]): 每层热源的上表面 z 坐标 (默认 [0.1/1000, 0.22/1000, 0.34/1000]),
                单位同 Lz 等；若 None, 则使用默认值
            intensity (float): 热源强度 (默认为1e11)
            max_attempts (int): 放置热源时，为避免碰撞，最大尝试次数
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.n_layers = n_layers
        self.n_sources_per_layer = n_sources_per_layer
        if layers_top_z is None:
            # 默认 3 层：0.1, 0.22, 0.34
            self.layers_top_z = [0.1, 0.22, 0.34]
        else:
            self.layers_top_z = layers_top_z

        # 确保长度一致
        if len(self.layers_top_z) != self.n_layers:
            raise ValueError(
                f"heatsources_top_z长度({len(self.layers_top_z)})与n_layers({self.n_layers})不一致"
            )
        
        self.intensity = intensity
        self.max_attempts = max_attempts

        # 总热源数
        self.total_sources = self.n_layers * self.n_sources_per_layer
    
    #是否有更快速的办法？支持并行的方式？
    def random(self, size):
        """
        生成 `size` 个随机函数，每个函数包含 n_layers * n_sources_per_layer 个
        小长方体热源的 (x0, y0, z0)，返回形状 (size, total_sources*3)。

        x0 ∈ [0, Lx - dx]
        y0 ∈ [0, Ly - dy]
        z0 = layers_top_z[layer_idx] - dz  (使得热源顶面平齐到 layers_top_z[layer_idx])
        不允许热源之间重叠
        """
        x0_min, x0_max = -self.Lx/2, self.Lx/2 - self.dx
        y0_min, y0_max = -self.Ly/2, self.Ly/2 - self.dy
        
        feats = []
        for _ in range(size):
            placed_sources = []  # 存放已放置热源 (x0, y0, z0)
            # 共 n_layers，每层放 n_sources_per_layer 个热源
            for layer_idx in range(self.n_layers):
                # 计算本层 z0
                z_top = self.layers_top_z[layer_idx]
                z0_fixed = z_top - self.dz
                for __ in range(self.n_sources_per_layer):
                    attempt = 0
                    while True:
                        attempt += 1
                        if attempt > self.max_attempts:
                            raise ValueError(
                                f"无法在 {self.max_attempts} 次尝试中放置第 {__+1} 个热源（不重叠）"
                                f" (第{layer_idx+1}层)"
                            )
                        x0 = np.random.uniform(x0_min, x0_max)
                        y0 = np.random.uniform(y0_min, y0_max)
                        z0 = z0_fixed  # 固定

                        # 检测是否与已放置热源碰撞
                        overlap = False
                        for (xx, yy, zz) in placed_sources:
                            if is_overlap_3d(
                                x0, y0, z0, self.dx, self.dy, self.dz,
                                xx, yy, zz, self.dx, self.dy, self.dz
                            ):
                                overlap = True
                                break
                        if not overlap:
                            # 无重叠，可以放置
                            placed_sources.append((x0, y0, z0))
                            break

            # flatten 成 (x0,y0,z0)*total_sources
            fea = []
            for (xx, yy, zz) in placed_sources:
                fea.extend([xx, yy, zz])
            feats.append(fea)

        return np.array(feats, dtype=np.float32) #传出9个热源的坐标  [[x0,y0,z0]*total_sources]

    def eval_one(self, feature, x):
        """
        在单个点 x=(x, y, z) 上评估一个热源场 feature。shape=(total_sources*3,)
        如果 (x, y, z) 落入某个小长方体内 => val += self.intensity
        """
        n = self.total_sources
        val = 0.0
        for i in range(n):
            x0 = feature[3*i + 0]
            y0 = feature[3*i + 1]
            z0 = feature[3*i + 2]
            if (x0 <= x[0] <= x0 + self.dx and
                y0 <= x[1] <= y0 + self.dy and
                z0 <= x[2] <= z0 + self.dz):
                val += self.intensity
        return val

    def eval_batch(self, features, xs):
        """
        批量评估:
        features: shape = (n_funcs, total_sources*3)
        xs: shape = (n_points, 3)
        Returns: shape = (n_funcs, n_points)
        """
        n_funcs = len(features)
        n_points = len(xs)
        results = np.zeros((n_funcs, n_points),dtype=np.float32)
        
        for i in range(n_funcs):
            feat = features[i]
            for j in range(n_points):
                val = 0.0
                px, py, pz = xs[j]
                # 遍历所有(共 total_sources)热源
                for k in range(self.total_sources):
                    x0 = feat[3*k + 0]
                    y0 = feat[3*k + 1]
                    z0 = feat[3*k + 2]
                    if (x0 <= px <= x0 + self.dx and
                        y0 <= py <= y0 + self.dy and
                        z0 <= pz <= z0 + self.dz):
                        val += self.intensity
                results[i, j] = val
        return results
    
    
def is_overlap_3d(
    x0_1, y0_1, z0_1, dx1, dy1, dz1,
    x0_2, y0_2, z0_2, dx2, dy2, dz2
):
    """
    判断两个小长方体在 3D 空间中是否重叠。
    box1: [x0_1, x0_1+dx1] × [y0_1, y0_1+dy1] × [z0_1, z0_1+dz1]
    box2: [x0_2, x0_2+dx2] × [y0_2, y0_2+dy2] × [z0_2, z0_2+dz2]
    如果重叠则返回 True，否则返回 False。
    """
    overlap_x = not (x0_1 + dx1 <= x0_2 or x0_2 + dx2 <= x0_1)
    overlap_y = not (y0_1 + dy1 <= y0_2 or y0_2 + dy2 <= y0_1)
    overlap_z = not (z0_1 + dz1 <= z0_2 or z0_2 + dz2 <= z0_1)
    return overlap_x and overlap_y and overlap_z


#####################
# test
# space = RandomCuboidHeatSourceMultiLayer(
#     Lx=3.2/1000,
#     Ly=3.2/1000,
#     Lz=0.5/1000,     # 主长方体高 0.5mm(示例)
#     dx=1/1000,
#     dy=1/1000,
#     dz=0.01/1000,
#     n_layers=3,
#     n_sources_per_layer=3,
#     layers_top_z=[0.1/1000, 0.22/1000, 0.34/1000],
#     intensity=4,
# )

# # 1) 生成 size=2 的随机函数, each feature 包含 9 个热源(每个 3 个坐标), 共 9*3=27
# features = space.random(size=2)
# print("random features shape:", features.shape)  # (2, 27)

# # 2) 在单点上评估
# point = np.array([1.6/1000, 1.6/1000, 0.095/1000], dtype=np.float32)
# val0 = space.eval_one(features[0], point)
# print("第0个随机函数在point处的热源值=", val0)

# # 3) 批量评估
# xs = np.random.rand(500, 3)
# xs[:, 0] *= 3.2/1000
# xs[:, 1] *= 3.2/1000
# xs[:, 2] *= 0.1/1000
# vals = space.eval_batch(features, xs)
# print("vals shape =", vals.shape)  
# print(vals)
