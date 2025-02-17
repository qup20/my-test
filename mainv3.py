# # %%
# import deepxde
# import deepxde as dde
# from deepxde.nn import NN
# from matplotlib.pylab import f
# import torch
# import numpy as np
# import time
# import datetime
# import pyvista as pv
# from MyCuboid import *
# from DeepONetV2 import DeepONet_V2
# from RandomCuboidHeatSourceMultiLayer import RandomCuboidHeatSourceMultiLayer
# from CustomPDEOperator import CustomPDEOperator
# import pyvista as pv
# from utils import *
# import os
# from parallel import DataParallelWrapper
# #torchrun --nproc_per_node=4 mainv3.py

# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"当前运行设备: {device}")
# # %%

# 参考量
# L_ref = 0.001  # mm
# T_ref = 273.15  # K
# k_ref = 398  #
# q_heat = 1e11 * L_ref**2 / (k_ref * T_ref)
# DD = 131/k_ref

# # 定义PDE方程
# def pde_heat(x, y,var):
#     """
#     x: 输入坐标 (N, 3) -> (x, y, z)
#     y: 网络输出 (N, 1) -> T(x, y, z)
#     f: 额外的源项或可选参数, 例如热源 f(x,y,z)
    
#     返回值: PDE 残差, 在稳态情况下要求残差=0
#     """
#     # 假设导热系数 D 是常数，也可以根据需要改成函数形式
#     # D = 131
#     # dx，dy，dz是1，1，0.1 还是 3.2，3.2，0.1
#     D = y[1][0]
#     y = y[0][:, 0, :]
#     Qheat = torch.zeros_like(D)
#     Qheat[D == DD] = q_heat

#     # 计算二阶偏导数 (拉普拉斯算子)
#     # T_xx, T_yy, T_zz 分别是对 x, y, z 的二阶偏导
#     T_xx = dde.grad.hessian(y, x, i=0, j=0)
#     T_yy = dde.grad.hessian(y, x, i=1, j=1)
#     T_zz = dde.grad.hessian(y, x, i=2, j=2)

#     # 稳态热传导方程: -D * (T_xx + T_yy + T_zz) = v
#     # 若 v=0, 表示无热源
#     # 稳态热传导方程: -D * (T_xx + T_yy + T_zz) = v
#     # 若 v=0, 表示无热源
#     lower_bound = 297.15
#     upper_bound = 380
#     y_pred = (y[:, 0]+1)*273.15  # 假设 y 是模型的输出
    
#     # 计算超出范围的惩罚项
#     penalty = torch.relu(lower_bound - y_pred) ** 2 + torch.relu(y_pred - upper_bound) ** 2
#     return  D * (T_xx + T_yy + T_zz) + Qheat + penalty * 0.1



# # %%
# import deepxde as dde
# import numpy as np
# from CustomBC import CustomNeumannBC, CustomRobinBC

# # 定义几何体
# geo_jiban = MyCuboid([-5, -5, -1.02], [5, 5, -0.02])
# geo_chiplayer = MyCuboid([-1.6, -1.6, -0.02], [1.6, 1.6, 0.36])
# geo_heatspead = MyCuboid([-5, -5, 0.36], [5, 5, 2.36])
# geo_heatsink = MyCuboid([-7.5, -7.5, 2.36], [7.5, 7.5, 6.36])



# # 组合几何体
# geo = dde.geometry.CSGUnion(dde.geometry.CSGUnion(geo_jiban, geo_chiplayer), geo_heatspead)
# geo = dde.geometry.CSGUnion(geo, geo_heatsink)


# # 侧壁 定义边界条件
# def boundary(x, on_boundary):
#     if not on_boundary:
#         return False
    
#     # 排除 z = -1.02/1000 和 z = 6.36/1000 的点
#     z_min = -1.02
#     z_max = 6.36
#     return not (dde.utils.isclose(x[2], z_min) or dde.utils.isclose(x[2], z_max))
# def boundary_top(x, on_boundary):
#     return on_boundary and dde.utils.isclose(x[2], 6.36)

# def boundary_bot(x, on_boundary):
#     return on_boundary and dde.utils.isclose(x[2], -1.02)

# def robin_func_top(x, y):
#     # y = y[:, 0, :]
#     D = 398 / k_ref
#     H = 5000 * L_ref / k_ref
#     # A = 15 * 15 / 1000000
#     t_ab = (273.15 + 25) / T_ref - 1
#     return -H / D * (y - t_ab)


# def robin_func_bot(x, y):
#     # y = y[:, 0, :]
#     D = 130 / k_ref
#     H = 10 * L_ref / k_ref
#     # A = 10 * 10 / 1000000
#     t_ab = (273.15 + 25) / T_ref - 1
#     return -H / D * (y - t_ab)


# bc_top = CustomRobinBC(geom=geo, func=robin_func_top, on_boundary=boundary_top)
# bc_bot = CustomRobinBC(geom=geo, func=robin_func_bot, on_boundary=boundary_bot)


# # 侧壁法向梯度为零的条件
# bc_cebi = CustomNeumannBC(geo, lambda x: 0, boundary)

# data = dde.data.PDE(
#     geo,
#     pde=pde_heat,
#     bcs=[bc_cebi, bc_top, bc_bot],
#     num_domain=100,
#     num_boundary=100,
#     train_distribution="uniform",
#     anchors=None
# )

# # %%
# from MyCuboid import MyCuboid
# space = RandomCuboidHeatSourceMultiLayer(
#     # Lx=3.2/1000,
#     # Ly=3.2/1000,
#     Lz=0.1,
#     dx=1,
#     dy=1,
#     dz=0.01,
#     n_sources_per_layer=3,
#     intensity=q_heat,       # 设置热源强度
#     max_attempts=200
# )


# # 定义离散热源空间所需几何体
# geo_heatspace1 = MyCuboid([-1.6, -1.6, 0.09], [1.6, 1.6, 0.1])
# geo_heatspace2 = MyCuboid([-1.6, -1.6, 0.21], [1.6, 1.6, 0.22])
# geo_heatspace3 = MyCuboid([-1.6, -1.6, 0.33], [1.6, 1.6, 0.34])

# # 划分离散热源空间的网格点
# eval_pts1 = geo_heatspace1.uniform_points_xyz(nx=32, ny=32, nz=3, boundary=True)
# eval_pts2 = geo_heatspace2.uniform_points_xyz(nx=32, ny=32, nz=3, boundary=True)
# eval_pts3 = geo_heatspace3.uniform_points_xyz(nx=32, ny=32, nz=3, boundary=True)
# eval_pts = np.vstack((eval_pts1, eval_pts2, eval_pts3))


# dataOperator = CustomPDEOperator(data, space, eval_pts, num_function=1, num_test=1,device=device)

# # %%

# net = DeepONet_V2(
#     [9216, 512, 256, 256, 128, 128],
#     [3, 512, 256, 256, 128, 128],
#     "swish",
#     "Glorot normal",
# ).to(device)

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
#     net = DataParallelWrapper(net, device_ids=list(range(torch.cuda.device_count())))

# print("检查 regularizer 是否存在:", hasattr(net, "regularizer"))
# model = dde.Model(dataOperator, net)
# model.compile("adam", lr=0.0005)


# # # 修改模型保存回调
# # checkpoint_path = "model_best.ckpt"
# # checker = dde.callbacks.ModelCheckpoint(
# #     checkpoint_path,
# #     save_better_only=True,
# #     period=100,
# #     monitor="test loss",
# #     save_model_fn=lambda state: torch.save({
# #         'state_dict': net.module.state_dict(),  # 处理 DataParallel 的模块前缀
# #         'loss_history': losshistory,
# #         'train_state': train_state
# #     }, checkpoint_path)
# # )
# # checker = dde.callbacks.ModelCheckpoint(
# #     "model_best.ckpt", save_better_only=True, period=100,monitor="test loss"
# # )
# # losshistory, train_state = model.train(iterations=5000, callbacks=[checker],display_every=100)
# losshistory, train_state = model.train(iterations=20000, display_every=100)

# current_time = datetime.datetime.now()
# fname = current_time.strftime("%m%d_%H-%M-%S-") + f"{current_time.microsecond:06d}"[:3]

# # 确保Results目录存在
# os.makedirs("Results", exist_ok=True)

# # 修改路径，加入Results子目录
# fname_fig = os.path.join("Results", fname + ".png")
# fname_log = os.path.join("Results", fname + ".csv")
# save_path_model = os.path.join("Results", fname)  # 模型保存路径

# dde.utils.plot_loss_history(losshistory, fname=fname_fig)
# dde.utils.save_loss_history(losshistory, fname=fname_log)
# model.save(save_path=save_path_model)


# # %%
# x_jiban = geo_jiban.uniform_points_xyz(nx=200,ny=200,nz=20)
# x_chiplayer = geo_chiplayer.uniform_points_xyz(nx=64,ny=64,nz=38)
# x_heatspead = geo_heatspead.uniform_points_xyz(nx=200,ny=200,nz=40)
# x_heatsink = geo_heatsink.uniform_points_xyz(nx=300,ny=300,nz=80)
# x_pre_all = np.vstack((x_jiban, x_chiplayer, x_heatspead, x_heatsink))

# func_feats_pre = space.random(1)
# np.savetxt("func_feats_pre.csv", func_feats_pre, delimiter=",")
# v = space.eval_batch(func_feats_pre, eval_pts)

# # %%
# x_list = [
#     ('x_jiban', x_jiban),
#     ('x_chiplayer', x_chiplayer),
#     ('x_heatspead', x_heatspead),
#     ('x_heatsink', x_heatsink),
#     ('x_pre_all', x_pre_all)
# ]

# for name, x in x_list:
#     try:
#         D_factor = getDFactorV3(x, func_feats_pre)
#         y = model.predict((v, x, D_factor))
#         temperature_data = y[0][:, 0, 0] 
#         temperature_data = (temperature_data + 1) * T_ref
        
#         grid = pv.PolyData(x)
#         grid.point_data['Temperature'] = temperature_data
#         grid.save(f"{name}.vtk") 
#     except Exception as e:
#         print(f"Error processing {name}: {e}")



import deepxde as dde
import torch
import numpy as np
import datetime
import pyvista as pv
import os
from MyCuboid import MyCuboid
from DeepONetV2 import DeepONet_V2
from RandomCuboidHeatSourceMultiLayer import RandomCuboidHeatSourceMultiLayer
from CustomPDEOperator import CustomPDEOperator
from CustomBC import CustomNeumannBC, CustomRobinBC
from parallel import DataParallelWrapper

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

L_ref = 0.001
T_ref = 273.15
k_ref = 398
q_heat = 1e11 * L_ref**2 / (k_ref * T_ref)
DD = 131 / k_ref

def pde_heat(x, y, var):
    D = y[1][:,0]
    y_val = y[0][:, 0, :]
    # print(f"D:{D.shape}")
    # print(f"y_val:{y_val.shape}")
    Qheat = torch.zeros_like(D, device=x.device)
    Qheat[D == DD] = q_heat
    T_xx = dde.grad.hessian(y_val, x, i=0, j=0)
    T_yy = dde.grad.hessian(y_val, x, i=1, j=1)
    T_zz = dde.grad.hessian(y_val, x, i=2, j=2)
    lower_bound = 297.15
    upper_bound = 380
    y_pred = (y_val[:, 0] + 1) * 273.15
    penalty = torch.relu(lower_bound - y_pred) ** 2 + torch.relu(y_pred - upper_bound) ** 2
    return D * (T_xx + T_yy + T_zz) + Qheat + penalty * 0.1

geo_jiban = MyCuboid([-5, -5, -1.02], [5, 5, -0.02])
geo_chiplayer = MyCuboid([-1.6, -1.6, -0.02], [1.6, 1.6, 0.36])
geo_heatspead = MyCuboid([-5, -5, 0.36], [5, 5, 2.36])
geo_heatsink = MyCuboid([-7.5, -7.5, 2.36], [7.5, 7.5, 6.36])
geo = dde.geometry.CSGUnion(dde.geometry.CSGUnion(geo_jiban, geo_chiplayer), geo_heatspead)
geo = dde.geometry.CSGUnion(geo, geo_heatsink)

def boundary(x, on_boundary):
    # not判断 所以容差低等于精度高
    if not on_boundary:
        return False
    atol = 1e-5
    z_min = -1.02
    z_max = 6.36
    # 原始条件：排除z轴的两个极值面
    original_cond = not (np.isclose(x[2], z_min, atol=atol) or np.isclose(x[2], z_max, atol=atol))
    
    # 新增条件：当z=0.36或z=-0.02时，判断是否在长方形边框上
    rect_z_values = [0.36, -0.02]
    is_special_z = any(np.isclose(x[2], z_val, atol=atol) for z_val in rect_z_values)
    
    # 长方形边长3.2，中心在原点 → 坐标范围x,y ∈ [-1.6, 1.6]
    x_abs = abs(x[0])
    y_abs = abs(x[1])
    border_threshold = 1.6
    
    # 检查点是否在长方形边框上：
    # 1. 满足x或y的绝对值等于1.6，且另一坐标绝对值在1.6内
    is_on_border = (
        (np.isclose(x_abs, border_threshold, atol=atol) and (y_abs <= border_threshold )) or
        (np.isclose(y_abs, border_threshold, atol=atol) and (x_abs <= border_threshold ))
    )
    
    # 最终条件：满足原始条件且不在特殊Z平面的边框上
    return original_cond and not (is_special_z and is_on_border)


def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[2], 6.36,atol=1.e-10)

def boundary_bot(x, on_boundary):
    return on_boundary and np.isclose(x[2], -1.02,atol=1.e-10)

def robin_func_top(x, y):
    D_val = 398 / k_ref
    H = 5000 * L_ref / k_ref
    t_ab = (273.15 + 25) / T_ref - 1
    return -H / D_val * (y - t_ab)

def robin_func_bot(x, y):
    D_val = 130 / k_ref
    H = 10 * L_ref / k_ref
    t_ab = (273.15 + 25) / T_ref - 1
    return -H / D_val * (y - t_ab)

bc_top = CustomRobinBC(geom=geo, func=robin_func_top, on_boundary=boundary_top)
bc_bot = CustomRobinBC(geom=geo, func=robin_func_bot, on_boundary=boundary_bot)
bc_cebi = CustomNeumannBC(geo, lambda x: 0, boundary)

data = dde.data.PDE(
    geo, pde=pde_heat,
    bcs=[bc_cebi, bc_top, bc_bot],
    num_domain=100,
    num_boundary=1000,
    train_distribution="uniform",
    anchors=None
)

space = RandomCuboidHeatSourceMultiLayer(
    Lz=0.1, dx=1, dy=1, dz=0.01,
    n_sources_per_layer=3,
    intensity=q_heat,
    max_attempts=200
)

geo_heatspace1 = MyCuboid([-1.6, -1.6, 0.09], [1.6, 1.6, 0.1])
geo_heatspace2 = MyCuboid([-1.6, -1.6, 0.21], [1.6, 1.6, 0.22])
geo_heatspace3 = MyCuboid([-1.6, -1.6, 0.33], [1.6, 1.6, 0.34])
eval_pts1 = geo_heatspace1.uniform_points_xyz(nx=32, ny=32, nz=3, boundary=True)
eval_pts2 = geo_heatspace2.uniform_points_xyz(nx=32, ny=32, nz=3, boundary=True)
eval_pts3 = geo_heatspace3.uniform_points_xyz(nx=32, ny=32, nz=3, boundary=True)
eval_pts = np.vstack((eval_pts1, eval_pts2, eval_pts3))

dataOperator = CustomPDEOperator(data, space, eval_pts, num_function=1, num_test=1, device=device)

net = DeepONet_V2(
    [9216, 512, 256, 256, 128, 128],
    [3, 512, 256, 256, 128, 128],
    "swish", "Glorot normal"
).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    net = DataParallelWrapper(net, device_ids=list(range(torch.cuda.device_count())))

# print("检查 regularizer 是否存在:", hasattr(net, "regularizer"))
model = dde.Model(dataOperator, net)
model.compile("adam", lr=0.0005)

losshistory, train_state = model.train(iterations=20000, display_every=100)
current_time = datetime.datetime.now()
fname = current_time.strftime("%m%d_%H-%M-%S-") + f"{current_time.microsecond:06d}"[:3]

os.makedirs("Results", exist_ok=True)
fname_fig = os.path.join("Results", fname + ".png")
fname_log = os.path.join("Results", fname + ".csv")
save_path_model = os.path.join("Results", fname)

dde.utils.plot_loss_history(losshistory, fname=fname_fig)
dde.utils.save_loss_history(losshistory, fname=fname_log)
model.save(save_path=save_path_model)