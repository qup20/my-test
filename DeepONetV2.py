# import torch

# from deepxde.nn import NN
# from deepxde.nn.pytorch.fnn import FNN
# from deepxde.nn import activations
# from deepxde.nn.deeponet_strategy import (
#     SingleOutputStrategy,
#     IndependentStrategy,
#     SplitBothStrategy,
#     SplitBranchStrategy,
#     SplitTrunkStrategy,
# )
# import deepxde as dde
# from nets import MultiFNN
# import pyvista as pv
# import os
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def pde_heat(x,outputs,D_factor):
#     """
#     x: 输入坐标 (N, 3) -> (x, y, z)
#     y: 网络输出 (N, 1) -> T(x, y, z)
#     f: 额外的源项或可选参数, 例如热源 f(x,y,z)
    
#     返回值: PDE 残差, 在稳态情况下要求残差=0
#     """
#     # 假设导热系数 D 是常数，也可以根据需要改成函数形式
#     # D = 131
#     # dx，dy，dz是1，1，0.1 还是 3.2，3.2，0.1
#     # 参考量
#     L_ref = 0.001  # mm
#     T_ref = 273.15  # K
#     k_ref = 398  #
#     q_heat = 1e11 * L_ref**2 / (k_ref * T_ref)
#     D = D_factor[0].unsqueeze(1)
#     y = outputs
#     Qheat = torch.zeros_like(D,device=device)
#     Qheat[D ==131/k_ref ] = q_heat

#     # 计算二阶偏导数 (拉普拉斯算子)
#     # T_xx, T_yy, T_zz 分别是对 x, y, z 的二阶偏导
#     T_xx = dde.grad.hessian(y, x, i=0, j=0)
#     T_yy = dde.grad.hessian(y, x, i=1, j=1)
#     T_zz = dde.grad.hessian(y, x, i=2, j=2)
#     # print("T_xx", T_xx)

#     # 稳态热传导方程: -D * (T_xx + T_yy + T_zz) = v
#     # 若 v=0, 表示无热源
    
#     return abs(D * (T_xx + T_yy + T_zz) + Qheat)

# class DeepONet_V2(NN):
#     """Deep operator network.

#     `Lu et al. Learning nonlinear operators via DeepONet based on the universal
#     approximation theorem of operators. Nat Mach Intell, 2021.
#     <https://doi.org/10.1038/s42256-021-00302-5>`_

#     Args:
#         layer_sizes_branch: A list of integers as the width of a fully connected network,
#             or `(dim, f)` where `dim` is the input dimension and `f` is a network
#             function. The width of the last layer in the branch and trunk net
#             should be the same for all strategies except "split_branch" and "split_trunk".
#         layer_sizes_trunk (list): A list of integers as the width of a fully connected
#             network.
#         activation: If `activation` is a ``string``, then the same activation is used in
#             both trunk and branch nets. If `activation` is a ``dict``, then the trunk
#             net uses the activation `activation["trunk"]`, and the branch net uses
#             `activation["branch"]`.
#         num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
#             `multi_output_strategy` below should be set.
#         multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
#             "split_trunk". It makes sense to set in case of multiple outputs.

#             - None
#             Classical implementation of DeepONet with a single output.
#             Cannot be used with `num_outputs` > 1.

#             - independent
#             Use `num_outputs` independent DeepONets, and each DeepONet outputs only
#             one function.

#             - split_both
#             Split the outputs of both the branch net and the trunk net into `num_outputs`
#             groups, and then the kth group outputs the kth solution.

#             - split_branch
#             Split the branch net and share the trunk net. The width of the last layer
#             in the branch net should be equal to the one in the trunk net multiplied
#             by the number of outputs.

#             - split_trunk
#             Split the trunk net and share the branch net. The width of the last layer
#             in the trunk net should be equal to the one in the branch net multiplied
#             by the number of outputs.
#     """

#     def __init__(
#         self,
#         layer_sizes_branch,
#         layer_sizes_trunk,
#         activation,
#         kernel_initializer,
#         num_outputs=1,
#         multi_output_strategy=None,
#     ):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if isinstance(activation, dict):
#             self.activation_branch = activation["branch"]
#             self.activation_trunk = activations.get(activation["trunk"])
#         else:
#             self.activation_branch = self.activation_trunk = activations.get(activation)
#         self.kernel_initializer = kernel_initializer

#         self.num_outputs = num_outputs
#         if self.num_outputs == 1:
#             if multi_output_strategy is not None:
#                 raise ValueError(
#                     "num_outputs is set to 1, but multi_output_strategy is not None."
#                 )
#         elif multi_output_strategy is None:
#             multi_output_strategy = "independent"
#             print(
#                 f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
#                 'Use "independent" as the multi_output_strategy.'
#             )
#         self.multi_output_strategy = {
#             None: SingleOutputStrategy,
#             "independent": IndependentStrategy,
#             "split_both": SplitBothStrategy,
#             "split_branch": SplitBranchStrategy,
#             "split_trunk": SplitTrunkStrategy,
#         }[multi_output_strategy](self)

#         self.branch, self.trunk = self.multi_output_strategy.build(
#             layer_sizes_branch, layer_sizes_trunk
#         )
#         if isinstance(self.branch, list):
#             self.branch = torch.nn.ModuleList(self.branch)
#         if isinstance(self.trunk, list):
#             self.trunk = torch.nn.ModuleList(self.trunk)
#         self.b = torch.nn.ParameterList(
#             [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
#         )
#         self.iters = 0
#     def build_branch_net(self, layer_sizes_branch):
#         # User-defined network
#         if callable(layer_sizes_branch[1]):
#             return layer_sizes_branch[1]
#         # Fully connected network
#         return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

#     def build_trunk_net(self, layer_sizes_trunk):
#         return MultiFNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer) # 需要

#     def merge_branch_trunk(self, x_func, x_loc, index):
        
#         # y = torch.einsum("bi,bi->b", x_func, x_loc)
#         # y = torch.unsqueeze(y, dim=1)
#         # y += self.b[index]
#         # return y
#         # x_func (bs, 1)
#         # x_loc (bs, 2, 1)           
#         x_func = x_func.unsqueeze(1).repeat(1, 2, 1)  # x_func (bs, 2, 1)
#         # Perform element-wise multiplication and sum over the feature dimension
#         y = torch.einsum("bif,bif->bi", x_func, x_loc)  # Shape: (bs, 2)

#         # Add a new dimension to match the required output shape
#         y = y.unsqueeze(-1)  # Shape: (bs, 2, 1)

#         # Add the bias term
#         y += self.b[index].unsqueeze(0).unsqueeze(-1)  # Shape: (bs, 2, 1)
#         # print(type(y))
#         return y
    

#     @staticmethod
#     def concatenate_outputs(ys):
#         return torch.concat(ys, dim=1)

#     def forward(self, inputs):
#         # print("inputs:")
#         # print(inputs)
#         # assert inputs[0].device == inputs[1].device
#         # assert inputs[1].device == inputs[2].device
#         # device = inputs[1].device
#         # print(inputs.device)
#         # assert all(t.device == torch.device('cpu') for t in inputs if isinstance(t, torch.Tensor)), \
#         #     "输入张量必须位于CPU以便多GPU分发"
#         x_func = inputs[0]
#         x_loc = inputs[1]
#         D_factor = inputs[2].T
#         # print(D_factor)
#         # Trunk net input transform
#         if self._input_transform is not None:
#             x_loc = self._input_transform(x_loc)
#         x_loc = (x_loc, D_factor)
#         # print(next(self.branch.parameters()).device)
#         # print (f"x-fuc:{x_func[0].device}")

#         # print(f"x-loc:{x_loc[0].device}")
#         # print(f"self-b:{self.b[0].device}")
#         # print(f"D:{D_factor[0].device}")
#         # print(x_loc.device)
#         x = self.multi_output_strategy.call(x_func, x_loc)
#         # 返回的x中不带D Factor
#         # x的形状是（bs，2，feat）
#         if self._output_transform is not None:
#             x = self._output_transform(inputs, x)
#         self.iters += 1 
#         if self.iters % 50 == 0: 
#             outputs = (x[:, 0, :]+1)*273.15
#             indices = torch.randperm(outputs.shape[0]) 
#             sampled_indices = indices[:100] # 采样个数

#             print(f"------outputs: min:{torch.min(outputs):.4f}, max:{torch.max(outputs):.4f}, median:{torch.median(outputs):.4f}------")
#             # print(f"------随机采样：{outputs[sampled_indices, 0]}------")
#             pdeloss = pde_heat(inputs[1],outputs,D_factor)
#             grid = pv.PolyData(inputs[1].detach().cpu().numpy())
#             grid.point_data['PDE-loss'] = pdeloss.detach().cpu().numpy()
#             grid.point_data['Temperature'] = outputs.detach().cpu().numpy()

#             # 创建目录 & 指定保存路径
#             os.makedirs("Results", exist_ok=True)  # 防御性创建目录
#             vtk_path = os.path.join("Results", "pdeloss-T.vtk")
#             grid.save(vtk_path)  # 带路径的保存
            
            
#             # x (bs 2 1)   D (2 bs)
#         return (x, D_factor)


import torch
from deepxde.nn import NN, activations
from modified_fnn import FNN  # 使用修改后的 FNN 实现
from deepxde.nn.deeponet_strategy import (
    SingleOutputStrategy,
    IndependentStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy,
)
import deepxde as dde
from nets import MultiFNN
import pyvista as pv
import os

def pde_heat(x, outputs, D_factor):
    L_ref = 0.001  # mm
    T_ref = 273.15  # K
    k_ref = 398
    q_heat = 1e11 * L_ref**2 / (k_ref * T_ref)
    device_local = x.device
    D = D_factor[0].unsqueeze(1)
    y = outputs
    Qheat = torch.zeros_like(D, device=device_local)
    Qheat[D == 131 / k_ref] = q_heat

    T_xx = dde.grad.hessian(y, x, i=0, j=0)
    T_yy = dde.grad.hessian(y, x, i=1, j=1)
    T_zz = dde.grad.hessian(y, x, i=2, j=2)

    return torch.abs(D * (T_xx + T_yy + T_zz) + Qheat)

class DeepONet_V2(NN):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=1,
        multi_output_strategy=None,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer

        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError(
                    "num_outputs is set to 1, but multi_output_strategy is not None."
                )
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                'Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0, device=self.device)) for _ in range(self.num_outputs)]
        )
        self.iters = 0

    def build_branch_net(self, layer_sizes_branch):
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return MultiFNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        x_func = x_func.unsqueeze(1).repeat(1, 2, 1)
        y = torch.einsum("bif,bif->bi", x_func, x_loc)
        y = y.unsqueeze(-1)
        y += self.b[index].unsqueeze(0).unsqueeze(-1)
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.concat(ys, dim=1)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1].to(x_func.device)
        D_factor = inputs[2].T.to(x_func.device)

        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc_tuple = (x_loc, D_factor)

        x = self.multi_output_strategy.call(x_func, x_loc_tuple)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        self.iters += 1
        if self.iters % 50 == 0:
            outputs = (x[:, 0, :] + 1) * 273.15
            print(f"------outputs: min:{torch.min(outputs):.4f}, max:{torch.max(outputs):.4f}, median:{torch.median(outputs):.4f}------")
            pdeloss = pde_heat(x_loc, outputs, D_factor)
            grid = pv.PolyData(x_loc.detach().cpu().numpy())
            grid.point_data['PDE-loss'] = pdeloss.detach().cpu().numpy()
            grid.point_data['Temperature'] = outputs.detach().cpu().numpy()
            os.makedirs("Results", exist_ok=True)
            vtk_path = os.path.join("Results", "pdeloss-T.vtk")
            grid.save(vtk_path)

        return (x, D_factor)