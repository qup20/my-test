# from deepxde.nn import activations, initializers
# from deepxde import config
# from deepxde.nn import NN
# import torch

# L_ref = 0.001  # mm
# T_ref = 273.15  # K
# k_ref = 398  #

# class MultiFNN(NN):
#     """Multi Branch Fully-connected neural network."""

#     def __init__(
#         self, 
#         layer_sizes, 
#         activation, 
#         kernel_initializer, 
#         regularization=None, 
#         branch_map={
#             131/k_ref:'0',
#             130/k_ref:'1',
#             1.5/k_ref:'2',
#             398/k_ref:'3'
#         }
#     ):
#         """
#         :param layer_sizes: List of layer sizes for the network.
#         :param activation: Activation function or list of functions.
#         :param kernel_initializer: Weight initialization method.
#         :param regularization: Regularization method (optional).
#         :param branch_map: Dictionary mapping D values to branch indices.
#         """
#         super().__init__()
#         if branch_map is None:
#             raise ValueError("branch_map cannot be None. Provide a mapping of D values to branches.")
        
#         if isinstance(activation, list):
#             if not (len(layer_sizes) - 1) == len(activation):
#                 raise ValueError(
#                     "Total number of activation functions do not match "
#                     "with sum of hidden layers and output layer!"
#                 )
#             self.activation = list(map(activations.get, activation))
#         else:
#             self.activation = activations.get(activation)
        
#         initializer = initializers.get(kernel_initializer)
#         initializer_zero = initializers.get("zeros")
#         self.regularizer = regularization
#         self.branch_map = branch_map

#         # Create branches based on branch_map
#         self.branches = torch.nn.ModuleDict()
#         for _, branch_idx in branch_map.items():
#             branch_linears = torch.nn.ModuleList()
#             for i in range(1, len(layer_sizes)):
#                 linear = torch.nn.Linear(
#                     layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
#                 )
#                 initializer(linear.weight)
#                 initializer_zero(linear.bias)
#                 branch_linears.append(linear)
#             self.branches[str(branch_idx)] = branch_linears  # Use str(branch_idx) for compatibility with ModuleDict



#     def forward(self, inputs):
#         # 边界条件trunk
#         X = inputs[0]
#         D_factors = inputs[1].to(X.device)
#         # print(X.shape)
#         # print(D_factor.shape)
#         D_len = D_factors.shape[0]
#         batch_size = X.size(0)
#         output_dim = self.branches['0'][-1].out_features  # 假设最后一层输出决定输出维度
#         outputs_all = torch.zeros(batch_size, D_len, output_dim, device=X.device)
#         for i in range(D_factors.shape[0]):
#             D_factor = D_factors[i]
#             # 初始化结果张量，形状为 (batch_size, output_dim)
#             outputs = torch.zeros(batch_size, output_dim, device=X.device, dtype=X.dtype)

#             if self._input_transform is not None:
#                 X = self._input_transform(X)


#             for branch_value, branch_idx in self.branch_map.items():
#                 mask = (D_factor == branch_value)  # Select samples matching the current branch value
#                 if torch.any(mask):  # If any samples match the current branch
#                     selected_x = X[mask]  # Extract inputs for the current branch
#                     linears = self.branches[str(branch_idx)]
                    
#                     # Forward pass through the selected branch
#                     for j, linear in enumerate(linears[:-1]):
#                         selected_x = (
#                             self.activation[j](linear(selected_x))
#                             if isinstance(self.activation, list)
#                             else self.activation(linear(selected_x))
#                         )
#                     selected_x = linears[-1](selected_x)
#                     outputs[mask] = selected_x  # Map outputs back to their original indices
#             outputs_all[:, i, :] = outputs
#         return outputs_all

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepxde.nn import activations, initializers
from deepxde import config
from deepxde.nn import NN

L_ref = 0.001  # mm
T_ref = 273.15  # K
k_ref = 398  #

class MultiFNN(NN):
    """Multi Branch Fully-connected neural network."""

    def __init__(
        self, 
        layer_sizes, 
        activation, 
        kernel_initializer, 
        regularization=None, 
        branch_map={
            131/k_ref:'0',
            130/k_ref:'1',
            1.5/k_ref:'2',
            398/k_ref:'3'
        }
    ):
        """
        :param layer_sizes: List of layer sizes for the network.
        :param activation: Activation function or list of functions.
        :param kernel_initializer: Weight initialization method.
        :param regularization: Regularization method (optional).
        :param branch_map: Dictionary mapping D values to branch indices.
        """
        super().__init__()
        if branch_map is None:
            raise ValueError("branch_map cannot be None. Provide a mapping of D values to branches.")
        
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match "
                    "with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization
        self.branch_map = branch_map

        # Create branches based on branch_map
        self.branches = torch.nn.ModuleDict()
        for _, branch_idx in branch_map.items():
            branch_linears = torch.nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                linear = torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
                initializer(linear.weight)
                initializer_zero(linear.bias)
                branch_linears.append(linear)
            self.branches[str(branch_idx)] = branch_linears  # Use str(branch_idx) for compatibility with ModuleDict

    def forward(self, inputs):
        # X: main input tensor; D_factors: tensor with each row corresponding to a D_factor.
        X = inputs[0]
        D_factors = inputs[1]
        
        # Determine target device based on the first branch's parameters.
        # If branches exist, move X and D_factors to that device.
        if len(self.branches) > 0:
            sample_branch = next(iter(self.branches.values()))
            target_device = next(sample_branch.parameters()).device
            X = X.to(target_device)
            D_factors = D_factors.to(target_device)
        else:
            target_device = X.device

        D_len = D_factors.shape[0]
        batch_size = X.size(0)
        output_dim = self.branches['0'][-1].out_features  # Assuming branch '0' exists and determines output dim.
        outputs_all = torch.zeros(batch_size, D_len, output_dim, device=target_device, dtype=X.dtype)
        
        # Loop over each D_factor (row in D_factors).
        for i in range(D_factors.shape[0]):
            D_factor = D_factors[i]
            # Initialize output tensor for current D_factor.
            outputs = torch.zeros(batch_size, output_dim, device=target_device, dtype=X.dtype)

            if self._input_transform is not None:
                X = self._input_transform(X)
            
            # Iterate through each branch mapping.
            for branch_value, branch_idx in self.branch_map.items():
                mask = (D_factor == branch_value)  # Select samples matching the current branch value.
                if torch.any(mask):  # If any samples match the current branch.
                    selected_x = X[mask]  # Extract inputs for the current branch.
                    # Ensure selected_x is moved to the target device.
                    selected_x = selected_x.to(target_device)
                    linears = self.branches[str(branch_idx)]
                    
                    # Forward pass through the selected branch.
                    for j, linear in enumerate(linears[:-1]):
                        if isinstance(self.activation, list):
                            selected_x = self.activation[j](linear(selected_x))
                        else:
                            selected_x = self.activation(linear(selected_x))
                    selected_x = linears[-1](selected_x)
                    outputs[mask] = selected_x  # Map outputs back to their original indices.
            outputs_all[:, i, :] = outputs
        return outputs_all
    
    # def forward(self, inputs):
    #     """
    #     inputs[0]: 主输入 X，形状为 (batch_size, input_dim)。
    #     inputs[1]: 辅助输入 D_factors，要求形状为 (batch_size, D_len)。
    #     """
    #     X = inputs[0]
    #     D_factors = inputs[1]
        
    #     # 将 X 与 D_factors 移动至目标设备
    #     if len(self.branches) > 0:
    #         sample_branch = next(iter(self.branches.values()))
    #         target_device = next(sample_branch.parameters()).device
    #         X = X.to(target_device)
    #         D_factors = D_factors.to(target_device)
    #     else:
    #         target_device = X.device

    #     # 保证 D_factors 的第一维为 batch_size。如果 D_factors 已经形成为 (batch_size, D_len) 则保持不变。
    #     # 这里不再对 D_factors 做额外转置，确保 mask 与 X 的第一维一致。
    #     if D_factors.shape[0] != X.size(0):
    #         raise IndexError("D_factors 的第一维必须等于 X 的 batch size.")

    #     batch_size = X.size(0)
    #     D_len = D_factors.shape[1]
    #     output_dim = self.branches['0'][-1].out_features  # 假定 branch '0' 存在
    #     outputs_all = torch.zeros(batch_size, D_len, output_dim, device=target_device, dtype=X.dtype)
        
    #     # 对于每个 D_factor值（对应每个分支），分别计算输出
    #     for i in range(D_len):
    #         # 提取第 i 列，形状为 (batch_size,)
    #         D_column = D_factors[:, i]
    #         outputs = torch.zeros(batch_size, output_dim, device=target_device, dtype=X.dtype)
            
    #         # 如果存在特殊输入变换则先变换 X
    #         if self._input_transform is not None:
    #             X_transformed = self._input_transform(X)
    #         else:
    #             X_transformed = X
            
    #         # 针对每个分支，通过 branch_map 判定 mask
    #         for branch_value, branch_idx in self.branch_map.items():
    #             mask = (D_column == branch_value)  # mask 形状 (batch_size,)
    #             if torch.any(mask):
    #                 selected_x = X_transformed[mask]  # 选取满足条件的样本，注意 mask 长度与 X_transformed 第一维一致
    #                 selected_x = selected_x.to(target_device)
    #                 linears = self.branches[str(branch_idx)]
                    
    #                 # 依次通过线性层、激活函数
    #                 for j, linear in enumerate(linears[:-1]):
    #                     if isinstance(self.activation, list):
    #                         selected_x = self.activation[j](linear(selected_x))
    #                     else:
    #                         selected_x = self.activation(linear(selected_x))
    #                 selected_x = linears[-1](selected_x)
    #                 outputs[mask] = selected_x
    #         outputs_all[:, i, :] = outputs
    #     return outputs_all