
# # parallel.py
# import torch.nn as nn
# import torch
# from deepxde.nn.pytorch.fnn import FNN 
# import numpy as np


# # class DataParallelWrapper(nn.DataParallel):
# #     def __init__(self, module, device_ids=None, output_device=None):
# #         super().__init__(module, device_ids, output_device)
# #         self.main_device = self.device_ids[0] if device_ids else torch.device("cpu")

# #     def __getattr__(self, name):
# #         try:
# #             return super().__getattr__(name)
# #         except AttributeError:
# #             return getattr(self.module, name)

# #     def move_to_device(self, obj):
# #         """递归转移对象中的Tensor至主设备"""
# #         if isinstance(obj, (list, tuple)):
# #             return type(obj)(self.move_to_device(e) for e in obj)
# #         elif isinstance(obj, dict):
# #             return {k: self.move_to_device(v) for k, v in obj.items()}
# #         elif isinstance(obj, torch.Tensor):
# #             return obj.to(self.main_device)
# #         else:
# #             return obj  # 非Tensor类型（如np数组）保持不动

# #     def forward(self, *inputs):
# #         # 递归处理输入数据中的所有Tensor
# #         inputs = self.move_to_device(inputs)
# #         return super().forward(*inputs)

# class DataParallelWrapper(nn.DataParallel):
#     def __init__(self, module, device_ids=None, output_device=None):
#         super().__init__(module, device_ids, output_device)
#         self.main_device = f"cuda:{device_ids[0]}" if device_ids else "cpu"
        
#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.module, name)

#     def forward(self, *inputs, **kwargs):
#         inputs = self._recursive_to_device(inputs)
#         kwargs = self._recursive_to_device(kwargs)
#         return super().forward(*inputs, **kwargs)
#     # def forward(self, *inputs, **kwargs):
#     #     # 将输入全部移至CPU，触发DataParallel的自动分发逻辑
#     #     inputs = [tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor 
#     #               for tensor in inputs]
#     #     return super().forward(*inputs, **kwargs)

#     # def _recursive_to_device(self, data):
#     #     if isinstance(data, (list, tuple)):
#     #         return type(data)(self._recursive_to_device(e) for e in data)
#     #     elif isinstance(data, dict):
#     #         return {k: self._recursive_to_device(v) for k, v in data.items()}
#     #     elif isinstance(data, torch.Tensor):
#     #         return data.to(self.main_device)
#     #     else:
#     #         return data  # numpy数组等非Tensor数据保留原样
#     def _recursive_to_device(self, data):
#         if isinstance(data, (list, tuple)):
#             return type(data)(self._recursive_to_device(e) for e in data)
#         elif isinstance(data, dict):
#             return {k: self._recursive_to_device(v) for k, v in data.items()}
#         elif isinstance(data, torch.Tensor):
#             return data.to(self.main_device)
#         elif isinstance(data, np.ndarray):  # 处理numpy数组的自动转换
#             return torch.as_tensor(data).to(self.main_device)
#         else:
#             return data


import torch.nn as nn
import torch
from deepxde.nn.pytorch.fnn import FNN 
import numpy as np

class DataParallelWrapper(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None):
        super().__init__(module, device_ids, output_device)
        # 主设备设置为第一个指定的 GPU（例如 "cuda:0"）
        self.main_device = f"cuda:{device_ids[0]}" if device_ids else "cpu"
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *inputs, **kwargs):
        # 当输入为 numpy array 时转换为 tensor，并不主动改变 tensor 的设备，让 DataParallel 内部进行 scatter
        def convert_data(data):
            if isinstance(data, (list, tuple)):
                return type(data)(convert_data(e) for e in data)
            elif isinstance(data, dict):
                return {k: convert_data(v) for k, v in data.items()}
            elif isinstance(data, np.ndarray):
                return torch.as_tensor(data)
            else:
                return data

        inputs = convert_data(inputs)
        kwargs = convert_data(kwargs)
        return super().forward(*inputs, **kwargs)