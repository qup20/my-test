import torch
import torch.nn as nn
from deepxde.nn import activations, initializers
from deepxde import config
class FNN(nn.Module):
    """Fully-connected neural network with explicit device management in forward()."""

    def __init__(
        self, layer_sizes, activation, kernel_initializer, regularization=None
    ):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "The number of activation functions does not match the number of layers."
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization

        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                nn.Linear(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                    dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if hasattr(self, "_input_transform") and self._input_transform is not None:
            x = self._input_transform(x)
        # 强制将输入迁移到第一个线性层所在的设备
        device_current = next(self.linears[0].parameters()).device
        x = x.to(device_current)
        for j, linear in enumerate(self.linears[:-1]):
            if isinstance(self.activation, list):
                x = self.activation[j](linear(x))
            else:
                x = self.activation(linear(x))
        x = self.linears[-1](x)
        if hasattr(self, "_output_transform") and self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x