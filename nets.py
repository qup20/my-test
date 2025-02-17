from deepxde.nn import activations, initializers
from deepxde import config
from deepxde.nn import NN
import torch

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
        # 边界条件trunk
        X = inputs[0]
        D_factors = inputs[1].to(X.device)
        # print(X.shape)
        # print(D_factor.shape)
        D_len = D_factors.shape[0]
        batch_size = X.size(0)
        output_dim = self.branches['0'][-1].out_features  # 假设最后一层输出决定输出维度
        outputs_all = torch.zeros(batch_size, D_len, output_dim, device=X.device)
        for i in range(D_factors.shape[0]):
            D_factor = D_factors[i]
            # 初始化结果张量，形状为 (batch_size, output_dim)
            outputs = torch.zeros(batch_size, output_dim, device=X.device, dtype=X.dtype)

            if self._input_transform is not None:
                X = self._input_transform(X)


            for branch_value, branch_idx in self.branch_map.items():
                mask = (D_factor == branch_value)  # Select samples matching the current branch value
                if torch.any(mask):  # If any samples match the current branch
                    selected_x = X[mask]  # Extract inputs for the current branch
                    linears = self.branches[str(branch_idx)]
                    
                    # Forward pass through the selected branch
                    for j, linear in enumerate(linears[:-1]):
                        selected_x = (
                            self.activation[j](linear(selected_x))
                            if isinstance(self.activation, list)
                            else self.activation(linear(selected_x))
                        )
                    selected_x = linears[-1](selected_x)
                    outputs[mask] = selected_x  # Map outputs back to their original indices
            outputs_all[:, i, :] = outputs
        return outputs_all
