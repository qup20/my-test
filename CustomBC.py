from deepxde.icbc import BC
import numpy as np

from deepxde import backend as bkd
from deepxde import gradients as grad
from deepxde import utils
from deepxde.backend import backend_name
from functools import wraps
import torch


class CustomRobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        # (x, D_factor)
        outputs = outputs[0][:, 0, :]
        # print(self.normal_derivative(X, inputs, outputs, beg, end))
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )



class CustomNeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        outputs = outputs[0][:, 0, :]
        values = self.func(X, beg, end, aux_var)
        # print(self.normal_derivative(X, inputs, outputs, beg, end))
        return abs(self.normal_derivative(X, inputs, outputs, beg, end) - values)
    

class CoupledRobinBC(BC):
    """Coupled Robin boundary conditions: T1 = T2 and k1*dT1/dn = k2*dT2/dn."""

    def __init__(self, geom, on_boundary, k1=1.0, k2=2.0, component_T1=0, component_T2=1):

        super().__init__(geom, on_boundary, component=component_T1) # 修改
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight1 = k1
        self.weight2 = k2
        self.component_T1 = component_T1
        self.component_T2 = component_T2
        self.D1 = 1
        self.D2 = 2
        self.on_boundary = lambda x, on: np.array(
            [geom.on_boundary(x[i]) for i in range(len(x))]
        )
        # self.on_boundary = lambda x, on: torch.tensor(
        #     [geom.on_boundary(x_i) for x_i in x],
        #     device=self.device,  # 强制指定设备
        #     dtype=torch.bool
        # )

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """
        Computes errors for the coupled Robin boundary conditions.
        """
        # print(beg,end)
        D1 = outputs[1][0][beg:end]
        # print(outputs,D1)
        D2 = outputs[1][1][beg:end]
        # print("一组")
        # print(D1,D2)
        # print(D1-D2)
        outputs = outputs[0]
        # print(outputs)
        outputs_1 = outputs[:, 0, :]
        outputs_2= outputs[:, 1, :]
        # beg, end = 0, outputs_1.shape[0]
        # T1 = outputs[beg:end, self.component_T1]
        # print(T1)
        # T2 = outputs[beg:end, self.component_T2]
        # print(T1-T2)

        # Compute normal erivative for T1 and T2
        #边界梯度基于取点所在的几何体，so边界点的法向梯度仅有1个！（代码层面）
        grad_T1 = self.normal_derivative(X, inputs, outputs_1, beg, end, component=0)
        grad_T2 = self.normal_derivative(X, inputs, outputs_2, beg, end, component=0)
        # print(grad_T1)
        # print(grad_T1,grad_T2)
        # Compute errors
        # error_T1_eq_T2 = T1 - T2 # Enforce T1 = T2
        error_gradient_relation = D1  * grad_T1 - D2 * grad_T2  # Enforce k1*dT1/dn = -k2*dT2/dn
        # print(error_T1_eq_T2 + abs(error_gradient_relation))
        # print (error_T1_eq_T2)
        # print(error_gradient_relation)
        return error_gradient_relation
    

    def normal_derivative(self, X, inputs, outputs, beg, end, component):
        dydx = grad.jacobian(outputs, inputs, i=component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end, None)
        return bkd.sum(dydx * n, 1, keepdims=True)
    
class CoupledRobinBCT(BC):
    """Coupled Robin boundary conditions: T1 = T2 and k1*dT1/dn = k2*dT2/dn."""

    def __init__(self, geom, on_boundary, k1=1.0, k2=2.0, component_T1=0, component_T2=1):

        super().__init__(geom, on_boundary, component=component_T1) # 修改
        self.weight1 = k1
        self.weight2 = k2
        self.component_T1 = component_T1
        self.component_T2 = component_T2
        self.D1 = 1
        self.D2 = 2
        self.on_boundary = lambda x, on: np.array(
            [geom.on_boundary(x[i]) for i in range(len(x))]
        )

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """
        Computes errors for the coupled Robin boundary conditions.
        """
        # print(beg,end)
        D1 = outputs[1][0][beg:end]
        # print(outputs,D1)
        D2 = outputs[1][1][beg:end]
        # print("一组")
        # print(D1,D2)
        # print(D1-D2)
        outputs = outputs[0]
        # print(outputs)
        # outputs_1 = outputs[:, 0, :]
        # outputs_2= outputs[:, 1, :]
        # beg, end = 0, outputs_1.shape[0]
        T1 = outputs[beg:end, self.component_T1]
        # print(T1)
        T2 = outputs[beg:end, self.component_T2]
        # print(T1-T2)

        # Compute normal erivative for T1 and T2
        #边界梯度基于取点所在的几何体，so边界点的法向梯度仅有1个！（代码层面）
        # grad_T1 = self.normal_derivative(X, inputs, outputs_1, beg, end, component=0)
        # grad_T2 = self.normal_derivative(X, inputs, outputs_2, beg, end, component=0)
        # print(grad_T1)
        # print(grad_T1,grad_T2)
        # Compute errors
        error_T1_eq_T2 = T1 - T2 # Enforce T1 = T2
        # error_gradient_relation = D1  * grad_T1 - D2 * grad_T2  # Enforce k1*dT1/dn = -k2*dT2/dn
        # print(error_T1_eq_T2 + abs(error_gradient_relation))
        # print (error_T1_eq_T2)
        # print(error_gradient_relation)
        return error_T1_eq_T2 

    def normal_derivative(self, X, inputs, outputs, beg, end, component):
        dydx = grad.jacobian(outputs, inputs, i=component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end, None)
        return bkd.sum(dydx * n, 1, keepdims=True)



class CustomDirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        outputs = outputs[0][:, 0, :]
        return outputs[beg:end, self.component : self.component + 1] - values


def npfunc_range_autocache(func):
    """Call a NumPy function on a range of the input ndarray.

    If the backend is pytorch, the results are cached based on the id of X.
    """
    # For some BCs, we need to call self.func(X[beg:end]) in BC.error(). For backend
    # tensorflow.compat.v1/tensorflow, self.func() is only called once in graph mode,
    # but for backend pytorch, it will be recomputed in each iteration. To reduce the
    # computation, one solution is that we cache the results by using @functools.cache
    # (https://docs.python.org/3/library/functools.html). However, numpy.ndarray is
    # unhashable, so we need to implement a hash function and a cache function for
    # numpy.ndarray. Here are some possible implementations of the hash function for
    # numpy.ndarray:
    # - xxhash.xxh64(ndarray).digest(): Fast
    # - hash(ndarray.tobytes()): Slow
    # - hash(pickle.dumps(ndarray)): Slower
    # - hashlib.md5(ndarray).digest(): Slowest
    # References:
    # - https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/16592241#16592241
    # - https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary/47922199
    # Then we can implement a cache function or use memoization
    # (https://github.com/lonelyenvoy/python-memoization), which supports custom cache
    # key. However, IC/BC is only for dde.data.PDE, where the ndarray is fixed. So we
    # can simply use id of X as the key, as what we do for gradients.

    cache = {}

    @wraps(func)
    def wrapper_nocache(X, beg, end, _):
        return func(X[beg:end])

    @wraps(func)
    def wrapper_nocache_auxiliary(X, beg, end, aux_var):
        return func(X[beg:end], aux_var[beg:end])

    @wraps(func)
    def wrapper_cache(X, beg, end, _):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]

    @wraps(func)
    def wrapper_cache_auxiliary(X, beg, end, aux_var):
        # Even if X is the same one, aux_var could be different
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end], aux_var[beg:end])
        return cache[key]

    if backend_name in ["tensorflow.compat.v1", "tensorflow", "jax"]:
        if utils.get_num_args(func) == 1:
            return wrapper_nocache
        if utils.get_num_args(func) == 2:
            return wrapper_nocache_auxiliary
    if backend_name in ["pytorch", "paddle"]:
        if utils.get_num_args(func) == 1:
            return wrapper_cache
        if utils.get_num_args(func) == 2:
            return wrapper_nocache_auxiliary
