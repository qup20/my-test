import itertools
from typing import Union, Literal

import numpy as np

from deepxde.geometry.geometry_2d import Rectangle
from deepxde.geometry.geometry_nd import Hypercube
from deepxde.backend import backend as bkd
from deepxde import config

#添加轴向均匀取点划分网格函数uniform_points_xyz
  
class MyCuboid(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))

    def random_boundary_points(self, n, random="pseudo"):
        pts = []
        density = n / self.area
        rect = Rectangle(self.xmin[:-1], self.xmax[:-1])
        for z in [self.xmin[-1], self.xmax[-1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u, np.full((len(u), 1), z))))
        rect = Rectangle(self.xmin[::2], self.xmax[::2])
        for y in [self.xmin[1], self.xmax[1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), y), u[:, 1:])))
        rect = Rectangle(self.xmin[1:], self.xmax[1:])
        for x in [self.xmin[0], self.xmax[0]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((np.full((len(u), 1), x), u)))
        pts = np.vstack(pts)
        if len(pts) > n:
            return pts[np.random.choice(len(pts), size=n, replace=False)]
        return pts

    def uniform_boundary_points(self, n):
        h = (self.area / n) ** 0.5
        nx, ny, nz = np.ceil((self.xmax - self.xmin) / h).astype(int) + 1
        x = np.linspace(self.xmin[0], self.xmax[0], num=nx)
        y = np.linspace(self.xmin[1], self.xmax[1], num=ny)
        z = np.linspace(self.xmin[2], self.xmax[2], num=nz)

        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = list(itertools.product(x, y))
            pts.append(np.hstack((u, np.full((len(u), 1), v))))
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = np.array(list(itertools.product(x, z[1:-1])))
                pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), v), u[:, 1:])))
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = list(itertools.product(y[1:-1], z[1:-1]))
                pts.append(np.hstack((np.full((len(u), 1), v), u)))
        pts = np.vstack(pts)
        if n != len(pts):
            print(
                "Warning: {} points required, but {} points sampled.".format(
                    n, len(pts)
                )
            )
        return pts

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+",
        where: Union[
            None, Literal["back", "front", "left", "right", "bottom", "top"]
        ] = None,
        inside: bool = True,
    ):
        """Compute the hard constraint factor at x for the boundary.

        This function is used for the hard-constraint methods in Physics-Informed Neural Networks (PINNs).
        The hard constraint factor satisfies the following properties:

        - The function is zero on the boundary and positive elsewhere.
        - The function is at least continuous.

        In the ansatz `boundary_constraint_factor(x) * NN(x) + boundary_condition(x)`, when `x` is on the boundary,
        `boundary_constraint_factor(x)` will be zero, making the ansatz be the boundary condition, which in
        turn makes the boundary condition a "hard constraint".

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry. Note that `x` should be a tensor type
                of backend (e.g., `tf.Tensor` or `torch.Tensor`), not a numpy array.
            smoothness (string, optional): A string to specify the smoothness of the distance function,
                e.g., "C0", "C0+", "Cinf". "C0" is the least smooth, "Cinf" is the most smooth.
                Default is "C0+".

                - C0
                The distance function is continuous but may not be non-differentiable.
                But the set of non-differentiable points should have measure zero,
                which makes the probability of the collocation point falling in this set be zero.

                - C0+
                The distance function is continuous and differentiable almost everywhere. The
                non-differentiable points can only appear on boundaries. If the points in `x` are
                all inside or outside the geometry, the distance function is smooth.

                - Cinf
                The distance function is continuous and differentiable at any order on any
                points. This option may result in a polynomial of HIGH order.

            where (string, optional): A string to specify which part of the boundary to compute the distance.
                "back": x[0] = xmin[0], "front": x[0] = xmax[0], "left": x[1] = xmin[1], 
                "right": x[1] = xmax[1], "bottom": x[2] = xmin[2], "top": x[2] = xmax[2]. 
                If `None`, compute the distance to the whole boundary. Default is `None`.
            inside (bool, optional): The `x` is either inside or outside the geometry.
                The cases where there are both points inside and points
                outside the geometry are NOT allowed. NOTE: currently only support `inside=True`.

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """
        if where not in [None, "back", "front", "left", "right", "bottom", "top"]:
            raise ValueError(
                "where must be one of None, back, front, left, right, bottom, top"
            )
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")
        if self.dim != 3:
            raise ValueError("self.dim must be 3")
        if not inside:
            raise ValueError("inside=False is not supported for Cuboid")

        if not hasattr(self, "self.xmin_tensor"):
            self.xmin_tensor = bkd.as_tensor(self.xmin)
            self.xmax_tensor = bkd.as_tensor(self.xmax)

        dist_l = dist_r = None
        if where not in ["front", "right", "top"]:
            dist_l = bkd.abs(
                (x - self.xmin_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
            )
        if where not in ["back", "left", "bottom"]:
            dist_r = bkd.abs(
                (x - self.xmax_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
            )

        if where == "back":
            return dist_l[:, 0:1]
        if where == "front":
            return dist_r[:, 0:1]
        if where == "left":
            return dist_l[:, 1:2]
        if where == "right":
            return dist_r[:, 1:2]
        if where == "bottom":
            return dist_l[:, 2:]
        if where == "top":
            return dist_r[:, 2:]

        if smoothness == "C0":
            dist_l = bkd.min(dist_l, dim=-1, keepdims=True)
            dist_r = bkd.min(dist_r, dim=-1, keepdims=True)
            return bkd.minimum(dist_l, dist_r)
        dist_l = bkd.prod(dist_l, dim=-1, keepdims=True)
        dist_r = bkd.prod(dist_r, dim=-1, keepdims=True)
        return dist_l * dist_r
    
    def uniform_points_xyz(self, n=None, boundary=True, nx=None, ny=None, nz=None):
        """
        在三维情况下, 若 nx, ny, nz 全部给定, 则在 x, y, z 上分别等分.
        否则走原先 (volume / n)^(1/dim) 方式.
        """
        if self.dim != 3:
            # 对于非三维, 保持原逻辑 or raise error
            if nx or ny or nz:
                raise ValueError("nx, ny, nz only supported for dim=3!")
        
        if (nx is not None and ny is not None and nz is not None and self.dim == 3):
            # == A. 三个方向分别取 nx, ny, nz 等分
            x_vals = self._generate_1d_points(self.xmin[0], self.xmax[0], nx, boundary)
            y_vals = self._generate_1d_points(self.xmin[1], self.xmax[1], ny, boundary)
            z_vals = self._generate_1d_points(self.xmin[2], self.xmax[2], nz, boundary)
            
            x = np.array(list(itertools.product(x_vals, y_vals, z_vals)))
            return x
        else:
            # == B. 原先逻辑
            dx = (self.volume / n) ** (1 / self.dim)
            xi = []
            for i in range(self.dim):
                ni = int(np.ceil(self.side_length[i] / dx))
                if boundary:
                    coords_1d = np.linspace(
                        self.xmin[i], self.xmax[i], num=ni, dtype=config.real(np)
                    )
                else:
                    coords_1d = np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype=config.real(np),
                    )[1:]
                xi.append(coords_1d)
            
            x = np.array(list(itertools.product(*xi)))
            if n != len(x):
                print(
                    f"Warning: {n} points required, but {len(x)} points sampled."
                )
            return x

    def _generate_1d_points(self, low, high, num, boundary):
        """
        辅助函数: 在 [low, high] 中等分 num 个点, 是否含边界由 boundary 控制.
        """
        if boundary:
            return np.linspace(low, high, num, dtype=config.real(np))
        else:
            # 不含边界 => 两端去掉
            pts = np.linspace(low, high, num + 1, endpoint=False, dtype=config.real(np))
            return pts[1:]
        