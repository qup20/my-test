import deepxde as dde
import numpy as np
import pyvista as pv
import utils
from CustomBC import CustomDirichletBC, CoupledRobinBC, CoupledRobinBCT
from deepxde.utils import run_if_all_none
import torch


# 参考量
L_ref = 0.001  # mm
T_ref = 273.15  # K
k_ref = 398  #

# #点数#4090
n_boundary_heatsource = 100
n_domain_heatsource = 100
n_interface_chiplayer = 300
n_domain_jiban = 1000
n_domain_chiplayer = 1000
n_domain_heatspread = 1000
n_domain_heatsink = 1000
n_boundary_geo = 500

#点数#H800
# n_boundary_heatsource = 200
# n_domain_heatsource = 500
# n_interface_chiplayer = 1000
# n_domain_jiban = 5000
# n_domain_chiplayer = 5000
# n_domain_heatspread = 6000
# n_domain_heatsink = 8000
# n_boundary_geo = 2500

class CustomCSGIntersection(dde.geometry.CSGIntersection):
    def boundary_normal(self, x):
        # 先分别计算各几何体边界/内部的判断结果，避免重复计算
        g1_on_bd = self.geom1.on_boundary(x)
        g2_on_bd = self.geom2.on_boundary(x)
        g1_in = self.geom1.inside(x)
        g2_in = self.geom2.inside(x)
        
        # 情况1：点在geo1边界且geo2内部 --> 用geo1的法向分量
        case1_mask = np.logical_and(g1_on_bd, g2_in)
        
        # 情况2：点在geo2边界且geo1内部 --> 用geo2的法向分量
        case2_mask = np.logical_and(g2_on_bd, g1_in)
        
        # 新增情况3：点同时在两个几何体的边界 --> 强行用geo1的法向分量
        case3_mask = np.logical_and(g1_on_bd, g2_on_bd)
        
        return (
            case1_mask[:, np.newaxis] * self.geom1.boundary_normal(x)
            + case2_mask[:, np.newaxis] * self.geom2.boundary_normal(x)
            + case3_mask[:, np.newaxis] * self.geom1.boundary_normal(x)
        )
    


class CustomPDEOperator(dde.data.PDEOperator):
    def __init__(self, pde, function_space, evaluation_points, num_function, function_variables=None, num_test=None,device=None):
        # super().__init__(pde, function_space, evaluation_points, num_function, function_variables, num_test)
        super(dde.data.PDEOperator, self).__init__()
        self.pde = pde
        self.func_space = function_space
        self.eval_pts = evaluation_points
        self.num_func = num_function
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(pde.geom.dim))
        )
        self.num_test = num_test

        self.num_bcs = [n * self.num_func for n in self.pde.num_bcs]
        self.train_bc = None
        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        self.origin_pde = pde.pde
        self.origin_bcs = pde.bcs
        self.origin_geom = pde.geom
        self.origin_anchors = pde.anchors
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 定义几何体
        self.geo_jiban = dde.geometry.Cuboid([-5, -5, -1.02], [5, 5, -0.02])
        self.geo_chiplayer = dde.geometry.Cuboid([-1.6, -1.6, -0.02], [1.6, 1.6, 0.36])
        self.geo_heatspead = dde.geometry.Cuboid([-5, -5, 0.36], [5, 5, 2.36])
        self.geo_heatsink = dde.geometry.Cuboid([-7.5, -7.5, 2.36], [7.5, 7.5, 6.36])

        # 定义固定材料边界
        self.surface_jiban_chiplayer = CustomCSGIntersection(self.geo_jiban, self.geo_chiplayer)
        self.surface_heatspread_chiplayer = CustomCSGIntersection(self.geo_heatspead, self.geo_chiplayer)

        self.num_boundary_points_geom1 = n_interface_chiplayer
        self.num_boundary_points_geom2 = n_interface_chiplayer

        
    # def sample_side_wall_points(self, n, tolerance=1e-5):
    #     """
    #     从 CSGUnion 几何体中采样 n 个侧壁点。
    
    #     参数:
    #         geom: CSGUnion 几何体对象。
    #         n: 需要采样的侧壁点数量。
    #         tolerance: 判断点是否在侧壁上的容差。
    
    #     返回:
    #         side_wall_points: 采样到的侧壁点，形状为 (n, dim)。
    #     """
            
    #     # 采样边界点，直到获得足够的侧壁点
    #     side_wall_points = []
    #     while len(side_wall_points) < n:
    #         # 每次采样 2n 个边界点，以提高效率
    #         boundary_points = self.origin_geom.random_boundary_points(2 * n)
    #         for x in boundary_points:
    #             if is_side_wall(x, on_boundary=True):
    #                 side_wall_points.append(x)
    #                 if len(side_wall_points) >= n:
    #                     break
    
    #     return np.array(side_wall_points)[:n]  # 返回前 n 个点

        # Todo:加入交界面的interface
    def get_heats_attrs(self, 
                        func_feats, 
                        geom, 
                        bc_fn=None,
                        dx=1,
                        dy=1,
                        dz=0.01,
                        ):
        """
        定义多个小长方体的边界条件，并返回相交面和 geom 的 Union。
    
        Args:
            func_feats (numpy.ndarray): 热源位置，形状为 (size, n_sources*3)，每行是 (x0, y0, z0)。
            geom (dde.geometry.Geometry): 给定的几何对象。
            dx (float): 小长方体的 x 方向大小。
            dy (float): 小长方体的 y 方向大小。
            dz (float): 小长方体的 z 方向大小。
            bc_fn (callable): 用于定义边界条件的函数，输入为坐标点。
    
        Returns:
            boundary_points (list): 每个小长方体与 geom 相交面的边界点。
            boundary_conditions (list): 对应的小长方体边界条件的列表。
        """
        n_sources = func_feats.shape[1] // 3  # 获取小长方体数量
        boundary_points = []
        boundary_conditions = []
        cuboids =[]
        domain_heatsouce_points=[]
        
        for i in range(n_sources):
            # 提取每个小长方体的左下角坐标
            x0, y0, z0 = func_feats[:, 3 * i], func_feats[:, 3 * i + 1], func_feats[:, 3 * i + 2]
    
            for j in range(len(x0)):  # 遍历每一行数据（热源实例）
                cuboid = dde.geometry.Cuboid([x0[j], y0[j], z0[j]], [x0[j] + dx, y0[j] + dy, z0[j] + dz])
    
                # 计算小长方体与给定几何体的相交面
                # intersection = dde.geometry.CSGIntersection(cuboid, geom)
                # intersections.append(intersection)
    
                # # 获取边界点
                boundary = cuboid.random_boundary_points(n=n_boundary_heatsource,random="pseudo") # 点数
                domain_heatsouce = cuboid.random_points(n=n_domain_heatsource)# 点数
                boundary_points.append(boundary)
    
                # 定义边界条件
                bc = CoupledRobinBC(
                    geom=cuboid,
                    on_boundary=cuboid.on_boundary,  
                )
                boundary_conditions.append(bc)
                bc = CoupledRobinBCT(
                    geom=cuboid,
                    on_boundary=cuboid.on_boundary,  
                )
                boundary_conditions.append(bc)
                domain_heatsouce_points.append(domain_heatsouce)
                cuboids.append(cuboid)
    
        return boundary_points, boundary_conditions,cuboids,domain_heatsouce_points

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        # 1.生成热源空间位置func_feats
        func_feats = self.func_space.random(self.num_func)

        # 2.热源空间边界点
        boundary_points, boundary_conditions, cuboids, domain_heatsource_points  = self.get_heats_attrs(func_feats, self.origin_geom, bc_fn=None)
        
        boundary_points = np.array(boundary_points)
        boundary_points = boundary_points.reshape(-1, 3)
        # print(boundary_points,type(boundary_points))
        # pv.PolyData(boundary_points).plot()
        domain_heatsource_points = np.array(domain_heatsource_points)
        domain_heatsource_points  = domain_heatsource_points.reshape(-1,3)
        
        # 取固定材料之间边界点
        boundary_points_jiban_chiplayer = self.surface_jiban_chiplayer.random_boundary_points(self.num_boundary_points_geom1)
        # pv.PolyData(boundary_points_jiban_chiplayer).plot()
        boundary_points_heatspread_chiplayer = self.surface_heatspread_chiplayer.random_boundary_points(self.num_boundary_points_geom2)
        boundary_points_2 = np.vstack((boundary_points_jiban_chiplayer, boundary_points_heatspread_chiplayer))
        boundary_points = np.vstack((boundary_points, boundary_points_2))
        # pv.PolyData(boundary_points).plot()

        domain_geo_jiban = self.geo_jiban.random_points(n=n_domain_jiban)# 点数
        domain_geo_chiplayer = self.generate_interior_points(self.geo_chiplayer, cuboids, n_domain_chiplayer)# 点数
        # pv.PolyData(domain_geo_chiplayer).save("domain_geo_chiplayer.vtk")
        # domain_geo_chiplayer = self.geo_chiplayer.random_points(n=1500)# 点数
        domain_geo_heatspead = self.geo_heatspead.random_points(n=n_domain_heatspread)# 点数
        domain_geo_heatsink = self.geo_heatsink.random_points(n=n_domain_heatsink)# 点数
        all_anchors = np.vstack((boundary_points, domain_geo_jiban, domain_geo_chiplayer, domain_geo_heatspead, domain_geo_heatsink,domain_heatsource_points))
        # side_wall_points = self.sample_side_wall_points(n=500)
        # boundary_points = np.vstack((boundary_points, side_wall_points))
        # bc_cebi = CustomNeumannBC(self.origin_geom, lambda x: 0, is_side_wall)
        
        # 自定义边界条件（针对相切面）
        def boundary_condition_jiban_chiplayer(x, on_boundary):
            return self.surface_jiban_chiplayer.inside(x)
    
        def boundary_condition_heatspread_chiplayer(x, on_boundary):
            return self.surface_heatspread_chiplayer.inside(x)
        
        # 固定材料之间边界条件
        bcs_2 = [
            CoupledRobinBC(self.surface_jiban_chiplayer, boundary_condition_jiban_chiplayer),
            CoupledRobinBC(self.surface_heatspread_chiplayer,boundary_condition_heatspread_chiplayer),
            CoupledRobinBCT(self.surface_jiban_chiplayer, boundary_condition_jiban_chiplayer),
            CoupledRobinBCT(self.surface_heatspread_chiplayer,boundary_condition_heatspread_chiplayer)
        ]
    
        all_bcs = self.origin_bcs + boundary_conditions + bcs_2 # + [bc_cebi]
        # all_anchor = np.concatenate([self.origin_anchors, boundary_points], axis=0)
        # 3.把热源空间边界点加入到anchors
        self.pde = dde.data.PDE(self.origin_geom, 
                                pde=self.origin_pde, 
                                bcs = all_bcs, 
                                num_domain=0, 
                                num_boundary=n_boundary_geo,# 上下表面+侧壁点数
                                train_distribution="uniform",
                                anchors=all_anchors
                                )
        self.num_bcs = [n * self.num_func for n in self.pde.num_bcs]
    
        func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
        v, x, vx = self.bc_inputs(func_feats, func_vals)
        if self.pde.pde is not None:
            v_pde, x_pde, vx_pde = self.gen_inputs(
                func_feats, func_vals, self.pde.train_x_all
            )
            v = np.vstack((v, v_pde))
            x = np.vstack((x, x_pde))
            vx = np.vstack((vx, vx_pde))
            v = torch.tensor(v, dtype=torch.float32, device=self.device)
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        # x 是训练点的坐标
        # D_factor = utils.getDFactorV2(x, func_feats, boundary_points)
        # D_factor = utils.getDFactorV2(x, func_feats, boundary_points)
        # D_factor = D_factor.to(self.device)
        D_factor = utils.getDFactorV2(x.cpu().numpy(), func_feats, boundary_points, device=self.device).T
        device = x.device
        D_factor=D_factor.to(device)
        # D_factor = torch.as_tensor(D_factor, dtype=torch.float32, device=self.device)
        # assert x.device == D_factor.device
        self.train_x = (v, x, D_factor)
        self.train_aux_vars = vx
        # print(v.device,x.device,D_factor.device)
        return self.train_x, self.train_y, self.train_aux_vars


    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_aux_vars = self.train_aux_vars
        else:
            func_feats = self.func_space.random(self.num_test)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            
            boundary_points, boundary_conditions, cuboids,domain_heatsource_points  = self.get_heats_attrs(func_feats, self.origin_geom, bc_fn=None)
            # all_bcs = self.origin_bcs + boundary_conditions
            boundary_points = np.array(boundary_points)
            boundary_points = boundary_points.reshape(-1, 3)
            domain_heatsource_points = np.array(domain_heatsource_points)
            domain_heatsource_points  = domain_heatsource_points.reshape(-1,3)

    
            boundary_points_jiban_chiplayer = self.surface_jiban_chiplayer.random_boundary_points(self.num_boundary_points_geom1)
            boundary_points_heatspread_chiplayer = self.surface_heatspread_chiplayer.random_boundary_points(self.num_boundary_points_geom2)
            boundary_points_2 = np.vstack((boundary_points_jiban_chiplayer, boundary_points_heatspread_chiplayer))
            boundary_points = np.vstack((boundary_points, boundary_points_2))
        
            domain_geo_jiban = self.geo_jiban.random_points(n=n_domain_jiban)# 点数
            domain_geo_chiplayer = self.generate_interior_points(self.geo_chiplayer, cuboids, n_domain_chiplayer)# 点数
            # domain_geo_chiplayer = self.geo_chiplayer.random_points(n=1500)# 点数
            domain_geo_heatspead = self.geo_heatspead.random_points(n=n_domain_heatspread)# 点数
            domain_geo_heatsink = self.geo_heatsink.random_points(n=n_domain_heatsink)# 点数
            all_anchors = np.vstack((boundary_points, domain_geo_jiban, domain_geo_chiplayer, domain_geo_heatspead, domain_geo_heatsink,domain_heatsource_points))
        

             # 自定义边界条件（针对相切面）
            def boundary_condition_jiban_chiplayer(x, on_boundary):
                return self.surface_jiban_chiplayer.inside(x)
    
            def boundary_condition_heatspread_chiplayer(x, on_boundary):
                return self.surface_heatspread_chiplayer.inside(x)

            bcs_2 = [
            CoupledRobinBC(self.surface_jiban_chiplayer, boundary_condition_jiban_chiplayer),
            CoupledRobinBC(self.surface_heatspread_chiplayer,boundary_condition_heatspread_chiplayer),
            CoupledRobinBCT(self.surface_jiban_chiplayer, boundary_condition_jiban_chiplayer),
            CoupledRobinBCT(self.surface_heatspread_chiplayer,boundary_condition_heatspread_chiplayer)
            ]
    
            
            all_bcs = self.origin_bcs + boundary_conditions + bcs_2 # + [bc_cebi]

            
            # 3.把热源空间边界点加入到anchors
            self.pde = dde.data.PDE(self.origin_geom, 
                                    pde=self.origin_pde, 
                                    bcs = all_bcs, 
                                    num_domain=0, 
                                    num_boundary=n_boundary_geo,# 点数
                                    train_distribution="uniform",
                                    anchors=all_anchors
                                    )
            self.num_bcs = [n * self.num_test for n in self.pde.num_bcs]
            

            # v, x, vx = self.train_bc
            v, x, vx = self.bc_inputs(func_feats, func_vals)
            if self.pde.pde is not None:
                v_pde, x_pde, vx_pde = self.gen_inputs(
                    func_feats, func_vals, self.pde.test_x[sum(self.pde.num_bcs) :]
                )
                v = np.vstack((v, v_pde))
                x = np.vstack((x, x_pde))
                vx = np.vstack((vx, vx_pde))
                v = torch.tensor(v, dtype=torch.float32, device=self.device)
                x = torch.tensor(x, dtype=torch.float32, device=self.device)


            # D_factor = utils.getDFactorV2(x, func_feats, boundary_points)
            # D_factor = utils.getDFactorV2(x, func_feats, boundary_points)
            # D_factor = torch.as_tensor(D_factor, dtype=torch.float32, device=self.device)
            D_factor = utils.getDFactorV2(x.cpu().numpy(), func_feats, boundary_points, device=self.device).T
            # assert x.device == D_factor.device
            self.test_x = (v, x, D_factor)
            self.test_aux_vars = vx
        return self.test_x, self.test_y, self.test_aux_vars
    
    def generate_interior_points(self, geo_A, geo_list, n, max_attempts=10000):
        """
        生成 geo_A 的域内点，确保这些点不在 geo_list 中任何几何体的边界上。

        参数:
            geo_A: 主几何体（dde.geometry 对象）。
            geo_list: 几何体列表（list of dde.geometry 对象）。
            n: 需要生成的域内点数量。
            max_attempts: 最大尝试次数（避免无限循环）。

        返回:
            interior_points: 符合条件的域内点，形状为 (n, dim)。
        """
        interior_points = []
        attempts = 0

        while len(interior_points) < n and attempts < max_attempts:
            # 生成 geo_A 的随机域内点
            point = geo_A.random_points(1)[0]

            # 检查点是否在 geo_list 中任何几何体的边界上
            on_any_boundary = any(geo.on_boundary(point) for geo in geo_list)

            # 如果点不在任何边界上，则添加到结果中
            if not on_any_boundary:
                interior_points.append(point)

            attempts += 1

        if len(interior_points) < n:
            raise RuntimeError(f"无法生成足够的域内点（尝试次数：{max_attempts}）。")

        return np.array(interior_points)[:n]

