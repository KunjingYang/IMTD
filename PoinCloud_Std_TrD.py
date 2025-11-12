import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import math
import open3d as o3d

data_all = ["data/heartp0.05"] 
################
#Here are the hyperparameters.
sr = 1
r = 6       # heart
lr_real = 0.00001 
thres = 0.01
down = 4
max_iter = 801
omega = 5
gamma_1 = 0.4
gamma_2 = 0.4
#################


    
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=omega): 
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Network(nn.Module):
    def __init__(self, r_1,r_2,r_3):
        super(Network, self).__init__()
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1))
        
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2))
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3))

    def forward(self, U_tube, V_tube, W_tube, x, flag):
        centre = torch.einsum('iqs,pjs,pqt->ijt', U_tube, V_tube, W_tube)
        U = self.U_net(x[:,0].unsqueeze(-1))
        V = self.V_net(x[:,1].unsqueeze(-1))
        W = self.W_net(x[:,2].unsqueeze(-1)) 
        if flag == 1:
            centre = centre.permute(1,2,0) 
            centre = centre @ U.t() 
            centre = centre.permute(2,1,0) 
            centre = torch.matmul(centre,V.unsqueeze(-1))  # V.unsqueeze的shape是 [299, 59, 1]
            centre = centre.permute(0,2,1) 
            centre = torch.matmul(centre,W.unsqueeze(-1)) 
        elif flag == 2:
            centre = centre.permute(1,2,0) 
            centre = centre @ U.t()
            centre = centre.permute(2,1,0)
            centre = centre @ V.t()
            centre = centre.permute(0,2,1) 
            centre = centre @ W.t()
        return centre

for data in data_all:
    
    # input dataset
    pcd = o3d.io.read_point_cloud(data+ '.pcd')
    X_np = np.array(pcd.points)[:,:]
    
    
    point_cloud = X_np
    num_sample = int(len(point_cloud) * sr)
    indices = np.random.choice(len(point_cloud), num_sample, replace=False)
    sampled_points = point_cloud[indices]

    # 可以再转成 Open3D 的 PointCloud 类型进行可视化等操作
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

    # 可视化
    # o3d.visualization.draw_geometries([sampled_pcd])
    X_np = np.array(sampled_pcd.points)[:,:]
    
    
    n = X_np.shape[0]
    print(n)

    # tucker rank
    mid_channel = 400
    r_1 = int(n/down)
    r_2 = int(n/down)
    r_3 = int(n/down)
    
    # 输入点的三维坐标存入U,V,W
    X_gt = torch.zeros(n,1).type(dtype)
    U_input = (torch.from_numpy(X_np[:,0])).reshape(n,1).type(dtype)
    U_input.requires_grad=True
    V_input = (torch.from_numpy(X_np[:,1])).reshape(n,1).type(dtype)
    V_input.requires_grad=True
    W_input = (torch.from_numpy(X_np[:,2])).reshape(n,1).type(dtype)
    W_input.requires_grad=True
    
    U_tube  = torch.Tensor(r_1,r,r).type(dtype)
    V_tube  = torch.Tensor(r,r_2,r).type(dtype)
    W_tube  = torch.Tensor(r,r,r_3).type(dtype)
    stdv = 1 / math.sqrt(n)
    U_tube.data.uniform_(-stdv, stdv)
    V_tube.data.uniform_(-stdv, stdv)
    W_tube.data.uniform_(-stdv, stdv)
    x_input = torch.cat((U_input, V_input, W_input),dim=1)
    
    # network setting
    model = Network(r_1,r_2,r_3).type(dtype)
    params = []
    params += [x for x in model.parameters()]
    U_tube.requires_grad=True
    params += [U_tube]
    V_tube.requires_grad=True
    params += [V_tube]
    W_tube.requires_grad=True
    params += [W_tube]
    optimizier = optim.Adam(params, lr=lr_real) 
    
    
    
    # main
    rand_num = 30
    add_border = 0.1
    for iter in range(max_iter):
        
        # 找到一个包围点云的立方体，并在里面随机取rand_num个点
        U_random = (torch.min(U_input)-add_border + 
                    (torch.max(U_input)-torch.min(U_input)+2*add_border) * torch.rand(rand_num,1).type(dtype)) 
        V_random = (torch.min(V_input)-add_border + 
                    (torch.max(V_input)-torch.min(V_input)+2*add_border) * torch.rand(rand_num,1).type(dtype))
        W_random = (torch.min(W_input)-add_border + 
                    (torch.max(W_input)-torch.min(W_input)+2*add_border) * torch.rand(rand_num,1).type(dtype))
        x_random = torch.cat((U_random,V_random, W_random),dim=1)  # rand_num个随机点
        
        
        X_Out = model(U_tube, V_tube, W_tube, x_input, flag = 1)
        loss_1 = torch.norm((X_Out)-X_gt,1)
        X_Out_off = model(U_tube, V_tube, W_tube, x_random, flag = 2)   # 随机取点时可以用规整的，故可以一次全部算出
        grad_ = gradient(X_Out_off,x_random)
        loss_2 = gamma_1 * torch.norm(grad_.norm(dim=-1)-rand_num**2,1)  
        loss_3 = gamma_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)),1) 
        loss = loss_1 + loss_2 + loss_3

        optimizier.zero_grad()
        loss.backward(retain_graph=True)
        optimizier.step()
        if iter % 200 == 0:
            print('iteration:', iter)
            number = 30
            range_ = torch.from_numpy(np.array(range(number))).type(dtype)
            u = (torch.min(U_input)-add_border + (
                torch.max(U_input)-torch.min(U_input)+2*add_border) * (range_/number)).reshape(number,1)
            v = (torch.min(V_input)-add_border + (
                torch.max(V_input)-torch.min(V_input)+2*add_border) * (range_/number)).reshape(number,1)
            w = (torch.min(W_input)-add_border + (
                torch.max(W_input)-torch.min(W_input)+2*add_border) * (range_/number)).reshape(number,1)
            x_in = torch.cat((u,v,w),dim=1)
            out = model(U_tube, V_tube, W_tube,x_in,flag = 2).detach().cpu().clone()  # out代表各个坐标输出的函数值
            idx = (torch.where(torch.abs(out)<thres))  # 值小于阈值才认为在点云上，然后才plot
            Pts = torch.cat((u[idx[0]],v[idx[1]]),dim = 1)
            Pts = torch.cat((Pts,w[idx[2]]),dim = 1).detach().cpu().clone().numpy()
            

        
            #Pts = X_np
            size_pc = 6
            fig = plt.figure(figsize=(10, 10), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1], projection='3d')

            xs = Pts[:,0]
            ys = Pts[:,1]
            zs = Pts[:,2]

            color_values = zs 

            # 绘制散点图，并使用 color_values 作为颜色依据
            sc = ax.scatter(xs, ys, zs, c=color_values, cmap='viridis', s=6)

            ax.set_axis_off()
            ax.grid(False)

            
            # 设置视角
            ax.view_init(elev=30, azim=90)

            output_path = r".\figure\ball_MITD.png" 
            plt.savefig(output_path, bbox_inches='tight', pad_inches=-1.9, transparent=False)# dpi 参数控制输出质量，bbox_inches='tight' 确保整个图表都被包含在内

     
            
            
