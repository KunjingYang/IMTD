import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
import random



data_all =["data/plane"]
c_all = ["2"]

################### 
# Here are the hyperparameters. 
w_decay = 1
lr_real = 0.0005
outer_iter =  2001
inner_iter = 101
r = 4
r_1, r_2, r_3, r_4 =  r, r, r, r

omega = 3
gamma = 0.1
mid_channel = 200
mid_channel1 = 50

###################
class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()
    
    def forward(self, x, lam):
        x_abs = x.abs()-lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out

def Quadruple_product(A, B, C, D):
    """
    Compute the fourth-order tensor product using Einstein summation.
    
    Parameters:
        A: np.ndarray of shape (n1, r, r, r)
        B: np.ndarray of shape (r, n2, r, r)
        C: np.ndarray of shape (r, r, n3, r)
        D: np.ndarray of shape (r, r, r, n4)
        
    Returns:
        X: np.ndarray of shape (n1, n2, n3, n4)
    """
    # 使用 einsum 进行四重张量乘积
    X = torch.einsum('iqst,pjst,pqmt,pqsk->ijmk', A, B, C, D)
    return X

class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()
    
    def forward(self, X):
        # 计算第4维度上的差分
        diff = X[:, :, :, 1:] - X[:, :, :, :-1]
        
        # 使用 torch.norm 计算 L2 范数，并对所有维度求和
        loss_TV = torch.norm(diff, p=2)  # 默认使用 L2 范数
        
        return loss_TV



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
    def __init__(self, r_1,r_2,r_3,r_4):
        super(Network, self).__init__()
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2*r_3*r_4))
        
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1*r_3*r_4))
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1*r_2*r_4))
        
        self.R_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1*r_2*r_3))

    def forward(self, U_input, V_input, W_input, R_input):
        
        U = self.U_net(U_input).reshape(U_input.size(0),r_2,r_3,r_4)
        V = self.V_net(V_input).reshape(r_1,V_input.size(0),r_3,r_4)
        W = self.W_net(W_input).reshape(r_1,r_2,W_input.size(0),r_4)
        R = self.R_net(R_input).reshape(r_1,r_2,r_3,R_input.size(0))
        X = Quadruple_product(U, V, W, R)

        return X, U, V, W, R
    




for data in data_all:
    for c in c_all:
        soft_thres=soft()
        R = 4
        I1, I2, I3, I4 = 7, 8, 9, 10
        r1, r2, r3, r4 = R, 5, R, 6

        A = np.random.rand(I1, r2, r3, r4)
        B = np.random.rand(r1, I2, r3, r4)
        C = np.random.rand(r1, r2, I3, r4)
        D = np.random.rand(r1, r2, r3, I4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        A = torch.tensor(A, dtype=torch.float32).to(device)
        B = torch.tensor(B, dtype=torch.float32).to(device)
        C = torch.tensor(C, dtype=torch.float32).to(device)
        D = torch.tensor(D, dtype=torch.float32).to(device)
        X = Quadruple_product(A, B, C, D)

        
        [n_1,n_2,n_3,n_4] = X.shape
        
        
   

        U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype)
        V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)
        R_input = torch.from_numpy(np.array(range(1,n_4+1))).reshape(n_4,1).type(dtype)

        model = Network(r_1,r_2,r_3,r_4).type(dtype)
        params = []
        params += [x for x in model.parameters()]
        optimizier = optim.Adam(params, lr=lr_real, weight_decay=w_decay) 
        
        
        S = torch.zeros(X.shape).type(dtype)
        for iter in range(outer_iter):
            

            X_Out, U_tube, V_tube, W_tube, R_tube = model( U_input, V_input, W_input, R_input)
            loss = torch.norm(X_Out-X,2)
            

            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            
            if iter % 500 == 0:
                re1 = torch.norm(X_Out-X)/torch.norm(X)
                print('iteration:',iter,'rel_eror',re1)

