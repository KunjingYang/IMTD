import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
import random
from scipy.io import savemat

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2)

data_all =['data/peppers']  # sample rate is 0.2,  noise level is 0.4
c_all = [""]

################### 
# Here are the hyperparameters. 
w_decay = 0
lr_outer = 0.0002
outer_iter =  1501
omega =3
r = 9
r1, r2, r3 = r, r, 14 # outer
gamma = 0.0025
phi = 5*10e-6
mid_channel = 600
mu = 1




""" ----------------------------------------------- """
""" -------------      Function    ---------------- """
""" ----------------------------------------------- """

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()
    
    def forward(self, x, lam):
        x_abs = x.abs()-lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out
    
class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()
    
    def forward(self, X):
        # 计算第4维度上的差分
        diff = X[1:, :, :] - X[:-1, :, :]
        #diff = diff[:, 1:, :] - diff[:, :-1, :]
        return diff
    
def Triple_product(A, B, C):
    """
    Compute the triple-order tensor product using Einstein summation.
    
    Parameters:
        A: np.ndarray of shape (n1, r, r)
        B: np.ndarray of shape (r, n2, r)
        C: np.ndarray of shape (r, r, n3)
        
    Returns:
        X: np.ndarray of shape (n1, n2, n3)
    """
    # 使用 einsum 进行四重张量乘积
    X = torch.einsum('iqs,pjs,pqt->ijt', A, B, C)
    return X

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
    X = np.einsum('iqst,pjst,pqmt,pqsk->ijmk', A, B, C, D)
    return X
    
    
""" ----------------------------------------------- """
""" -------------        Net       ---------------- """
""" ----------------------------------------------- """

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
    def __init__(self, r1,r2,r3):
        super(Network, self).__init__()
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r2*r3),
                                   )
        
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r3*r1)
                                   )
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r1*r2)
                                   )
        
        
        self.conv1 = torch.nn.Conv2d(1, r3, kernel_size=3, padding=1) 
        self.conv2 = torch.nn.Conv2d(1, r1, kernel_size=3, padding=1) 
        self.conv3 = torch.nn.Conv2d(1, r2, kernel_size=3, padding=1) 

    def forward(self,  U_input, V_input, W_input, n1, n2, n3):
        
        U = self.U_net(U_input)
        V = self.V_net(V_input)
        W = self.W_net(W_input)
        
        U_tube = U.reshape(n1,r2,r3)
        V_tube = V.reshape(r1,n2,r3)
        W_tube = W.reshape(r1,r2,n3)
        
        X = Triple_product(U_tube,V_tube,W_tube)
        return  X



""" ----------------------------------------------- """
""" -------------        Main      ---------------- """
""" ----------------------------------------------- """

for data in data_all:
    for c in c_all:
        soft_thres=soft()
        TVS = TV()
        
        file_name = data+'.mat'
        mat = scipy.io.loadmat(file_name)
        X_np = mat["X1"]
        X = X_np.astype(np.float32)
        X = torch.from_numpy(X).type(dtype).cuda()
        
        file_name = data+'_gt.mat'
        mat = scipy.io.loadmat(file_name)
        gt_np = mat["X"]
        gt = torch.from_numpy(gt_np).type(dtype).cuda()
        
        # X = gt
        
        [n_1,n_2,n_3] = X.shape

        
        mask = torch.ones(X.shape).type(dtype)
        mask[X == 0] = 0 
        X[mask == 0] = 0
        

        U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype)
        V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)

        model = Network(r1,r2,r3).type(dtype)
        params = []
        params += [x for x in model.parameters()]

        optimizier = optim.Adam(params, lr=lr_outer, weight_decay=w_decay) 
        
        
        X_old = torch.zeros(X.shape).type(dtype)
        Y_old = torch.zeros(n_1-1, n_2, n_3).type(dtype)
        for iter in range(outer_iter):
            
            ## only low-rank
            X_Out = model(U_input, V_input, W_input, n_1, n_2, n_3)
            if iter == 0:
                X_Out_exp = X_Out.detach()
                D = torch.zeros([X.shape[0],X.shape[1],X.shape[2]]).type(dtype)
                S = (X-X_Out.clone().detach()).type(dtype)
                V = S.clone().detach().type(dtype)

                
            V = soft_thres(S + D / mu, gamma / mu)
            S = (2*X - 2 * X_Out.clone().detach()+ mu * V-D)/(2+mu)
            
            loss = torch.norm(X*mask-X_Out*mask-S*mask,2)
            loss = loss + phi*torch.norm(X_Out[1:,:,:]-X_Out[:-1,:,:],1) 
            loss = loss + phi*torch.norm(X_Out[:,1:,:]-X_Out[:,:-1,:],1) 
            
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            D = (D + mu * (S  - V)).clone().detach()
            
            if iter % 100 == 0:
                ps = peak_signal_noise_ratio(np.clip(gt.cpu().detach().numpy(),0,1),
                                             X_Out.cpu().detach().numpy())
                print('iteration:',iter,'PSNR',ps)
                
                # re = torch.norm(gt-X_Out)/torch.norm(gt)
                # print('iteration:',iter,'rel_error',re)
                
savemat("MITD_Out.mat", {'XX': X_Out.cpu().detach().numpy()})