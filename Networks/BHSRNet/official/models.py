# -----------------------------------------------------------------------------
# If you use this code in your research, please cite our paper:
# Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model
# Thanks
# -----------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from .modules import *

class BHSR(nn.Module):  
    def __init__(self, stage_nBHSR, C, c, sigma1, sigma2):
        super(BHSR, self).__init__()
        self.stage_nBHSR = stage_nBHSR
        
        self.gx_modules = nn.ModuleList([GX(C, c, sigma1) for _ in range(stage_nBHSR)])
        self.x_solvers = nn.ModuleList([XSolver(C) for _ in range(stage_nBHSR)])
        self.l1_mus = nn.ModuleList([L1Updater(C, sigma1) for _ in range(stage_nBHSR)])

        self.y_solvers = nn.ModuleList([YSolver(c, C, sigma2) for _ in range(stage_nBHSR)])
        self.l2_mus = nn.ModuleList([L2Updater(c, sigma2) for _ in range(stage_nBHSR)])

        self.gx_update_param = nn.Parameter(torch.tensor(1.0))  
        self.l1_update_param = nn.Parameter(torch.tensor(1.0)) 
        self.y_update_param = nn.Parameter(torch.tensor(1.0)) 
        self.l2_update_param = nn.Parameter(torch.tensor(1.0)) 

    def forward(self, z, y):
      
        H, W = y.shape[2], y.shape[3]
        z_ = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)
        y_ = y
        x = z_
        L1 = torch.zeros_like(x)  
        L2 = torch.zeros_like(y)  
        Xs = []

        for i in range(self.stage_nBHSR):

            gx = self.gx_modules[i](x, L1, y_, z_)
            x = x + self.x_solvers[i](gx) * self.gx_update_param
            L1 = L1 + self.l1_mus[i](L1, x)* self.l1_update_param  
            y_ = y_ + self.y_solvers[i](y, y_, x, L2) * self.y_update_param
            L2 = L2 + self.l2_mus[i](L2, y_) * self.l2_update_param
            Xs.append(x)

        return Xs,y_
