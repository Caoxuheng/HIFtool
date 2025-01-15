import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from .Model.net import  *
from .Model.Spa_downs import *
from .Model.function import *



# learning rate decay
def LR_Decay(optimizer, n,lr):
    lr_d = lr * (0.5**n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_d





def meta_train_adaptor(GT,opt,device):
    lr = 5e-5
    lr_da = 1e-4
    lr_dc = 1e-4

    fusion_model = torch.load(opt.fusion_model_path)
    fusion_model = fusion_model.to(device)

    factor = opt.sf
    KS = 32
    kernel = torch.rand(1, 1, KS, KS)
    kernel[0, 0, :, :] = torch.from_numpy(get_kernel(factor, 'gauss', 0, KS, sigma=3))
    Conv = nn.Conv2d(1, 1, KS, factor)
    Conv.weight = nn.Parameter(kernel)
    dow = nn.Sequential(nn.ReplicationPad2d(int((KS - factor) / 2.)), Conv)
    downs = Apply(dow, 1).cuda()
    optimizer_d = torch.optim.Adam(downs.parameters(), lr=lr_da, weight_decay=1e-5)

    model = FineNet_SelfAtt_InputK_P_V2().to(device)
    # loss & optimizer
    L1 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # spatial and spectral downsampler
    P = sio.loadmat(opt.pre_srf)
    P_ob = Variable(torch.from_numpy(P[opt.pre_srf_key])).type(torch.FloatTensor).to(device)
    # Learnable spectral downsampler
    down_spc = L_Dspec(opt.hsi_channel, opt.msi_channel, P_ob).type(torch.FloatTensor).to(device)
    optimizer_spc = torch.optim.Adam(down_spc.parameters(), lr=lr_dc, weight_decay=1e-5)

    WS = [[7, 1 / 2], [8, 3], [9, 2], [13, 4], [15, 1.5]]

    n=0

    for epoch in range(100):
        # Randomly sampling the spatial downsampler
        ws = np.random.randint(0,5,1)[0]
        ws = WS[ws]
        ks = np.random.randint(2,9,1)[0]
        down_spa = Spa_Downs(
            opt.hsi_channel, opt.sf, kernel_type=ks, kernel_width=ws[0],
            sigma=ws[1], preserve_size=True
        ).to(device)
        [K_GT, K_Size] = down_spa.Return_Kernel()
        mins, mod = (KS-K_GT.shape[1])/2, (KS-K_GT.shape[1])%2
        pad_size_GT = [int(mins), int(mins)+mod, int(mins), int(mins)+mod]

        K_GT = Variable(torch.from_numpy(K_GT), requires_grad=False).type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0).repeat(GT.shape[0], 1, 1, 1)
        K_GT = torch.nn.functional.pad(K_GT, pad_size_GT, mode='constant', value=0)

        # Randomly sampling the spectral response matrix
        threshold = torch.rand(1)[0]
        if threshold > 0.3:
            R_deg = 1e-4 * torch.randint(50, 80, [1])[0]
            P_re = torch.unsqueeze((P_ob + R_deg)/(P_ob.sum(1).unsqueeze(1) + R_deg*GT.shape[1]), 0)
        else:
            P_re = P_ob.unsqueeze(0)


        # Generate the LR_HSI
        LR_HSI = down_spa(GT)
        # Generate the HR_MSI
        HR_MSI = torch.matmul(P_re, GT.reshape(-1, GT.shape[1], GT.shape[2]*GT.shape[3])).reshape(-1, 3, GT.shape[2], GT.shape[3])
        # Generate the UP_HSI
        LR_HSI = Variable(LR_HSI.detach(), requires_grad=False).type(torch.cuda.FloatTensor)
        HR_MSI = Variable(HR_MSI.detach(), requires_grad=False).type(torch.cuda.FloatTensor)

        with torch.no_grad():
            out = fusion_model(HR_MSI,LR_HSI)

        param_K = list(downs.named_parameters())
        K = param_K[0][1][0].detach().unsqueeze(0).repeat(GT.shape[0], 1, 1, 1)
        param_P = list(down_spc.named_parameters())
        P = param_P[0][1].detach().permute(1,0).unsqueeze(0).unsqueeze(0).repeat(GT.shape[0], 1, 1, 1)

        # MetaLearning for Alter estimating the K, P
        param_spa = downs.state_dict()
        param_spc = down_spc.state_dict()

        out_d = downs(out.detach())
        out_spc = down_spc(out.detach())
        optimizer_d.zero_grad()
        optimizer_spc.zero_grad()
        loss_d = L1(out_d, LR_HSI)
        loss_spc = L1(out_spc, HR_MSI)
        loss_spc.backward()
        loss_d.backward()
        optimizer_d.step()
        optimizer_spc.step()

        out = Variable(out.detach(), requires_grad=False).type(torch.FloatTensor).to(device)

        out_d = downs(out.detach())
        out_spc = down_spc(out.detach())
        optimizer_d.zero_grad()
        optimizer_spc.zero_grad()
        loss_d = L1(out_d, LR_HSI)
        loss_spc = L1(out_spc, HR_MSI)
        loss_spc.backward()
        loss_d.backward()
        downs.load_state_dict(param_spa)
        down_spc.load_state_dict(param_spc)
        optimizer_d.step()
        optimizer_spc.step()

        # furthermore meta-learning for adaptation module
        model_param = model.state_dict()

        # update Theta to Theta_i
        out = model(out, K, P)

        HSI_out = downs(out)
        MSI_out = down_spc(out)
        loss1 = L1(HSI_out, LR_HSI)
        loss2 = L1(MSI_out, HR_MSI)
        optimizer.zero_grad()
        optimizer_d.zero_grad()
        optimizer_spc.zero_grad()
        loss = 1 * loss1 + 2 * loss2
        loss.backward()
        optimizer.step()
        optimizer_d.step()
        optimizer_spc.step()

        out = Variable(out.detach(), requires_grad=False).type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        optimizer_d.zero_grad()
        optimizer_spc.zero_grad()
        out_ = model(out, K, P)
        loss = L1(out_, GT)
        loss.backward()
        model.load_state_dict(model_param)
        optimizer.step()
        optimizer_d.step()
        optimizer_spc.step()


        if epoch%10 == 9:
            LR_Decay(optimizer, n,lr)
            n += 1


        if epoch%10 == 9:
            torch.save(model, opt.save_path+'_model_'+str(int(epoch/10))+'.pth')
            torch.save(downs, opt.save_path+'_Donwspa_'+str(int(epoch/10))+'.pth')
            torch.save(down_spc, opt.save_path+'_Donwspc_'+str(int(epoch/10))+'.pth')






