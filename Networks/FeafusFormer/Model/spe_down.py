import cv2
import torch
from torch import nn
from torch.nn import init

def trans1(data):
    return data[0].detach().numpy().T
def trans2(data):
    return torch.tensor(data.T).float().unsqueeze(0)


def getGaussiankernel(ksize,sigma):
    x = torch.arange( -(ksize-1)//2,(ksize-1)//2+1)
    kernel = torch.exp(-1/2*torch.pow(x/sigma,2))
    kernel/=kernel.sum()
    return kernel.view(-1,1).float()

def init(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            m.weight.data.fill_(1 /3 )
    net.apply(init_func)




class SpeDown(nn.Module):
    def __init__(self,span,predefine=None,iscal=True):
        super(SpeDown,self).__init__()
        msi_band = len(span)
        span_incep = [len(span[i]) for i in range(msi_band)]
        self.span = span
        self.ED = nn.ModuleList([nn.Conv2d(in_channels= _i0_,out_channels=1,kernel_size=1,stride=1,
                                          padding=0,bias=None) for _i0_ in span_incep ])
        self.act = nn.ReLU()
        self.iscal = iscal

        if iscal is False:
            self.spedown = nn.Conv2d(in_channels=31,out_channels=3,kernel_size=1,bias=None)
            if predefine is not None:
                kernel = torch.tensor(predefine.T).float()[:,:,None,None]
                self.spedown.weight.data = kernel

            else:
                init(self.specdown)

            self.spedown.requires_grad_(iscal)

    def forward(self,x):
        if self.iscal is False:

            return self.spedown(x)
        else:
            msi_list = []
            for idx,spe_d in enumerate(self.ED):
                spe_d.weight.data = torch.maximum(torch.tensor(1e-2),spe_d.weight.data)
                pan_msi =self.act( spe_d(x[:,self.span[idx][0]:self.span[idx][-1]+1,:,:]))
                msi_list.append(pan_msi)

            return torch.concat(msi_list,axis=1)

    def get_srf(self):
         srf = np.zeros([31,3])
         for i in range(3):
            a1 = self.ED[i].weight.ravel().tolist()
            srf[self.span[i][0]: self.span[i][-1] + 1,i] = a1
         return srf

class SpeDown(nn.Module):
    def __init__(self,span,predefine=None,iscal=True):
        super(SpeDown,self).__init__()
        self.iscal = iscal

        msi_band = len(span)
        span_incep = [len(span[i]) for i in range(msi_band)]
        self.span = span
        self.act = nn.ReLU()

        if iscal == True:
            if predefine ==None:
                self.srf_list = nn.ParameterList([nn.Parameter(getGaussiankernel(span_incep[i],2)[:,0]) for i in range(msi_band)])
                self.conv_lst = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding='same') for i in range(msi_band)])
        if iscal is False:
            self.spedown = nn.Conv2d(in_channels=31,out_channels=3,kernel_size=1,bias=None)
            if predefine is not None:
                kernel = torch.tensor(predefine.T).float()[:,:,None,None]
                self.spedown.weight.data = kernel
            else:
                raise AttributeError('Need predefined spectral response function when iscal == True')
            self.spedown.requires_grad_(iscal)

    def forward(self,x):
        if self.iscal is False:
            return self.spedown(x)
        else:

            msi_list = []
            for idx,layer in enumerate(self.srf_list):
                layer =self.act(self.conv_lst[idx](layer.unsqueeze(0)))
                Extr_BandImg = x[:, self.span[idx][0]:self.span[idx][-1] + 1, :, :]
                b,c,h,w = Extr_BandImg.shape
                out = Extr_BandImg.view(c,h*w)
                out = torch.matmul(layer, out)
                msi_list.append(out.view(b,1,h,w))
                layer_d1 = layer[:,1:]- layer[:,:-1]
                layer_d2 = layer_d1[:,1:]- layer_d1[:,:-1]

            return torch.concat(msi_list, axis=1)


    def get_srf(self):
        srf = torch.zeros([31,3])
        srf[self.span[0][0]:self.span[0][-1]+1, 0] = self.act(self.conv_lst[0](self.srf_list[0].unsqueeze(0)))
        srf[self.span[1][0]:self.span[1][-1]+1, 1]= self.act(self.conv_lst[1](self.srf_list[1].unsqueeze(0)))
        srf[self.span[2][0]:self.span[2][-1]+1, 2]= self.act(self.conv_lst[2](self.srf_list[2].unsqueeze(0)))
        return srf.detach()


def get_tvloss(module,band = 3):
    band = 3
    module = downnet
    para_list =list( module.parameters())
    # para_list = trans2(srf)[0]

    der2_total = 0
    for i in range(band):
        para_i = para_list[i].flatten()

        der_1 = para_i[1:]-para_i[:-1]
        der_2 = der_1[1:]-der_1[:-1]

        der2_total+=(torch.abs(der_2).mean())

    return torch.abs(torch.maximum(torch.tensor(0.011),der2_total)-0.011)



def initialize_SpeDNet_test(module, msi, hsi, sf):
    # 2022年7月24日
    msi_1 = nn.functional.avg_pool2d(nn.functional.pad(msi, pad=[sf ] * 4, mode='reflect'),
                                     kernel_size=2 * sf+1, stride=sf)
    hsi_1 = nn.functional.avg_pool2d(nn.functional.pad(hsi, pad=[1] * 4, mode='reflect'),
                                     kernel_size=3, stride=1)

    trainer = torch.optim.Adam(params=module.parameters(), lr=0.005,weight_decay=0.0001)
    lrsched=torch.optim.lr_scheduler.StepLR(trainer,50,0.9)

    max_epochs = 500
    psnrs2 = []
    psnrs=[]
    L1 = nn.L1Loss()
    for epoch in range(max_epochs):
        trainer.zero_grad()
        pre_msi = module(hsi_1)
        psnr = L1(msi_1, pre_msi)
        # psnr = L1(msi_1, pre_msi) +  torch.log10(get_tvloss(module))
        psnr.backward()
        trainer.step()
        lrsched.step()

        # psnr2 = PSNR_GPU(module(trans2(img)),msi)

        psnrs.append(psnr.detach().numpy())
        # psnrs2.append(psnr2.detach().numpy())

    print('Initialized results')
    psnr2 = PSNR_GPU(module(trans2(img)), msi)
    print(psnr2)
    print(PSNR_GPU( msi_1,pre_msi))
    print(trainer.state_dict()['param_groups'][0]['lr'])
    # plt.subplot(1,2,1),plt.plot(psnrs2)
    plt.subplot(122),plt.plot(psnrs)
    plt.show()
    return module

def initialize_SpeDNet(module, msi, hsi, sf):
    # 2022年7月24日
    msi_1 = nn.functional.avg_pool2d(nn.functional.pad(msi, pad=[sf] * 4, mode='reflect'),
                                     kernel_size=2 * sf + 1, stride=sf)
    hsi_1 = nn.functional.avg_pool2d(nn.functional.pad(hsi, pad=[1] * 4, mode='reflect'),
                                     kernel_size=3, stride=1)
    trainer = torch.optim.Adam(params=module.parameters(), lr=0.01, weight_decay=0.0001)
    lrsched = torch.optim.lr_scheduler.StepLR(trainer, 50, 0.9)
    max_epochs = 500
    L1 = nn.L1Loss()
    for epoch in range(max_epochs):
        trainer.zero_grad()
        pre_msi = module(hsi_1)
        psnr = L1(msi_1, pre_msi)
        psnr.backward()
        trainer.step()
        lrsched.step()

    print('Initialize Spectral Degradation Net Successfully. Epoch:{}\tlr:{:.2e}\tPSNR:{:.2f}'.format(epoch,trainer.state_dict()['param_groups'][0]['lr'],-psnr.detach().cpu().numpy()))
    return module



if __name__ == '__main__':
    import scipy.io as sio
    import imgvision as iv
    import matplotlib.pyplot as plt
    from utils import PSNR_GPU
    import numpy as np

    torch.manual_seed(10)

    # np.random.seed(10)
    sf = 32
    span = [list(range(18, 31)), list(range(10, 23)), list(range(12))]


    srf = np.load('G:/FusionBasedHSISR/UMGAL/NikonD700.npy')
    downnet = SpeDown(span)



    # a = get_tvloss(module=downnet)
    # print(a)


    img = sio.loadmat('F:/CAVE/1.mat')['HSI']
    # img = torch.rand(1,3,512,512)
    im = torch.tensor(img.T).float().unsqueeze(0)
    hsi = trans2( cv2.GaussianBlur(img,(sf-1,sf-1),sf//2)[sf//2-1::sf,sf//2-1::sf] )
    msi_ = trans2(iv.spectra().space(img,'nkd700'))
    initialize_SpeDNet(downnet,msi_,hsi,sf)

    # a = get_tvloss(module=downnet)
    # print(a)
    a = downnet.get_srf()
    plt.plot(a)
    plt.show()
