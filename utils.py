import numpy as np
import torch
from torch import nn
import os

# Network Tool
def reshuffle(img,patch_size):
    b,c,h,w = img.shape
    n = int(np.sqrt(b))
    recon = np.empty([n*patch_size,n*patch_size,c])
    for i in range(n):
        for j in range(n):
            recon[i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1)  * patch_size] = img[j*n+i].T.detach().cpu().numpy()
    return recon

def save_checkpoint(model_folder, model, optimizer, lr, epoch):  # save model function

    model_out_path = model_folder + "{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr":lr
    }
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder,exist_ok=True)
    torch.save(checkpoint, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))

#  Evaluation Metrics - GPU Version
def PSNR_GPU(im_true, im_fake):
    data_range = 1
    _,C,H,W = im_true.size()
    err = torch.pow(im_true.clone()-im_fake.clone(),2).mean(dim=(-1,-2), keepdim=True)
    psnr = 10. * torch.log10((data_range**2)/err)
    return torch.mean(psnr)

def SSIM_GPU(r_img,f_img,k1=0.01, k2=0.03):
    l = 1
    x1_ = r_img.reshape(r_img.size(1),-1)
    x2_ = f_img.reshape(f_img.size(1),-1)
    u1 = x1_.mean(dim=-1,keepdim=True)
    u2 = x1_.mean(dim=-1,keepdim=True)
    Sig1 = torch.std(x1_, dim=-1,keepdim=True)
    Sig2 = torch.std(x2_, dim=-1,keepdim=True)
    sig12 = torch.sum((x1_ - u1) * (x2_ - u2), dim=-1) / (x1_.size(-1) - 1)
    c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
    return SSIM.mean()

# Degradation part
def fspecial(kernel_type, kernel_size, sigma=None):
    from scipy import signal
    if kernel_type == 'gaussian':
        kernel = signal.windows.gaussian(kernel_size, sigma)
    elif kernel_type == 'average':
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    elif kernel_type == 'laplacian':
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return kernel
def getGaussiankernel(ksize,sigma):
    x = torch.arange( -(ksize-1)//2,(ksize-1)//2+1)
    kernel = torch.exp(-1/2*torch.pow(x/sigma,2))
    kernel/=kernel.sum()
    return kernel.view(-1,1).float()

class Spatial_Degradation(nn.Module):
    def __init__(self,sf,predefine=None,batch=1):
        super(Spatial_Degradation, self).__init__()

        if predefine is not False:
            kernel = torch.tensor(predefine).float()
            stride = sf
            sf = kernel.shape[0]
        else:
            kernel = getGaussiankernel(sf,sf*0.866)
            kernel @=kernel.T
            stride = sf
        if batch ==1 :

            self.down = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=sf,stride=stride,bias=None)
            self.down.weight.data[0,0] = kernel
        self.down.requires_grad_(False)

    def forward(self,x):
        return self.down(x.transpose(1,0)).transpose(1,0)

class SpaDown(nn.Module):
    def __init__(self,sf,predefine=False):
        super(SpaDown, self).__init__()
        self.net = Spatial_Degradation(sf,predefine)
    def forward(self,x):
        return self.net(x)

def SpeDown(x,srf):
    return (x[0].permute(1,2,0) @srf) .permute(2,0,1).unsqueeze(0)

def generate_lrhsi(gt,net):
    return net(torch.tensor(gt.T).unsqueeze(0).float())

def generate_hrrgb(gt,srf):
    return gt @ srf

def get_pairs(gt,net,srf,noise=None,NSR_HSI=30,NSR_RGB=40):
    lrhsi = generate_lrhsi(gt,net)[0]
    hrrgb = torch.tensor(generate_hrrgb(gt,srf).T).float()


    if noise:
        lrhsi = add_noise(lrhsi,NSR_HSI)
        hrrgb = add_noise(hrrgb,NSR_RGB)
    return lrhsi,hrrgb


def getInputImgs(GT,dataset_name,spadown,srf,noise=False):

    lr_hsi, hr_rgb = get_pairs(GT, spadown, srf, noise=noise)
    return lr_hsi.unsqueeze(0),hr_rgb.unsqueeze(0),GT
