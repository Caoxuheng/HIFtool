
import torch.utils.data as data
import torch
import numbers
import imgvision as iv
import numpy as np
import scipy.io as sio
from torch import nn
from pathlib import Path


#  Spatial Degradation
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
            kernel = getGaussiankernel(sf,sf//1.5)
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
        lrhsi = iv.add_noise(lrhsi,NSR_HSI)
        hrrgb = iv.add_noise(hrrgb,NSR_RGB)
    return lrhsi,hrrgb

def getInputImgs(args,name,index,spadown,SRF,noise=False):
    dataset = Dataloader(name,args)
    GT = dataset.load(index)
    lr_hsi, hr_rgb = get_pairs(GT, spadown, SRF, noise=noise)
    return lr_hsi.unsqueeze(0),hr_rgb.unsqueeze(0),GT

class Dataloader():
    def __init__(self,name:str,opt):
        self.name = name
        self.opt = opt
        self.dataset_root = Path(getattr(opt, 'dataset_root', 'D:/CaoXuheng/dataset'))
        self.harvard_files = sorted((self.dataset_root / 'HARVARD').rglob('*.mat'))
        if name.upper() == 'HARVARD' and not self.harvard_files:
            raise FileNotFoundError(f'No Harvard MAT files found in {self.dataset_root / "HARVARD"}')

    def load(self,index):
        if self.name.upper() =='CAVE':
            path = self.dataset_root / 'CAVE' / f'{index}.mat'
            if not path.is_file():
                raise FileNotFoundError(f'CAVE sample not found: {path}')
            base = sio.loadmat(path)
            key = list(base.keys())[-1]
            Ground_Truth = base[key]

        elif self.name.upper()=='HARVARD':
            base = sio.loadmat(self.harvard_files[index])
            Ground_Truth = base['ref']
            Ground_Truth = Ground_Truth / Ground_Truth.max()
            Ground_Truth = Ground_Truth[:1024, :1024]

        return Ground_Truth

class RandomCrop(object):

    def __init__(self, size,  pad_if_needed=False, fill=0, factor=0,padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.factor = factor

    @staticmethod
    def get_params(img, output_size,index_):
        h, w , _= img.shape
        th, tw = output_size
        h_ = h//th
        w_ = w //tw
        i = index_//w_
        j = index_%h_
        return i*th, j*tw,  th, tw


    def __call__(self, clean, noisy,lr,index_):

        h, w , _ = clean.shape
        sf = self.factor
        i,  j, h, w = self.get_params(clean, self.size,index_)
        # print(index_,i,j)
        return (clean[i : i + h,  j : j+w, :], noisy[i : i + h ,  j : j+w, :], lr[i//sf : i//sf + h//sf ,  j//sf : j//sf+w//sf, :])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class Large_dataset(data.Dataset):

    def __init__(self, opt,patch_size,name='CAVE',type='train', crop=True, lazy=False):  # , root, fn, crop=True, loader=default_loader):
        SRF = np.load('./Dataloader_tool/srflib/NikonD700.npy')
        self.HR_list = []
        self.RGB_list = []
        self.LR_list = []
        Spatialdown = SpaDown(opt.sf, predefine=False)
        if name.upper() =='CAVE':
            self.n = 512 // patch_size
            split_protocol = getattr(opt, 'cave_split', 'official')
            if split_protocol == 'sequential_60_40':
                # First 18 CAVE samples for supervision; last 13 are held out.
                train_set = list(range(18))
                val_list = train_set
                test_list = list(range(18, 31))
            elif split_protocol == 'official':
                test_list = [0,13,7,25,15,18,28,1,20,6]
                val_list = [4,19,30]
                train_set = list(set(range(31))-set(test_list)-set(val_list))
            else:
                raise ValueError(f'Unsupported CAVE split protocol: {split_protocol}')
            self.test_name = test_list if type == 'test' else (train_set if type == 'train' else val_list)
            if type=='train':
                data_set = train_set
            elif type =='eval':
                data_set =val_list
            elif type =='test':
                data_set = test_list
        elif name.upper() == 'HARVARD':
            if patch_size > 1024 or 1024 % patch_size:
                raise ValueError('HARVARD patch_size must be a positive divisor of 1024.')
            self.n = 1024 // patch_size
            test_list = [74, 62, 71, 70, 57, 66, 75, 2, 35, 34, 23, 59, 29,
                         65, 28, 8, 43, 63, 42, 19, 13, 33, 20, 36, 9]
            loader = Dataloader(name, opt)
            if max(test_list) >= len(loader.harvard_files):
                raise ValueError('The Harvard dataset does not contain the required test samples.')
            val_list = test_list[:5]
            train_set = list(set(range(len(loader.harvard_files))) - set(test_list))
            if type == 'train':
                data_set = train_set
            elif type == 'eval':
                data_set = val_list
            elif type == 'test':
                data_set = test_list
            else:
                raise ValueError(f'Unsupported dataset split: {type}')
            self.test_name = [loader.harvard_files[i].stem for i in data_set]
        else:
            raise ValueError(f'Unsupported dataset: {name}')

        self.lazy = lazy
        self.sample_ids = data_set
        self.opt = opt
        self.name = name
        self.spatial_down = Spatialdown
        self.srf = SRF
        if not lazy:
            for i in data_set:
                HSI, MSI, GT = getInputImgs(opt,name, i, Spatialdown,SRF)
                self.HR_list.append(GT)
                self.RGB_list.append(MSI[0].permute([2,1,0]))
                self.LR_list.append(HSI[0].permute([2,1,0]))
        self.count = len(data_set)
        self.channel = opt.msi_channel
        self.crop = crop
        self.factor = opt.sf
        if crop:
            self.crop_LR = RandomCrop(patch_size,factor=opt.sf)

        else:
            self.crop_LR = False

    def __getitem__(self, index_):

        index = index_ // (self.n**2)
        if self.lazy:
            HSI, MSI, HR = getInputImgs(
                self.opt, self.name, self.sample_ids[index], self.spatial_down, self.srf
            )
            RGB = MSI[0].permute([2, 1, 0])
            LR = HSI[0].permute([2, 1, 0])
        else:
            HR = self.HR_list[index]
            RGB = self.RGB_list[index]
            LR = self.LR_list[index]

        if self.crop:

            HR, RGB, LR = self.crop_LR(HR, RGB, LR,index_%(self.n**2))

        # -----------------------------------------------------

        LR = np.transpose(LR, [2, 0, 1])
        HR = np.transpose(HR, [2, 0, 1])
        RGB = np.transpose(RGB, [2, 0, 1])

        HR = torch.FloatTensor(HR)
        RGB = torch.FloatTensor(RGB)
        LR = torch.FloatTensor(LR)
        return (HR, LR, RGB)

    def __len__(self):
        return self.count*(self.n**2)

