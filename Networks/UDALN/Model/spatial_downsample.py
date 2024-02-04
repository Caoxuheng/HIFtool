# -*- coding: utf-8 -*-
"""
SpaDnet
"""
import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize PSF network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer ):
    print('spatial_downsample1')
    net.to(device)  
    if initializer :
        init_weights(net,init_type, init_gain)
    else:
        print('Spatial_downsample with default initialize')
    return net

def Spatial_downsample(args,psf, init_type='mean_space', init_gain=0.02,initializer=False):
    if args.isCal_PSF is False:
        net = matrix_dot_hr2lr(psf,args.sf)
        net.to(args.device)
        print('isCal_PSF==No,PSF is known as a prior information')
        return net
    elif args.isCal_PSF is True:
        net = PSF(scale=args.sf)
        print('isCal_PSF==Yes,adaptively learn PSF')
        return init_net(net,args.device, init_type, init_gain ,initializer)

class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.psf = nn.Conv2d(1, 1, scale, scale, 0, bias=False) #in_channels, out_channels, kernel_size, stride, padding

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.psf(x[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)

class matrix_dot_hr2lr(nn.Module):
    def __init__(self, PSF,sf):
        super(matrix_dot_hr2lr, self).__init__()
        
        self.register_buffer('PSF', torch.tensor(PSF).float())
        pad= self.PSF.shape[0]-sf if self.PSF.shape[0]>sf else 0
        self.psf = nn.Conv2d(1, 1, self.PSF.shape[0], sf, pad,padding_mode='circular', bias=False)
        self.psf.weight.data[0,0]=self.PSF
        self.psf.requires_grad_(False)

    def __call__(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.psf(x[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)

