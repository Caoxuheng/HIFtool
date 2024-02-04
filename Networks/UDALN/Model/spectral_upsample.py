# -*- coding: utf-8 -*-
"""
SpeUnet
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np

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
    
    print('Spectral_upsample initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer ):
    print('spectral_upsample1')
    net.to(device)  
    if initializer :
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net


def Spectral_upsample(args,msi_channels,hsi_channels,init_type='kaiming', init_gain=0.02,initializer=False ):

    net = spectral_upsample(msi_channels,hsi_channels)
    
    return init_net(net, args.device, init_type, init_gain ,initializer)

class spectral_upsample(nn.Module):
    def __init__(self,msi_channels,hsi_channels,need_clamp=False):
        super(spectral_upsample, self).__init__()
        self.layers=[]
        self.need_clamp=need_clamp
        self.num_ups=int(np.log2(hsi_channels/msi_channels))
        
        for i in range(1,self.num_ups+1):
            self.layers += [ nn.Conv2d(msi_channels*(2**(i-1)), msi_channels*(2**i), kernel_size=1, stride=1, padding=0),
                               nn.LeakyReLU(0.2, True)]
        
        
        self.layers+= [nn.Conv2d(msi_channels*(2**self.num_ups), hsi_channels, kernel_size=1, stride=1, padding=0)]
        self.sequential=nn.Sequential(*self.layers)
    
    def forward(self,input):
        x=input
        if self.need_clamp:
            x=self.sequential(x)
            return x.clamp_(0,1)
        else:
            x=self.sequential(x)
            return nn.LeakyReLU(0.2, True)(x)
   
        
        
