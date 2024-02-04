# -*- coding: utf-8 -*-
"""
SpeDnet
"""
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            #print(classname,m,'_______')
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
            elif init_type == 'Gaussian':
                batchsize, channel, height, weight = list(m.weight.data.size())
                t=(channel-1.)/2
                y= np.ogrid[-t:t+1]
                h=np.exp( -(y*y ) / (2.*3.5*3.5) )
                h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
                sumh = h.sum()
                h /= sumh
                m.weight.data[0,:,0,0]=torch.tensor(h,dtype=torch.float32)
                
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize SRF network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer ):
    net.to(device) 
    if initializer :
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net


def Spectral_downsample(args, hsi_channels, msi_channels, sp_matrix, sp_range,  init_type='Gaussian', init_gain=0.02,initializer=False):
    if args.isCal_SRF is False:
        net = matrix_dot_hr2msi(sp_matrix)
        net.to(args.device)
        print('isCal_SRF==No,SRF is known as a prior information')
        return net
    elif args.isCal_SRF is True:
        net = convolution_hr2msi(hsi_channels, msi_channels, sp_range)
        print('isCal_SRF==Yes,adaptively learn SRF')

        return init_net(net,args.device, init_type, init_gain ,initializer)

class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:,1] - self.sp_range[:,0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(x,1,1,1,0,bias=False) for x in self.length_of_each_band])
       

    def forward(self, input):
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:,self.sp_range[i,0]:self.sp_range[i,1]+1,:,:]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list,1)
    def get_srf(self):
        a = self.conv2d_list[0].weight
        srf = a[0,:,0].detach().cpu().numpy()
        return srf
class matrix_dot_hr2msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(matrix_dot_hr2msi, self).__init__()
        self.register_buffer('sp_matrix', torch.tensor(spectral_response_matrix.transpose(1,0)).float())

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(self.sp_matrix.expand(batch,-1,-1),
                         torch.reshape(x,  (batch, channel_hsi, heigth*width))  ).view(batch,channel_msi_sp, heigth, width)
        return hmsi
    
