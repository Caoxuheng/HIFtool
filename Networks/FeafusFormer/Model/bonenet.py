

from .common import Denoiser
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
def HSI2MSI(x,srf):
    return (x[0].permute(1,2,0) @srf) .permute(2,0,1).unsqueeze(0)


class FeatureExtractor(nn.Module):

    def __init__(self,band):
        super(FeatureExtractor, self).__init__()
        self.ls = [5, 12, 23, 33, 43]
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        if band ==3:
            self.extractor =nn.ModuleList( [backbone.features[i] for i in range(len(backbone.features))])
        elif band !=3:
            self.extractor = nn.ModuleList([nn.Conv2d(band,64,3,1,1)]+[backbone.features[i] for i in range(1,len(backbone.features))])

    def forward(self,x):
        x = (x-0.5)/0.5
        features = []
        for i in range(self.ls[-1]+1):
            x = self.extractor[i](x)
            if i in self.ls:
                features.append(x)
        return features

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention

        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output = torch.matmul(attn, v)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, in_channel, linear_dim):
        super().__init__()
        # Parameters
        self.n_head = n_head  # No of heads
        self.in_channel= in_channel  # No of pixels in the input image
        self.linear_dim = linear_dim  # Dim of linear-layer (outputs)

        # Linear layers

        self.w_qs = nn.Linear(in_channel, n_head * linear_dim, bias=False)  # Linear layer for queries
        self.w_ks = nn.Linear(in_channel, n_head * linear_dim, bias=False)  # Linear layer for keys
        self.w_vs = nn.Linear(in_channel, n_head * linear_dim, bias=False)  # Linear layer for values
        self.fc = nn.Linear(n_head * linear_dim, in_channel, bias=False)  # Final fully connected layer

        # Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_channel ** 0.5)
        self.bn=Zm_Bn()
        # Batch normalization layer
        # self.OutBN = nn.BatchNorm2d(num_features=in_channel)

    def forward(self, q, k, v, mask=None):
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head = self.n_head
        linear_dim = self.linear_dim
        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h * w).transpose(1,2)
        k = k.view(b, c, h * w).transpose(1,2)
        v = v.view(b, c, h * w).transpose(1,2)
        # Save V
        output = v
        # Separate different heads: b x hw x n x dv
        q = self.w_qs(q).view(b, h*w, n_head, linear_dim)
        k = self.w_ks(k).view(b, h*w, n_head, linear_dim)
        v = self.w_vs(v).view(b, h*w, n_head, linear_dim)
        # Transpose for attention dot product: b x n x hw x dv
        q, k, v = q.transpose(1, 2).transpose(-2, -1), k.transpose(1, 2).transpose(-2, -1), v.transpose(1, 2).transpose(-2, -1)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)
        # Transpose to move the head dimension back: b x hw x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.permute(0,3,1,2).contiguous().view(b, h*w, n_head * linear_dim)
        v_attn = self.fc(v_attn).transpose(-1,-2)
        output = output.transpose(-1,-2)

        # Reshape output to original image format
        output = output.view(b, c, h, w)
        v_attn = v_attn.view(b, c, h, w)

        # output = self.OutBN(output)
        return output+self.bn(v_attn)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class Zm_Bn(nn.Module):
    def __init__(self):
        super(Zm_Bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)

class SpaFM(nn.Module):
    def __init__(self,MSIband,HSIband,SRF,act_fun):
        super(SpaFM, self).__init__()
        self.bn = Zm_Bn()
        self.srf=SRF
        self.refine =nn.Sequential( nn.Conv2d(MSIband,2*MSIband,1) ,act_fun,nn.Conv2d(2*MSIband,HSIband,1) ,act_fun)
    def forward(self,x,MSI):
        # pre_MSI = HSI2MSI(x,self.srf)
        pre_MSI=self.srf(x)
        Error = MSI-pre_MSI
        return F.elu( self.bn(self.refine(Error))+x)

class MultiLevelFus(nn.Module):
    def __init__(self,HSI,MSI,srf):
        super(MultiLevelFus, self).__init__()
        self.T_H,self.T_W = MSI.shape[2:]
        HSIband, MSIband = HSI.shape[1],MSI.shape[1]
        Head = 5
        self.attention = MultiHeadAttention(Head, HSIband,HSIband//5)
        self.attention2 = MultiHeadAttention(Head,HSIband, HSIband//5)
        self.attention3 = MultiHeadAttention(Head,HSIband, HSIband//5)
        ACT ='LeakyReLU'
        act_fun = nn.ELU()
        skip_numchannel = 191
        UP_MODE = 'bilinear'
        sk=20

        self.super = Denoiser( HSIband+MSIband, HSIband,
            num_channels_down=[skip_numchannel,], num_channels_up=[skip_numchannel], num_channels_skip=[sk],n_scales=3
             )
        self.super2 = Denoiser( num_input_channels=HSIband+MSIband, num_output_channels=HSIband,
            num_channels_down=[ skip_numchannel], num_channels_up=[ skip_numchannel], num_channels_skip=[ sk],
            filter_size_down=3, filter_size_up=3, filter_skip_size=1,
            need_sigmoid=False, need_bias=True,
            pad='reflection', upsample_mode=UP_MODE , downsample_mode='stride', act_fun=ACT,
            need1x1_up=True)
        self.super3 = Denoiser( num_input_channels=HSIband+MSIband, num_output_channels=HSIband,
            num_channels_down=[skip_numchannel], num_channels_up=[skip_numchannel], num_channels_skip=[ sk],
            filter_size_down=3, filter_size_up=3, filter_skip_size=1,
            need_sigmoid=True, need_bias=True,
            pad='reflection', upsample_mode=UP_MODE , downsample_mode='stride', act_fun=ACT,
            need1x1_up=True)

        self.FeatureMerge1 =nn.Sequential( nn.Conv2d(256,HSIband,kernel_size=3,padding=1), act_fun, nn.Conv2d(HSIband,HSIband,kernel_size=3,padding=1) )
        self.FeatureMerge2 =nn.Sequential( nn.Conv2d(256, HSIband,kernel_size=1), act_fun, nn.Conv2d(HSIband,HSIband,kernel_size=3,padding=1) )
        self.FeatureMerge3 =nn.Sequential( nn.Conv2d(128, HSIband, kernel_size=1), act_fun, nn.Conv2d(HSIband,HSIband,kernel_size=3,padding=1) )
        self.FeatureMerge4 =nn.Sequential( nn.Conv2d(128, HSIband, kernel_size=1), act_fun, nn.Conv2d(HSIband,HSIband,kernel_size=3,padding=1) )
        self.FeatureMerge5 =nn.Sequential( nn.Conv2d(64,HSIband, kernel_size=1), act_fun, nn.Conv2d(HSIband,HSIband,kernel_size=3,padding=1))
        self.FeatureMerge6 =nn.Sequential( nn.Conv2d(64, HSIband, kernel_size=1), act_fun, nn.Conv2d(HSIband,HSIband,kernel_size=3,padding=1) )
        self.HSIFE = FeatureExtractor(HSIband)
        self.MSIFE = FeatureExtractor(MSIband)
        self.bn = Zm_Bn()

        self.MSI = MSI

        self.coarse_HSI = F.interpolate(HSI, [self.T_H, self.T_W],mode='bilinear')
        self.HSI_c = F.interpolate(HSI, [self.T_H//8,self.T_W//8])
        self.MSI_c1 = F.interpolate(MSI,[self.T_H//8,self.T_W//8])
        self.MSI_c2 = F.interpolate(MSI, [self.T_H//2, self.T_W//2])
        self.rm = SpaFM(MSIband,HSIband,srf,act_fun,)

    def forward(self):
        hsi_feature = self.HSIFE(self.coarse_HSI)
        msi_feature = self.MSIFE(self.MSI)

        HSI_c = self.super(torch.concat([self.HSI_c,self.MSI_c1],dim=1))
        feature =self.FeatureMerge1(  hsi_feature[2])
        feature2 = self.FeatureMerge2(msi_feature[2])
        # print(feature.shape,feature2.shape,HSI_c.shape)
        HSI_c = self.attention(feature2,feature,HSI_c)
        HSI_c = self.rm(HSI_c,self.MSI_c1)

        HSI_c = F.interpolate(HSI_c, [self.T_H//2, self.T_W//2])
        HSI_c = self.super2(torch.concat([HSI_c,  self.MSI_c2 ],dim=1))
        feature = self.FeatureMerge3(hsi_feature[1])
        feature2 = self.FeatureMerge4(msi_feature[1])
        HSI_c = self.attention2(feature2, feature, HSI_c)
        HSI_c = self.rm(HSI_c, self.MSI_c2)

        HSI_c = F.interpolate(HSI_c, [self.T_H,self.T_W])
        HSI_c = self.super3(torch.concat([HSI_c, self.MSI],dim=1))
        feature = self.FeatureMerge5(hsi_feature[0])
        feature2 = self.FeatureMerge6(msi_feature[0])
        HSI_c = self.attention3(feature2, feature, HSI_c)
        HSI_c = self.rm(HSI_c, self.MSI)

        return HSI_c
