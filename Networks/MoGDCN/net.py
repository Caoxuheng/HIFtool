# coding=UTF-8

import torch
import torch.nn.functional as F
import numpy as np
import os
# import matplotlib.pyplot as plt
# from clean_util import H_z, HT_y, para_setting

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""
modify fast DVD(vedio denoising) 
"""


class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):

        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        down = self.act(self.conv5(f_e))
        return f_e, down


class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in=64):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e


class Decoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)
        # self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        self.batch = 1
        # self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = torch.nn.ConvTranspose2d(c_in, 128, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input, label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])
        Deconv = self.up(input)

        return Deconv

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input, label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])

        Deconv = self.up(input)

        return Deconv

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class Unet_Spatial(torch.nn.Module):
    def __init__(self, cin):
        super(Unet_Spatial, self).__init__()

        self.Encoding_block1 = Encoding_Block(64)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)
        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(128)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        sz = x.shape
        # x = x.view(-1,1,sz[2],sz[3])

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)
        # y = x[:,1:2,:,:] + decode0
        # y = x + decode0

        return decode0, encode0


class Unet_Spectral(torch.nn.Module):
    def __init__(self, cin):
        super(Unet_Spectral, self).__init__()

        self.Encoding_block1 = Encoding_Block(cin)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)
        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(128)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        sz = x.shape
        x = x.view(-1, 31, sz[2], sz[3])

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)
        y = x + decode0
        return y


class VSR_CAS(torch.nn.Module):
    """
    network of 'Burst Denoising with Kernel Prediction Networks'
    """

    def __init__(self, args, P=None):
        super(VSR_CAS, self).__init__()

        self.channel0 =args.hsi_channel
        self.up_factor = args.sf
        self.patch_size = args.patch_size

        # self.P = torch.nn.Parameter(P)
        # self.P.requires_grad = False
        self.acti = torch.nn.PReLU()

        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_1 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_1 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_2 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_2 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_3 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_4 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_4 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_5 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_5 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_6 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_6 = torch.nn.Parameter(torch.tensor(0.9))
        self.spatial = Unet_Spatial(args.hsi_channel)
        # self.spatial1 = Unet_Spatial(3)  # if no use then comment it out
        self.fe_conv1 = torch.nn.Conv2d(in_channels=args.hsi_channel, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv3 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv4 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)

        self.fe_conv5 = torch.nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv6 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv7 = torch.nn.Conv2d(in_channels=448, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv8 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.conv_downsample = torch.nn.Conv2d(in_channels=31, out_channels=31, kernel_size=13,stride=8, padding=13 // 2)
        # self.conv_upsample   = torch.nn.ConvTranspose2d(in_channels=31, out_channels=31, kernel_size=13,stride=8, padding=13 // 2 )
        self.conv_downsample = torch.nn.Upsample(scale_factor=1 / args.sf)

        self.conv_upsample = torch.nn.Upsample(scale_factor=args.sf)
        self.conv_torgb = torch.nn.Conv2d(in_channels=args.hsi_channel, out_channels=args.msi_channel, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tohsi = torch.nn.Conv2d(in_channels=args.msi_channel, out_channels=args.hsi_channel, kernel_size=3, stride=1, padding=3 // 2)
        self.reset_parameters()


    def recon_noisy(self, z, noisy, v, RGB, id_layer):
        if id_layer == 0:
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1:
            DELTA = self.delta_1
            ETA = self.eta_1
        elif id_layer == 2:
            DELTA = self.delta_2
            ETA = self.eta_2
        elif id_layer == 3:
            DELTA = self.delta_3
            ETA = self.eta_3
        elif id_layer == 4:
            DELTA = self.delta_4
            ETA = self.eta_4
        elif id_layer == 5:
            DELTA = self.delta_5
            ETA = self.eta_5

        sz = z.shape
        # err1 = RGB.reshape(sz[0], 3, sz[2] * sz[3]) - torch.matmul( self.P.transpose(0,1).unsqueeze(0) ,z.reshape(sz[0], sz[1], sz[2] * sz[3]))
        err1 = RGB - self.conv_torgb(z)
        # err1 = torch.matmul(self.P.unsqueeze(0), err1)
        err1 = self.conv_tohsi(err1)
        # err1 = err1.reshape(sz)
        err2 = noisy - ETA * v
        err2 = err2.reshape(sz)

        out = (1 - DELTA - DELTA * ETA) * z + DELTA * err1 + DELTA * err2
        return out

    def recon(self, features, recon, LR, RGB, id_layer):
        if id_layer == 0:
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1:
            DELTA = self.delta_1
            ETA = self.eta_1
        elif id_layer == 2:
            DELTA = self.delta_2
            ETA = self.eta_2
        elif id_layer == 3:
            DELTA = self.delta_3
            ETA = self.eta_3
        elif id_layer == 4:
            DELTA = self.delta_4
            ETA = self.eta_4
        elif id_layer == 5:
            DELTA = self.delta_5
            ETA = self.eta_5

        # fft_B, fft_BT = para_setting('gaussian_blur', self.up_factor, [self.patch_size, self.patch_size])
        # fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2),
        #                         torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2).cuda()
        # fft_BT = torch.cat(
        #     (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2) .cuda()
        #
        # recon_h1 = int(recon.shape[2])
        # recon_h2 = int(recon.shape[3])

        # down = self.Down(recon, self.up_factor , fft_B )
        sz = recon.shape
        down = self.conv_downsample(recon)
        # err1 = self.UP(down - LR , self.up_factor ,fft_BT)
        err1 = self.conv_upsample(down - LR)

        # to_rgb = torch.matmul(self.P.transpose(0,1).unsqueeze(0)   , recon.reshape(sz[0], sz[1], sz[2] * sz[3]))
        to_rgb = self.conv_torgb(recon)
        # err_rgb = RGB - to_rgb.reshape(sz[0], 3, sz[2] , sz[3])
        err_rgb = RGB - to_rgb
        # err3    = torch.matmul(self.P.unsqueeze(0),err_rgb.reshape(sz[0], 3, sz[2] * sz[3]))
        err3 = self.conv_tohsi(err_rgb)
        err3 = err3.reshape(sz)
        ################################################################

        out = (1 - DELTA * ETA) * recon + DELTA * err3 + DELTA * err1 + DELTA * ETA * features
        ################################################################
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, LR, RGB):  # [batch_size ,3 ,7 ,270 ,480] ;
        ## LR [1 31 6 6]
        ## RGB [1 31 48 48]

        label_h1 = int(LR.shape[2]) * self.up_factor
        label_h2 = int(LR.shape[3]) * self.up_factor

        # x = bicubic_interp_2d(input, [label_h1, label_h2])
        x = torch.nn.functional.interpolate(LR, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        # _, fft_BT = para_setting('uniform_blur', self.up_factor, [self.patch_size, self.patch_size])
        # fft_BT = torch.cat(
        #    (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2).cuda()
        # x = self.UP( LR , self.up_factor ,fft_BT)
        y = LR

        # for i in range(0, 3):
        z = x
        v, fe = self.spatial(self.fe_conv1(z))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe1 = self.spatial(self.fe_conv2(torch.cat((self.fe_conv1(z), fe), 1)))
        conv_out = conv_out + z

        x = self.recon(conv_out, x, y, RGB, id_layer=3)

        z = x
        v, fe2 = self.spatial(self.fe_conv3(torch.cat((self.fe_conv1(z), fe), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe3 = self.spatial(self.fe_conv4(torch.cat((self.fe_conv1(z), fe2), 1)))
        conv_out = conv_out + z
        x = self.recon(conv_out, x, y, RGB, id_layer=3)

        z = x
        v, fe4 = self.spatial(self.fe_conv5(torch.cat((self.fe_conv1(z), fe, fe2), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe5 = self.spatial(self.fe_conv6(torch.cat((self.fe_conv1(z), fe4), 1)))
        conv_out = conv_out + z
        x = self.recon(conv_out, x, y, RGB, id_layer=3)

        z = x
        v, fe6 = self.spatial(self.fe_conv7(torch.cat((self.fe_conv1(z), fe, fe2, fe4), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, _ = self.spatial(self.fe_conv8(torch.cat((self.fe_conv1(z), fe6), 1)))
        conv_out = conv_out + z

        return conv_out

