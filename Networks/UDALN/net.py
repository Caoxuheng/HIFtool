# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from .Model.spectral_upsample   import Spectral_upsample
from .Model.spectral_downsample import Spectral_downsample
from .Model.spatial_downsample import Spatial_downsample

import matplotlib.pyplot as plt

class udaln():
    def __init__(self,opt,sp_range):
        self.opt = opt
        self.sp_range = sp_range
    def equip(self,sp_matrix,psf):
        self.sp_matrix = sp_matrix
        self.psf=psf
    def stage1(self,lhsi,hmsi):
        lr = self.opt.lr_stage1

        self.optimizer_Spectral_down = torch.optim.Adam(self.Spectral_down_net.parameters(), lr=self.opt.lr_stage1)
        self.optimizer_Spatial_down = torch.optim.Adam(self.Spatial_down_net.parameters(), lr=self.opt.lr_stage1)

        for epoch in range(1, self.opt.epoch_stage1 + 1):

            self.optimizer_Spatial_down.zero_grad()
            self.optimizer_Spectral_down.zero_grad()

            out_lrmsi_fromlrhsi = self.Spectral_down_net(lhsi)  # spectrally degraded from lrhsi
            out_lrmsi_frommsi = self.Spatial_down_net(hmsi)  # spatially degraded from hrmsi
            loss1 = self.L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)
            loss1.backward()

            self.optimizer_Spatial_down.step()
            self.optimizer_Spectral_down.step()

            if epoch % 100 == 0:  # print traning results in the screen every 100 epochs
                print(f'\r{epoch},:{loss1}',end='')
                with torch.no_grad():
                    out_fromlrhsi = out_lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1, 2,
                                                                                            0)  # spectrally degraded from lrhsi
                    out_frommsi = out_lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1, 2,
                                                                                        0)  # spatially degraded from hrmsi

                    # print('estimated PSF:', self.Spatial_down_net.psf.weight.data)




            if (epoch > self.opt.decay_begin_epoch_stage1 - 1):
                each_decay = self.opt.lr_stage1 / (self.opt.epoch_stage1 - self.opt.decay_begin_epoch_stage1 + 1)
                lr = lr - each_decay
                for param_group in self.optimizer_Spectral_down.param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizer_Spatial_down.param_groups:
                    param_group['lr'] = lr

        return  out_lrmsi_frommsi,out_lrmsi_fromlrhsi

    def stage2(self,lhsi,hmsi,out_lrmsi_frommsi_new):
        lr=self.opt.lr_stage2
        for epoch in range(1,self.opt.epoch_stage2+1):
            self.optimizer_Spectral_up.zero_grad()
            lrhsi=self.Spectral_up_net(out_lrmsi_frommsi_new)    #learn SpeUnet, the spectral inverse mapping from low MSI to low HSI

            loss2=self.L1Loss(lrhsi, lhsi)
            loss2.backward()
            self.optimizer_Spectral_up.step()
            if epoch % 100 ==0:  #print traning results in the screen every 100 epochs

                with torch.no_grad():
                    g = self.Spectral_up_net(hmsi)
                print(f'\r{epoch},:{loss2};PSNR:{self.PSNR_GPU(g,self.GT)}',end='')


            if (epoch>self.opt.decay_begin_epoch_stage2-1):
                        each_decay=self.opt.lr_stage2/(self.opt.epoch_stage2-self.opt.decay_begin_epoch_stage2+1)
                        lr = lr-each_decay
                        for param_group in self.optimizer_Spectral_up.param_groups:
                            param_group['lr'] = lr

    def stage3(self,lhsi,hmsi):
        if self.isBlind:
            self.optimizer_Spatial_down.zero_grad()
            self.optimizer_Spectral_down.zero_grad()
            for param_group in self.optimizer_Spectral_down.param_groups:
                param_group['lr'] = self.opt.lr_stage3
            for param_group in self.optimizer_Spatial_down.param_groups:
                param_group['lr'] = self.opt.lr_stage3
        for param_group in self.optimizer_Spectral_up.param_groups:
            param_group['lr'] = self.opt.lr_stage3
        lr = self.opt.lr_stage3

        for epoch in range(1, self.opt.epoch_stage3 + 1):

            self.optimizer_Spectral_up.zero_grad()
            # moduleⅠ 和 Ⅱ
            out_lrmsi_fromlrhsi = self.Spectral_down_net(lhsi)  # spectrally degraded from lrhsi
            out_lrmsi_frommsi = self.Spatial_down_net(hmsi)  # spatially degraded from hrmsi
            loss1 = self.L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)

            # module Ⅲ
            lrhsi = self.Spectral_up_net(
                out_lrmsi_frommsi)  # learn SpeUnet, the spectral inverse mapping from low MSI to low HSI
            loss2 = self.L1Loss(lrhsi, lhsi)

            pre_hhsi = self.Spectral_up_net(
                hmsi)  # use the learned SpeUnet to generate estimated HHSI in the last stage
            pre_msi = self.Spectral_down_net(pre_hhsi)  # spectrally degraded from pre_hhsi
            pre_lrhsi = self.Spatial_down_net(pre_hhsi)  # spatially degraded from pre_hhsi
            loss3 = self.L1Loss(pre_msi, hmsi)
            loss4 = self.L1Loss(pre_lrhsi, lhsi)

            if self.isBlind:
                print('1')
                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                self.optimizer_Spectral_down.step()
                self.optimizer_Spatial_down.step()
                self.optimizer_Spatial_down.zero_grad()
                self.optimizer_Spectral_down.zero_grad()

            else:
                loss=0.5*loss2 + loss3 + loss4
                loss.backward()

            self.optimizer_Spectral_up.step()


            if epoch % 100 == 0:  # print traning results in the screen every 100 epochs
                print(f'\r{epoch},:{loss2};PSNR:{self.PSNR_GPU(pre_hhsi,self.GT)}',end='')

            if (epoch > self.opt.decay_begin_epoch_stage3 - 1):
                each_decay = self.opt.lr_stage3 / (self.opt.epoch_stage3 - self.opt.decay_begin_epoch_stage3 + 1)
                lr = lr - each_decay
                if self.isBlind:
                    for param_group in self.optimizer_Spectral_down.param_groups:
                        param_group['lr'] = lr
                    for param_group in self.optimizer_Spatial_down.param_groups:
                        param_group['lr'] = lr
                for param_group in self.optimizer_Spectral_up.param_groups:
                    param_group['lr'] = lr

        return pre_hhsi.detach().cpu().numpy()[0].T

    def PSNR_GPU(self, im_true, im_fake):
        data_range = 1
        _, C, H, W = im_true.size()
        err = torch.pow(im_true.clone() - im_fake.clone(), 2).mean(dim=(-1, -2), keepdim=True)
        psnr = 10. * torch.log10((data_range ** 2) / err)
        return torch.mean(psnr)
    def __call__(self, lhsi,hmsi,GT):

        hsi_channels=self.opt.hsi_channel
        msi_channels=self.opt.msi_channel

        self.Spectral_down_net = Spectral_downsample(self.opt, self.opt.hsi_channel, self.opt.msi_channel,
                                                     self.sp_matrix, self.sp_range,
                                                     init_type='Gaussian', init_gain=0.02, initializer=True)
        self.Spatial_down_net = Spatial_downsample(self.opt, self.psf, init_type='mean_space', init_gain=0.02,
                                                   initializer=True)

        self.Spectral_up_net   = Spectral_upsample(self.opt,msi_channels,hsi_channels,init_type='normal', init_gain=0.02,initializer=False)
        self.optimizer_Spectral_up=torch.optim.Adam(self.Spectral_up_net.parameters(),lr=self.opt.lr_stage2)

        self.L1Loss = nn.L1Loss(reduction='mean')



        '''
        #begin stage 1
        '''
        ##S1 start
        self.isBlind=self.opt.isCal_SRF or self.opt.isCal_PSF
        if self.isBlind:
          self.stage1(lhsi,hmsi)

        out_lrmsi_frommsi = self.Spatial_down_net(hmsi)  # spatially degraded from hrmsi

        self.GT=GT
        out2 = self.Spectral_down_net(GT)
        plt.imshow(np.hstack([out2[0, 1].detach().cpu().numpy(), hmsi[0,1].detach().cpu().numpy()]))
        plt.show()
        print('\nStage 1 Finished')

        '''
        #begin stage 2
        '''
        out_lrmsi_frommsi_new=out_lrmsi_frommsi.clone().detach()
        self.stage2(lhsi,hmsi,out_lrmsi_frommsi_new)
        print('\nStage 2 Finished')



        '''
        #begin stage 3
        '''
        recon = self.stage3(lhsi,hmsi)
        print('\nStage 3 Finished')

        #
        return recon






