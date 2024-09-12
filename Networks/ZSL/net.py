
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
from model1 import *


PSF = fspecial('gaussian', 7, 3)
p=10


LR=1e-3

loss_optimal=1.75

import imgvision
import numpy as np
from .Model.cnn import ZSL_cnn
from .utils import  *
class ZSL():
    def __init__(self,args):
        self.args = args
        self.step=0
        self.model = ZSL_cnn(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.equip_degradation(args)
        
    def degradation_est(self,lhsi,hmsi,size_B):
        B,L,M,N = hmsi.shape
        R = np.ones(self.args.msi_channel,self.args.hsi_channel);
        R /= self.args.hsi_channel

        HSI_2D = lhsi.reshape([-1,self.args.hsi_channel])
        s0 = 1
        iter = 20
        B = np.abs(np.random.randn(size_B, size_B));

        mu = 0
        B /= B.sum()
        for i in range(iter):
            fft_B = psf2otf(B, [M N]);

        MSI_BS = Gaussian_downsample(hmsi, fft_B, self.args.sf, s0)
        MSI_BS = MSI_BS.reshape([-1,self.args.msi_channel])
        R = R_update2(MSI_BS, HSI_2D, R, mu)

        R_HSI = R * HSI_2D
        R_HSI =R_HSI.reshape(lhsi.shape)
        R_HSI_up = zeros(M, N, L)
        R_HSI_up[s0:: self.args.sf, s0::self.args.sf]=R_HSI
        B = B_update2(R_HSI_up, hmsi, size_B, self.args.sf, B, mu)
        return R,B


    def equip_degradation(self,args,lhsi,hmsi,size_):
        self.degradation_est(lhsi,hmsi,size_B)
        self.spadown =
        self.spedown = 


    def get_traindata(self,lhsi,hmsi,U):
        augument = [0]
        HSI_aug = []
        MSI_aug = []
        HSI_aug.append(lhsi)
        MSI_aug.append(hmsi)
        train_hrhs,train_hrms,train_lrhs=[],[],[]

        for j in augument:
            HSI = cv2.flip(lhsi, j)
            HSI_aug.append(HSI)
        for j in range(len(HSI_aug)):
            HSI = HSI_aug[j]
            HSI_Abun = np.tensordot(U.T, HSI, axes=([1], [0]))
            HSI_LR_Abun = self.spadown(HSI_Abun)
            MSI_LR = self.spedown(HSI)

            for j in range(0, HSI_Abun.shape[1] - self.args.args.training_size + 1, 1):
                for k in range(0, HSI_Abun.shape[2] - self.args.training_size + 1, 1):
                    temp_hrhs = HSI[:, j:j + self.args.training_size, k:k + self.args.training_size]
                    temp_hrms = MSI_LR[:, j:j + self.args.training_size, k:k + self.args.training_size]
                    temp_lrhs = HSI_LR_Abun[:, int(j / self.args.sf):int((j + self.args.training_size) / self.args.sf),
                                int(k / self.args.sf):int((k + self.args.training_size) / self.args.sf)]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)

            train_hrhs = torch.Tensor(train_hrhs).cuda()
            train_lrhs = torch.Tensor(train_lrhs).cuda()
            train_hrms = torch.Tensor(train_hrms).cuda()
            train_data = HSI_MSI_Data(train_hrhs, train_hrms, train_lrhs)
            self.train_loader = data.DataLoader(dataset=train_data, batch_size=self.args.BATCH_SIZE, shuffle=True)


    def __call__(self, lhsi, hmsi, GT):

        lhsi_np = lhsi[0].detach().cpu().numpy()
        U0, S, V = np.linalg.svd(np.dot(lhsi_np, lhsi_np.T))
        U0 = U0[:, 0:int(p)]
        U = U0
        U22 = torch.Tensor(U0)
        L1loss = nn.L1Loss()

        maxiteration = 2 * math.ceil(((GT.shape[1] / self.args.sf - self.args.training_size) // 1 + 1) * (
                    (GT.shape[2] / self.args.sf - self.args.training_size) // 1 + 1) / self.args.BATCH_SIZE) * 400

        warm_iter = np.floor(maxiteration / 40)
        for step in range(400):
            for step1, (lhsi_data, hmsi_data, a3) in enumerate(self.train_loader):
                lr = warm_lr_scheduler(self.optimizer, self.args.init_lr1, self.args.init_lr2, warm_iter, step, lr_decay_iter=1, max_iter=maxiteration,
                                   power= self.args.decay_power)

                output, Xre = self.model(lhsi_data, hmsi_data,U22)
                loss = L1loss(self.spadown(Xre),lhsi_data)+L1loss(self.spedown(Xre),hmsi_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return  Xre[0].detach().cpu().numpy().T

cnn=CNN(args).cuda()

for epoch in range(EPOCH): 
    for step1, (a1, a2,a3) in enumerate(train_loader): 
        cnn.train()
        lr=warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,step, lr_decay_iter=1,  max_iter=maxiteration, power=decay_power)
        step=step+1
        output,Xre= cnn(a3.cuda(),a2.cuda())

        loss = loss_func(Xre, a1.cuda(),a2.cuda(),downsample_factor)
        optimizer.zero_grad()           
        loss.backward()               
        optimizer.step()
    cnn.eval()
    with torch.no_grad():
        abudance = cnn(HSI_1.cuda(),MSI_1.cuda())
        abudance=abudance.cpu().detach().numpy()
        abudance1=np.squeeze(abudance)
    Fuse2=np.tensordot(U0, abudance1, axes=([1], [0]))
    sum_loss,psnr_=metrics.rmse1(np.clip(Fuse2,0,1),HRHSI)
    if sum_loss<loss_optimal:
       loss_optimal=sum_loss
    loss_list.append(sum_loss)
    print(epoch,lr,sum_loss,psnr_)
    torch.save(cnn.state_dict(), data2+'1.pkl',_use_new_zipfile_serialization=False) 

