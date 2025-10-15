import torch
from sympy.codegen import Print
from torch import nn

from .model_v2.PositionEncoding import *
from .model_v2.iternet import VRcnn
from .model_v2.tools import device


class SSFFcn(torch.nn.Module):
    def __init__(self, L, out_dim):
        super(SSFFcn, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * L + 1, out_features=2 * L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=out_dim)#,
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.layers(x)

        return y


class ConvFcn(torch.nn.Module):
    def __init__(self, L, out_dim):

        super(ConvFcn, self).__init__()

        self.layers = nn.Sequential(
            torch.nn.Conv2d(4 * L + 2, 4 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(4 * L, 2 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(2 * L, 1 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(1 * L, 2, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(2, out_dim, (1, 1), padding=0)#,
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.layers(x)

        return y



class Fullcnn(torch.nn.Module):
    def __init__(self, args):

        super(Fullcnn, self).__init__()

        petype = 'sin_cos'



        self.args = args
        self.specsrcnn = VRcnn(args)

        '''
        use trainable vars directly
        '''


        '''
        use implicit neural field
        '''
        self.input_1D = torch.from_numpy(positionencoding1D(31, 4)).float().to(device)
        self.ssffcn = SSFFcn(4, 3)

        self.input_2D = torch.from_numpy(positionencoding2D(self.args.ker_sz, self.args.ker_sz, 1, petype)).float().to(device).permute(2,0,1).unsqueeze(0)
        self.convfcn = ConvFcn(1, 1)

    def equip(self,SRF,PSF):

            self.Phi = nn.Parameter(torch.FloatTensor(SRF).cuda())
            self.Conv = PSF

    def forward(self, Iryb, Ispec, pos):
        ksz = self.args.ker_sz # kernel border
        imsz = self.args.imsz

        if self.args.isCal_PSF:
            Conv = self.convfcn(self.input_2D) ** 2
            self.Conv = torch.nn.Softmax(dim=0)(Conv.reshape(ksz**2)).reshape(ksz, ksz)
        if self.args.isCal_SRF:
            Phi = self.ssffcn(self.input_1D)  # self.Phi*self.Phi
            self.Phi = Phi ** 2
        Phi = self.Phi
        Conv = self.Conv

        spec_cnn = self.specsrcnn(Iryb, Ispec, Phi, Conv, pos)

        pre_ryb_cnn = torch.matmul(Phi.transpose(1, 0), spec_cnn.view(-1, 31, imsz**2)).view(-1, 3, imsz, imsz)

        temp_spec_cnn = spec_cnn.view((31, imsz // ksz, ksz, imsz // ksz, ksz)).permute(0, 1, 3, 2, 4)
        pre_spec_cnn = torch.permute(
            torch.matmul(temp_spec_cnn.reshape(31, imsz // ksz, imsz // ksz, ksz**2), Conv.reshape(ksz**2, 1)),
            [3,0,1,2]
        )



        return spec_cnn, pre_ryb_cnn, pre_spec_cnn, Phi, Conv

class BUSI():
    def __init__(self, args):
        self.args = args
        self.busi = Fullcnn(args).cuda()
        self.optimizer = torch.optim.Adam(self.busi.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()

    def set_input(self,HRMSI):
        _,c, h, w = HRMSI.shape

        pos =  positionalencoding2d(self.args.L * 4, h, w)
        return pos
    def equip(self,srf,psf):
        self.busi.equip(srf,psf)

    def __call__(self, LRHSI,HRMSI,GT=None):
        pos = self.set_input(HRMSI).cuda()
        if self.args.ker_sz <= 4:
            tv_wei = 1.5
            lowrank_wei = 0.7
        else:
            # for real image
            tv_wei = 1.8
            lowrank_wei = 0.02

        for epoch in range(2000):

            resspec_cnn, ryb_pred_cnn, spec_pred_cnn, phi, kernel = self.busi(HRMSI, LRHSI,pos[None])

            ryb_loss_cnn =self. criterion(ryb_pred_cnn, HRMSI) * self.args.rgb_wei
            spec_loss_cnn = self.criterion(spec_pred_cnn, LRHSI)
            # tv_loss = 1 * 5e-6 * tv_wei * torch.sum(torch.abs(phi[1:, :] - phi[:-1, :]))  # phi is SSF
            # U, S, Vh = torch.linalg.svd(kernel, full_matrices=True)
            # lr_loss = 1 * 1e-4 * lowrank_wei * torch.sum(S)
            # tv_img_loss_cnn = 5e-7 * torch.sum(torch.abs(resspec_cnn[:, 1:, :, :] - resspec_cnn[:, :-1, :, :]))

            # loss_all = ryb_loss_cnn / ryb_loss_cnn.detach() + spec_loss_cnn / spec_loss_cnn.detach()
            loss_all = ryb_loss_cnn + spec_loss_cnn

            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

        return  resspec_cnn