import torch
from torch import nn

from model_v2.PositionEncoding import *
from model_v2.iternet import VRcnn
from model_v2.tools import device


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


fcnway = True # now it is always true, we use neural representation for degradation parameters

class Fullcnn(torch.nn.Module):
    def __init__(self, args):

        super(Fullcnn, self).__init__()

        petype = 'sin_cos'



        self.args = args
        self.specsrcnn = VRcnn(args)

        '''
        use trainable vars directly
        '''
        if not fcnway:
            A = torch.ones(31, 3) * 5e-1 + torch.rand(31, 3) * 5e-2
            self.Phi = nn.Parameter(A)

            _b = args.ker_sz
            C = torch.rand(_b, _b) * 1e-4
            self.Conv = nn.Parameter(C)

        '''
        use implicit neural field
        '''
        self.input_1D = torch.from_numpy(positionencoding1D(31, 4)).float().to(device)
        self.ssffcn = SSFFcn(4, 3)

        self.input_2D = torch.from_numpy(positionencoding2D(self.args.ker_sz, self.args.ker_sz, 1, petype)).float().to(device).permute(2,0,1).unsqueeze(0)
        self.convfcn = ConvFcn(1, 1)


    def forward(self, Iryb, Ispec, pos):
        ksz = self.args.ker_sz # kernel border
        imsz = self.args.imsz

        if fcnway:
            Phi = self.ssffcn(self.input_1D)#self.Phi*self.Phi
            Phi = Phi **2

            Conv = self.convfcn(self.input_2D) ** 2
            Conv = torch.nn.Softmax(dim=0)(Conv.reshape(ksz**2)).reshape(ksz, ksz)
        else:
            Phi = self.Phi ** 2
            Conv = self.Conv ** 2
            Conv = torch.nn.Softmax(dim=0)(Conv.reshape(ksz ** 2)).reshape(ksz, ksz)

        #torch.nn.Softmax()((self.Conv * self.Conv).reshape(ksz*ksz)).reshape(ksz, ksz)
        #Phi = self.Phi
        #Conv = self.Conv

        spec_cnn = self.specsrcnn(Iryb, Ispec, Phi, Conv, pos)

        pre_ryb_cnn = torch.matmul(Phi.transpose(1, 0), spec_cnn.view(-1, 31, imsz**2)).view(-1, 3, imsz, imsz)

        temp_spec_cnn = spec_cnn.view((31, imsz // ksz, ksz, imsz // ksz, ksz)).permute(0, 1, 3, 2, 4)
        pre_spec_cnn = torch.permute(
            torch.matmul(temp_spec_cnn.reshape(31, imsz // ksz, imsz // ksz, ksz**2), Conv.reshape(ksz**2, 1)),
            [3,0,1,2]
        )



        return spec_cnn, pre_ryb_cnn, pre_spec_cnn, Phi, Conv

