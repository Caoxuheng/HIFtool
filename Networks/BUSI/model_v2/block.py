import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


# Define VRCNN-Net Denoiser
import torch

withposi = True
separate = False

xydim = 6

class ResBlock(torch.nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        num_filter = 64
        self.numfilter = num_filter
        self.conv1 = spectral_norm(torch.nn.Conv2d(31 + xydim if withposi else 31, num_filter, (3, 3), padding=1))
        self.conv2 = spectral_norm(torch.nn.Conv2d(num_filter, num_filter, (3, 3), padding=1))
        self.conv3 = spectral_norm(torch.nn.Conv2d(num_filter, num_filter, (3, 3), padding=1))
        self.conv4 = spectral_norm(torch.nn.Conv2d(num_filter, num_filter, (3, 3), padding=1))
        self.conv5 = spectral_norm(torch.nn.Conv2d(num_filter, 31, (3, 3), padding=1))


    def output(self, input, output):
        return input + output

    def forward(self, x_input):

        y = x_input
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = self.conv5(y)
        result = self.output(x_input[:,0:31,:,:], y)
        return result



class BasicStage(torch.nn.Module):
    def __init__(self, index, args):
        super(BasicStage, self).__init__()

        global xydim, withposi
        if args.mis == 'allpe':
            xydim = args.L * 4 + 2
        elif args.mis == 'xyonly':
            xydim = 2
        elif args.mis == 'nope':
            withposi = False


        self.args = args
        self.index = index

        A = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(A, 1e-1)
        self.lamda = nn.Parameter(A)

        B = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(B, 1e-1)
        self.alpha = nn.Parameter(B)

        C = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(C, -1e5)
        self.beta = nn.Parameter(C)

        self.denoisenet1 = ResBlock()
        self.denoisenet2 = ResBlock()

    def forward(self, x, v, Phiy, PhiTPhi, convTz, ConvTConv, pos):
        ksz = self.args.ker_sz # kernel border
        imsz = self.args.imsz

        #===============================gd===========================
        temp_x = x.view(-1, 31, imsz//ksz, ksz, imsz//ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1, ksz**2)
        convxTx = torch.matmul(temp_x, ConvTConv.reshape(ksz**2,ksz**2))
        convxTx = convxTx.view(-1, 31, imsz//ksz, imsz//ksz, ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1, 31, imsz, imsz)

        PhiTPhix = torch.matmul(PhiTPhi, x.reshape(-1, 31,imsz*imsz))
        PhiTPhix = PhiTPhix.reshape(-1, 31, imsz, imsz)

        g = PhiTPhix - Phiy + self.args.eta*(convxTx - convTz) + F.relu(self.lamda)*(x-v)

        x_next = x - F.relu(self.alpha) * g

        if withposi:

            v_next = self.denoisenet1(torch.cat((x_next, pos), dim=1))
            v_next = self.denoisenet2(torch.cat((v_next, pos), dim=1))
        else:
            v_next = self.denoisenet1(x_next)
            v_next = self.denoisenet2(v_next)

        return x_next, v_next

