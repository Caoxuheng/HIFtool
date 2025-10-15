import torch
from torch import nn

from .block import BasicStage


class VRcnn(torch.nn.Module):
    def __init__(self, args):
        self.args = args

        super(VRcnn, self).__init__()
        onelayer = []
        self.LayerNo = self.args.layer_num

        for i in range(self.LayerNo):
            onelayer.append(BasicStage(i, args))

        self.fcs = nn.ModuleList(onelayer)

        imsz = self.args.imsz
        self.randpos = torch.rand([1, 6, imsz, imsz]).to('cuda')

    def forward(self, Iryb, Ispec, Phi, Conv, pos):
        ksz = self.args.sf # kernel border
        imsz = self.args.imsz

        '''pos 前高后低'''
        # pos = torch.cat([self.randpos, pos[:,-2:,:,:]], dim=1)
        # pos = self.randpos
        # pos = pos[:,[0,41,84,169,210,253],:,:]
        '''选择维度 消融'''
        if self.args.mis=='high':
            pos = pos[:, :6, :, :] #high
        elif self.args.mis == 'highxy':
            pos = pos[:, [0,1,2,3,-2,-1], :, :] #highxy
        elif self.args.mis =='low':
            pos = pos[:, -8:-2, :, :] #low
        elif self.args.mis =='lowxy':
            pos = pos[:, -6:, :, :] #lowxy
        elif self.args.mis in ['unixy', 'withpe']:
            pos = pos[:,[0,63,128,193,-2,-1],:,:]#unixy withpe
        elif self.args.mis == 'allpe':
            pos = pos #allpe
        #     '''是否pe 消融'''
        # elif self.args.mis =='random':
        #     pos = self.randpos #random
        # elif self.args.mis == 'xyonly':
        #     pos = pos[:,-2:,:,:]  # xyonly
        # elif self.args.mis == 'nope':
        #     pass # nope
        #
        # else:
        #     exit(15)

        y = Iryb
        z = Ispec

        Phiy = torch.matmul(Phi, y.view(3, imsz*imsz)).view(-1, 31, imsz, imsz)
        PhiTPhi = torch.matmul(Phi, Phi.permute(1, 0))

        ConvTConv = torch.matmul(Conv.reshape(ksz**2, 1), Conv.reshape(1, ksz**2))
        convTz = z.reshape(31, imsz//ksz, imsz//ksz, 1, 1).repeat(1, 1, 1, ksz, ksz) \
                 * Conv.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(31, imsz//ksz, imsz//ksz, 1, 1)

        convTz = convTz.permute(0, 1, 3, 2, 4).reshape(-1, 31, imsz, imsz)

        x = Phiy
        v = x

        for i in range(self.LayerNo):
            x, v = self.fcs[i](x, v, Phiy, PhiTPhi, convTz, ConvTConv, pos)


        return v

