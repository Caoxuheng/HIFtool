from torch.autograd import Variable
from .downsampler import *
import numpy as np
from .model_1 import *
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter



# The dataset of spatial downsampled HR MSI to HR MSI
class RGB_sets(data.Dataset):
    def __init__(self, RGB, Label, P_N, P_S, sf):
        self.RGB = RGB
        self.size = RGB.shape
        self.P_S = P_S
        self.Label = Label
        self.P_N = P_N
        self.D_spa = Downsampler(n_planes=RGB.shape[0], factor=sf, kernel_type='gauss12', phase=0, preserve_size=True).type(  torch.cuda.FloatTensor)

    def __getitem__(self, index):
        [p, q] = self.Label[index, :]
        P_S = self.P_S
        RGB = self.RGB[:, p:p + P_S, q:q + P_S]
        D_RGB = self.D_spa(RGB)

        return torch.squeeze(D_RGB, 0), RGB

    def __len__(self):
        return self.P_N


# The dataset of spectral downsampled LR HSI to LR HSI
class HSI_sets(data.Dataset):
    def __init__(self, HSI, Label, P_N, P_S,srf,msi_channel):
        self.HSI = HSI
        self.size = HSI.shape
        self.P_S = P_S
        self.Label = Label

        self.P_N = P_N
        if srf is None:
            self.P = np.full([msi_channel, HSI.shape[0]], 1 / HSI.shape[0])
        else:
            self.P = srf.T
    #         c x C

    def __getitem__(self, index):
        [p, q] = self.Label[index, :]

        P_S = self.P_S

        HSI = self.HSI[:, p:p + P_S, q:q + P_S]
        P = torch.from_numpy(self.P).float().cuda()


        HSI = torch.reshape(HSI, (HSI.shape[0], P_S ** 2))

        D_HSI = torch.matmul(P, HSI).view(-1, P_S, P_S)
        HSI = HSI.view(HSI.shape[0], P_S, P_S)

        return D_HSI, HSI

    def __len__(self):
        return self.P_N


# Crop the test image into patches.
def Get_Label(im_size, patch_size):
    m, n = 0, 0
    P_number = 0
    Lable_table = []
    while 1:
        if m + patch_size < im_size[1] and n + patch_size < im_size[0]:
            Lable_table.append([m, n])
            m = m + patch_size
            P_number += 1
        elif m + patch_size >= im_size[1] and n + patch_size < im_size[0]:
            m = im_size[1] - patch_size
            Lable_table.append([m, n])
            m, n = 0, n + patch_size
            P_number += 1
        elif m + patch_size < im_size[1] and n + patch_size >= im_size[0]:
            Lable_table.append([m, im_size[0] - patch_size])
            m += patch_size
            P_number += 1
        elif m + patch_size >= im_size[1] and n + patch_size >= im_size[0]:
            Lable_table.append([im_size[1] - patch_size, im_size[0] - patch_size])
            P_number += 1
            break
    return np.array(Lable_table), P_number


# Spatial upsample model
def Spa_UpNet(image,opt):
    if opt.sf == 8:
        P_S = 32
    elif opt.sf == 16:
        P_S = 64
    else:
        P_S = 256
    Label, P_n = Get_Label(image.shape[1:], P_S)

    dataset = RGB_sets(image, Label, P_n, P_S, sf=opt.sf)

    data = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    N1 = nn.Sequential()
    N1.add(get_net_1(image.shape[0], 'skip', 'reflection', n_channels=image.shape[0], skip_n33d=40, skip_n33u=40, skip_n11=1, num_scales=2,
                     upsample_mode='bilinear'))

    N1.add(nn.Upsample(scale_factor=opt.sf, mode='bilinear'))
    N1.add(get_net_1(image.shape[0], 'skip', 'reflection', n_channels=image.shape[0], skip_n33d=40, skip_n33u=40, skip_n11=1, num_scales=2,
                     upsample_mode='bilinear'))
    N1 = N1.cuda()
    L1Loss = nn.L1Loss(reduce=True)
    optimizer_N1 = torch.optim.Adam(N1.parameters(), lr=5e-4)

    for epoch in range(500):
        running_loss = 0
        for i, batch in enumerate(data, 1):
            lr, hr = batch[0], batch[1]
            lr, hr = Variable(lr).cuda(), Variable(hr).cuda()

            out = N1(lr)
            loss = L1Loss(out, hr)
            running_loss += loss.detach()
            optimizer_N1.zero_grad()
            loss.backward()
            optimizer_N1.step()
    return N1


# spectral upsample model
def Spc_UpNet(image,msi_channel,srf=None):
    P_S = 8
    Label, P_n = Get_Label(image.shape[1:], P_S)
    dataset = HSI_sets(image, Label, P_n, P_S,srf,msi_channel)
    data = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    N2 = get_net_1(msi_channel, 'skip', 'reflection', n_channels=image.shape[0], skip_n33d=image.shape[0], skip_n33u=image.shape[0], skip_n11=4, num_scales=5,
                   upsample_mode='bilinear')
    N2 = N2.cuda()
    L1Loss = nn.L1Loss(reduce=True)
    optimizer_N2 = torch.optim.Adam(N2.parameters(), lr=5e-4)

    for epoch in range(500):
        running_loss = 0
        for i, batch in enumerate(data, 1):
            lr, hr = batch[0], batch[1]

            lr, hr = Variable(lr).float().cuda(), Variable(hr).cuda()

            out = N2(lr)
            loss = L1Loss(out, hr)
            running_loss += loss.detach()
            optimizer_N2.zero_grad()
            loss.backward()
            optimizer_N2.step()
    return N2


# Learnable spectral response matrix
class L_Dspec(nn.Module):
    def __init__(self, in_channel, out_channel, P_init):
        super(L_Dspec, self).__init__()
        self.in_channle = in_channel
        self.out_channel = out_channel
        self.P = Parameter(P_init)

    def forward(self, input):
        S = input.shape
        out = torch.reshape(input, [S[1], S[2] * S[3]])
        out = torch.matmul(self.P, out)

        return torch.reshape(out, [self.out_channel, S[2], S[3]])


class DBSR():
    def __init__(self,opt):
        self.opt = opt

    def PSNR_GPU(self, im_true, im_fake):
        data_range = 1
        _, C, H, W = im_true.size()
        err = torch.pow(im_true.clone() - im_fake.clone(), 2).mean(dim=(-1, -2), keepdim=True)
        psnr = 10. * torch.log10((data_range ** 2) / err)
        return torch.mean(psnr)

    def equip(self,srf,psf):
        self.srf = srf
        self.psf= psf
        if srf is not None:
            self.p=torch.FloatTensor(srf.T).cuda()

    def __call__(self,im_h,im_m,GT):
        L1Loss = nn.L1Loss()
        k = 0
        lr = self.opt.lr_i
        k += 1
        if self.opt.U_spc == 1:
            hsi_channel = im_h.shape[1]
            P_N = torch.full([im_m.shape[1], hsi_channel], 1 / hsi_channel)
            down_spc = L_Dspec(hsi_channel, im_m.shape[1], P_N).type(torch.cuda.FloatTensor).cuda()
            optimizer_spc = torch.optim.Adam(down_spc.parameters(), lr=self.opt.lr_dc, weight_decay=1e-5)
        if self.opt.pretrain == 1:
            print('Stage one : Pretrain the Spc_UpNet.')
            Spc_up = Spc_UpNet(im_h[0],im_m.shape[1],srf=self.srf)
            H_RGB = Spc_up(im_m)
            print('Stage two : Pretrain the Spa_UpNet.')
            Spa_up = Spa_UpNet(H_RGB[0],self.opt)
            H_HSI = Spa_up(im_h)
            net_input = Variable(0.8 * H_RGB + 0.2 * H_HSI).cuda()
        if self.opt.U_spa == 1:
            # Learnable spatial downsampler
            print('learn spatial')
            KS = 32
            dow = nn.Sequential(nn.ReplicationPad2d(int((KS - self.opt.sf) / 2.)), nn.Conv2d(1, 1, KS, self.opt.sf))
            class Apply(nn.Module):
                def __init__(self, what, dim, *args):
                    super(Apply, self).__init__()
                    self.dim = dim
                    self.what = what

                def forward(self, input):
                    inputs = []
                    for i in range(input.size(self.dim)):
                        inputs.append(self.what(input.narrow(self.dim, i, 1)))
                    return torch.cat(inputs, dim=self.dim)

                def __len__(self):
                    return len(self._modules)
            downs = Apply(dow, 1)
            downs = downs.cuda()
            optimizer_d = torch.optim.Adam(downs.parameters(), lr=self.opt.lr_da, weight_decay=1e-5)
        # get_input
        net = get_net_1(im_h.shape[1], 'skip', 'reflection', n_channels=im_h.shape[1], skip_n33d=256, skip_n33u=256,
                      skip_n11=1, num_scales=2, upsample_mode='bilinear')
        net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for i in range(self.opt.max_epoch):
            # input data
            output = net(net_input)
            # procese of output
            S = output.shape
            if self.opt.U_spa == 0:
                Dspa_O = downs(output)
                Dspa_O = Dspa_O.view(Dspa_O.shape[1], Dspa_O.shape[2], Dspa_O.shape[3])
            else:
                Dspa_O = downs(output)
                Dspa_O = torch.squeeze(Dspa_O, 0)
            if self.opt.U_spc == 0:
                out = output.view(S[1], S[2] * S[3])
                Dspc_O = torch.matmul(self.p, out).view(im_m.shape[1], S[2], S[3])
            else:
                Dspc_O = down_spc(output)
            # zero the grad
            optimizer.zero_grad()
            if self.opt.U_spc == 1:
                optimizer_spc.zero_grad()
            if self.opt.U_spa == 1:
                optimizer_d.zero_grad()
            loss = L1Loss(Dspa_O, im_h) + L1Loss(Dspc_O, im_m)
            # backward the loss
            loss.backward()
            # optimize the parameter
            optimizer.step()
            if self.opt.U_spc == 1:
                optimizer_spc.step()
            if self.opt.U_spa == 1:
                optimizer_d.step()
            if i % 100 == 0:
                output = Variable(output, requires_grad=False).cuda()
                output = output.view(S[1], S[2], S[3])
                if i % 1000 == 0:
                    lr = 0.7 * lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            if i == self.opt.max_epoch - 1:
                out = np.array(output.squeeze().detach().cpu())
                out = out.T
        return out 
                




