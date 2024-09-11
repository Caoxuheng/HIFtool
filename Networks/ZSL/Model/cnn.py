from torch import nn
import torch

class ZSL_cnn(nn.Module):
    def __init__(self,args):
        super(ZSL_cnn, self).__init__()
        a,b = args.p,args.msi_channel
        self.conv1 = nn.Sequential(        
            nn.Conv2d(a+b, 128-b, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
            )
        self.conv2 = nn.Sequential(        
        nn.Conv2d(128, 128-b, 3, 1, 1),     
        nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv3 = nn.Sequential(        
        nn.Conv2d(128, 128-b, 3, 1, 1),     
        nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv4 = nn.Sequential(        
        nn.Conv2d(128, a, 3, 1, 1),     
        )
        
        basecoeff = torch.Tensor([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
                     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
                    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
                    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
                     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
                    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
                     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
                     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
                     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
                    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
                    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
                     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
                    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
                     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
                     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])
        coeff = torch.mm(basecoeff.T, basecoeff)
        coeff = torch.Tensor(coeff)[None,None]
        self.coeff = torch.repeat_interleave(coeff, a,0)
        self.Upsample_4 = nn.ConvTranspose2d(bias=None,stride=4,padding=21,output_padding=1,groups=args.hsi_channel,dilation=1)
        self.Upsample_4.weight.data = self.coeff
        self.Upsample_4.requires_grad_(False)

    def forward(self, HSI,MSI,U22):

        x1=self.Upsample_4(HSI)
        x2 = torch.cat((x1,MSI),1)
        x2 = torch.cat((self.conv1(x2),MSI),1)
        x2 = torch.cat((self.conv2(x2),MSI),1)
        x2 = torch.cat((self.conv3(x2),MSI),1)
        x3 = self.conv4(x2)
        output = x3+x1
        Fuse1=torch.tensordot(U22, output, dims=([1], [1]))
        Xre=torch.Tensor.permute(Fuse1,(1,0,2,3))
        return x3+x1, Xre
