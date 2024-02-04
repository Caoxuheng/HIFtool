from .Model.spa_down import SpaDown, initialize_SpaDNet
from .Model.spe_down import SpeDown, initialize_SpeDNet
import torch
from torch import nn
from .Model.bonenet import MultiLevelFus
def tensor_svds(data, K):
    b, c, h, w = data.shape
    U, S, vh = torch.linalg.svd(data.flatten(2)[0], full_matrices=False)
    phi = U[:, :K] @ torch.diag(S[:K])
    C = vh[:K]
    return phi.view([1, c, K, 1]), C.view([1, K, h, w])
  
def PSNR_GPU(im_true, im_fake):
    data_range = 1
    _,C,H,W = im_true.size()
    err = torch.pow(im_true.clone()-im_fake.clone(),2).mean(dim=(-1,-2), keepdim=True)
    psnr = 10. * torch.log10((data_range**2)/err)
    return torch.mean(psnr)
  
def Couple_init(spa,spe, msi, hsi, k=3):
    Batch, Channel, Height, Weight = hsi.shape
    phi, c = tensor_svds(hsi, k)
    trainer_SPE = torch.optim.AdamW(params=spe.parameters(), lr=5e-3, weight_decay=0.0001)
    lrsched_SPE = torch.optim.lr_scheduler.StepLR(trainer_SPE, 50, 0.8)
    trainer_SPA = torch.optim.AdamW(params=spa.parameters(), lr=5e-3, weight_decay=0.0001)
    lrsched_SPA = torch.optim.lr_scheduler.StepLR(trainer_SPA, 50, 0.8)
    max_epochs = 500
    L1 = nn.L1Loss()
    L = []
    for epoch in range(max_epochs):
        trainer_SPA.zero_grad()
        trainer_SPE.zero_grad()
        pre_phi = spe(phi)
        lrmsi = spa(msi)
        loss =L1(lrmsi, torch.bmm(pre_phi[:, :, :, 0], c.view(1, k, -1)).view(1, msi.shape[1], Height, Weight)) + L1(lrmsi,spe(hsi))
        loss.backward()
        trainer_SPE.step()
        lrsched_SPE.step()
        trainer_SPA.step()
        lrsched_SPA.step()
        L.append(loss.detach().cpu().numpy())

class Feafusformer():
    def __init__(self, opt, sp_range,device):
        self.opt = opt
        self.sp_range = sp_range
        self.device = device

    def equip(self, sp_matrix, psf):
        self.sp_matrix = sp_matrix
        self.psf = psf

    def __call__(self, HSI,MSI):
        best = 0
        bestep = 0
        Batch, Channel, Height, Weight = HSI.shape

        # Define Networks
        SpaDNet = SpaDown(sf=self.opt.sf, predefine=None,iscal=self.opt.isCal_PSF).to(self.device)
        SpeDNet = SpeDown(span=self.sp_range,predefine=None,iscal=self.opt.isCal_SRF).to(self.device)


        # Fusformer = MultiLevelFus(HSI,MSI,torch.FloatTensor(self.sp_matrix).cuda()).to(self.device)
        # # Define Loss
        Fusformer = MultiLevelFus(HSI, MSI, SpeDNet.cuda()).to(self.device)

        loss = nn.L1Loss()
        # pec_loss = PerceptualLoss(Fusformer,MSI)
        self.isBlind = self.opt.isCal_SRF or self.opt.isCal_PSF
        if self.isBlind:
            phi, c = tensor_svds(HSI, K=self.opt.K)
            # initialize SpaDNet and SpeDNet
            initialize_SpeDNet(module=SpeDNet, msi=MSI, hsi=HSI, sf=self.opt.sf)
            initialize_SpaDNet(module=SpaDNet, msi=MSI, msi2=SpeDNet(HSI))
            # joint initialization
            Couple_init(SpaDNet, SpeDNet, MSI, HSI)

            trainer_spa = torch.optim.AdamW(params=SpaDNet.parameters(), lr=1e-5)
            trainer_spe = torch.optim.AdamW(params=SpeDNet.parameters(), lr=1e-5)

        trainer_fuse = torch.optim.AdamW(params=Fusformer.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.StepLR(trainer_fuse, step_size=100, gamma=0.99)
        l0 = []
        for i in range(self.opt.pre_epoch):
            # Pre-train
            trainer_fuse.zero_grad()
            pre = Fusformer()
            recon_HSI = SpaDNet(pre)
            recon_MSI = SpeDNet(pre)

            if self.isBlind:
                if i < self.opt.pre_epoch // 3:
                    # pre-launch
                    l = loss(recon_HSI, HSI) + loss(recon_MSI, MSI)
                    l.backward()
                else:
                    trainer_spa.zero_grad()
                    trainer_spe.zero_grad()
                    pre_phi = SpeDNet(phi)
                    l = loss(recon_HSI, HSI) + loss(recon_MSI, MSI) +  \
                        loss(SpaDNet(MSI),SpeDNet(HSI)) + \
                        loss(SpaDNet(MSI), torch.bmm(pre_phi[:, :, :, 0], c.view(1, self.opt.K, -1)).view(1, MSI.shape[1], Height, Weight))
                    l.backward()
                    trainer_spa.step()
                    trainer_spe.step()

            else:
                l = loss(recon_HSI, HSI) + loss(recon_MSI, MSI)
                l.backward()
            trainer_fuse.step()
            sched.step()

        return pre[0].detach().cpu().numpy().T

