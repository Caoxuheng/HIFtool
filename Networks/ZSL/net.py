import numpy as np
from .Model.cnn import ZSL_cnn
from .utils import  warm_lr_scheduler

class ZSL():
    def __init__(self,args):
        self.args = args
        self.step=0
        self.model = ZSL_cnn(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def __call__(self, lhsi, hmsi, GT):

        lhsi_np = lhsi[0].detach().cpu().numpy()
        U0, S, V = np.linalg.svd(np.dot(lhsi_np, lhsi_np.T))
        U0 = U0[:, 0:int(p)]
        U = U0
        U22 = torch.Tensor(U0)

        for step in range(400):

            lr = warm_lr_scheduler(self.optimizer, self.args.init_lr1, self.args.init_lr2, warm_iter, step, lr_decay_iter=1, max_iter=maxiteration,
                                   power= self.args.decay_power)

            output, Xre = self.model(lhsi, hmsi,U22)
            loss = loss_func(Xre, a1.cuda(), a2.cuda(), self.args.sf)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return  Xre[0].detach().cpu().numpy().T
