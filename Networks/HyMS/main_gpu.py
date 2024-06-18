from .Model.AngSim_init import initialization


class HyMS():
    def __init__(self,args,device):
        self.d = args.d
        self.sf = args.sf
        self.k=args.k
        self.beta, self.gamma=args.beta, args.gamma
        self.device = device



    def equip(self,srf,*args):
        self.srf = srf
    def __call__(self,LR_HSI,HR_MSI,name):
        if 'cuda' or 'gpu' in self.device.lower:


            from .Model.PolyAct_gpu import Modification
            from .Model.SpeSpa_Refine_gpu import Refinement
            # Stage1: Initialization
            X_in = initialization(LR_HSI.copy(), HR_MSI.copy(), self.d, self.srf, 64, self.sf)
            # Stage2: Modificaiton
            X_mod = Modification(X_in.copy(), HR_MSI.copy(), self.srf,u=64, data=name)
            # Stage3: Refinement
            X_re = Refinement(X_mod, HR_MSI, self.beta, self.gamma, self.srf.T,self.k)

        else:
            from .Model.PolyAct import Modification
            from .Model.SpeSpa_Refine import Refinement
            # Stage1: Initialization
            X_in = initialization(LR_HSI.copy(), HR_MSI.copy(), self.d, self.srf, 64, self.sf)
            # Stage2: Modificaiton
            X_mod = Modification(X_in.copy(), HR_MSI.copy(), self.srf, data=name)
            # Stage3: Refinement
            X_re = Refinement(X_mod, HR_MSI, self.beta, self.gamma, self.srf.T,self.k)

        return X_re
