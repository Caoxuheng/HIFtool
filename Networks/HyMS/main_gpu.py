
# from .Model.PolyAct_gpu import Modification
from .Model.PolyAct import Modification
from .Model.AngSim_init import initialization
# from .Model.SpeSpa_Refine_gpu import Refinement
from .Model.SpeSpa_Refine import Refinement


class HyMS():
    def __init__(self,args):
        self.d = args.d
        self.sf = args.sf
        self.beta, self.gamma=args.beta, args.gamma


    def equip(self,srf):
        self.srf = srf
    def __call__(self,LR_HSI,HR_MSI,name):
        # Stage1: Initialization
        X_in = initialization(LR_HSI.copy(), HR_MSI.copy(), self.d, self.srf, 64, self.sf)
        # Stage2: Modificaiton
        X_mod = Modification(X_in.copy(), HR_MSI.copy(), self.srf,data=name)
        # Stage3: Refinement
        X_re = Refinement(X_mod, HR_MSI, self.beta, self.gamma, self.srf.T)
        return X_re