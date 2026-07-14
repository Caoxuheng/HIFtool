from .Model.AngSim_init import initialization, initialization_cpu


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
        use_gpu = 'cuda' in self.device.lower() or 'gpu' in self.device.lower()
        if use_gpu:
            try:
                from .Model.PolyAct_gpu import Modification
                from .Model.SpeSpa_Refine_gpu import Refinement
            except ImportError:
                # The original GPU implementation requires CuPy/CuPyX.  Keep
                # HyMS runnable in the standard PyTorch environment by using
                # the upstream SciPy implementation when that optional CUDA
                # package is unavailable.
                use_gpu = False

        if use_gpu:
            X_in = initialization(LR_HSI.copy(), HR_MSI.copy(), self.d, self.srf, 64, self.sf)
            X_mod = Modification(X_in.copy(), HR_MSI.copy(), self.srf, u=64, data=name)
            X_re = Refinement(X_mod, HR_MSI, self.beta, self.gamma, self.srf.T, self.k)
        else:
            from .Model.PolyAct import Modification
            from .Model.SpeSpa_Refine import Refinement
            X_in = initialization_cpu(LR_HSI.copy(), HR_MSI.copy(), self.d, self.srf, self.sf)
            X_mod = Modification(X_in.copy(), HR_MSI.copy(), self.srf, data=name)
            X_re = Refinement(X_mod, HR_MSI, self.beta, self.gamma, self.srf.T, self.k)

        return X_re
