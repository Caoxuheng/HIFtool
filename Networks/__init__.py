import numpy as np


def model_generator(method: str, device="cuda"):

    method_key = method.lower().replace('-', '').replace('_', '').replace(' ', '')

    if method_key in {'cdmap', 'cdmapcpu', 'cdmapcuda', 'cdmapgpu'}:
        from .CDMAP import CDMAP, CDMAPConfig
        if method_key == 'cdmapcpu':
            backend = 'cpu'
        elif method_key in {'cdmapcuda', 'cdmapgpu'}:
            backend = 'cuda'
        else:
            backend = 'auto'
        opt = CDMAPConfig(backend=backend)
        model = CDMAP(opt)
    elif 'emrdiff' in method_key:
        from .EMRDiff.config import args_parser
        from .EMRDiff.net import EMRDiffNet
        opt = args_parser()
        model = EMRDiffNet(opt).to(device)
    elif 'psrfdiff' in method_key:
        from .PSRFDiff.config import args_parser
        from .PSRFDiff.net import PSRFDiffNet
        opt = args_parser()
        model = PSRFDiffNet(opt).to(device)
    elif 'bhsrnet' in method_key or method_key == 'bhsr':
        from .BHSRNet.config import args_parser
        from .BHSRNet.net import BHSRNet
        opt = args_parser()
        model = BHSRNet(opt).to(device)
    elif 'CaFormer' in method:
        from .CaFormer.net import CaFormer
        from .CaFormer.Config import args as opt
        num_iterations = int(method.split('_')[-1])
        model = CaFormer(sf=opt.sf, in_c=opt.hsi_channel, out_c=opt.msi_channel,
                         n_feat=opt.n_feat, nums_stages=num_iterations - 1, n_depth=opt.n_depth).to(device)
    elif 'DTDNML' in method:
        from .DTDNML.dtdnml import DTDNML
        from .DTDNML.Config import args as opt
        sp_range = [[0,10],[10,20],[20,30]]
        model = DTDNML()
        model.initialize(opt,sp_range=sp_range)
    elif 'BUSI' in method:
        from .BUSI.model import BUSI
        from .BUSI.Config import args as opt
        # BUSIFusion uses ker_sz as both its PSF kernel size and the synthetic
        # spatial scale.  Keep it in sync with --sf for CAVE/HARVARD.
        opt.ker_sz = opt.sf
        model = BUSI(opt)

    elif 'UTAL' in method:
        
        from .UTAL.net import ThreeBranch_Net,Meta_train, Specific_Learning
        from .UTAL.config import args_parser
        opt = args_parser()

        if 'meta' in method:
            model =  Meta_train(opt,device)
        elif 'specific' in method:
            model =  Specific_Learning(opt,device)
            print('specific learning')
        else:
            model = ThreeBranch_Net(opt,device).to(device)

    elif 'PSRT' in method:
        from .PSRT.net import PSRTnet
        from  .PSRT.config import args_parser
        opt = args_parser()
        model = PSRTnet(opt).to(device)
    elif 'MSST' in method:
        from .MSST.net import Net
        from .MSST.Config import argsParser
        opt = argsParser()
        model = Net(opt).to(device)
    elif 'MoGDCN' in method:
        from .MoGDCN.net import VSR_CAS
        from .MSST.Config import argsParser
        opt = argsParser()
        model = VSR_CAS(opt).to(device)
    elif 'Fusformer' in method :
        from .Fusformer.net import MainNet
        from .MSST.Config import argsParser
        opt = argsParser()
        model = MainNet(sf=opt.sf, channel=opt.hsi_channel,msichannel=opt.msi_channel).to(device)
    elif 'DCTransformer' in method :
        from .DCTransformer.net import DCT
        from .DCTransformer.Config import opt
        model =DCT(opt.hsi_channel,opt.msi_channel, opt.sf).to(device)
    elif 'PSTUN' in method:
        from .PSTUN.net import PSTUN
        from .PSTUN.config import opt
        model = PSTUN(in_channels=opt.msi_channel, in_feat=32, out_channels=opt.hsi_channel).to(device)
    elif 'HyMS' in method:
        from .HyMS.config import args
        from .HyMS.main_gpu import HyMS

        model =HyMS(args,device)
        opt= args
    elif 'HySure' in method:
        from .HySure.config import args
        from .HySure.HySure import HySure

        model =HySure(args)
        opt= args
    elif 'UDALN' in method:
        from .UDALN.net import udaln
        from .UDALN.config import args as opt

        # sp_range = [list(range(30)),list(range(13,50)),list(range(41,84)),list(range(68,128))]
        sp_range = np.array([[0, 10], [10, 20], [20, 31]])
        model = udaln(opt,sp_range)
    elif 'DBSR' in method:
        from .DBSR.net import DBSR
        from .DBSR.config import  opt

        model = DBSR(opt)
    elif 'FeafusFormer' in method:
        from .FeafusFormer.net import Feafusformer
        from .FeafusFormer.config import opt
        sp_range = [list(range(0, 10)), list(range(10, 20)), list(range(20, 31))]
        # sp_range = np.array([range(4)])
        model = Feafusformer(opt,sp_range,device)
    elif 'ZSL' in method:
        from .ZSL.config import args_parser
        from .ZSL.net import ZSL
        opt = args_parser()
        model = ZSL(opt, device)

    elif 'PUTPDN' in method:
        from .PUTPDN.config import args_parser
        from .PUTPDN.net import PUTPDN
        opt = args_parser()
        model = PUTPDN(opt).to(device)

    
    else:
        raise ValueError(f'Method {method!r} is not defined.')

    return model, opt
