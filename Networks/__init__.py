import numpy as np


def model_generator(method:str, device="cuda"):


    if 'CaFormer' in method:
        from .CaFormer.net import CaFormer
        from .CaFormer.Config import args as opt
        num_iterations = int(method.split('_')[-1])
        model = CaFormer(sf=opt.sf,in_c=opt.msi_channel, n_feat=opt.n_feat, nums_stages=num_iterations - 1,n_depth=opt.n_depth).to(device)
    elif 'DTDNML' in method:
        from .DTDNML.dtdnml import DTDNML
        from .DTDNML.Config import args as opt
        sp_range = [[0,10],[10,20],[20,30]]
        model = DTDNML()
        model.initialize(opt,sp_range=sp_range)

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
        sp_range = np.array([[0,30],[13, 50], [41, 84], [68, 127]])
        model = udaln(opt,sp_range)
    elif 'DBSR' in method:
        from .DBSR.net import DBSR
        from .DBSR.config import  opt

        model = DBSR(opt)
    elif 'FeafusFormer' in method:
        from .FeafusFormer.net import Feafusformer
        from .FeafusFormer.config import opt
        sp_range = [list(range(30)),list(range(13,50)),list(range(41,84)),list(range(68,128))]
        # sp_range = np.array([range(4)])
        model = Feafusformer(opt,sp_range,device)
    elif 'ZSL' in method:
        from .ZSL.net import ZSL
        from .ZSL.config import opt

        model = ZSL(opt)

    
    else:
        print(f'opt.Method {method} is not defined !!!!')

    return model, opt
