import numpy as np


def model_generator(method:str, device="cuda"):


    if 'PKLNet' in method:
        from .PKLNet.PKLNet import PKLNet
        from .PKLNet.Config import args as opt
        num_iterations = int(method.split('_')[-1])
        model = PKLNet(sf=opt.sf,in_c=opt.inchannel, n_feat=opt.n_feat, nums_stages=num_iterations - 1,n_depth=opt.n_depth).to(device)
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

        opt = None
        model = MainNet(sf=4, channel=4,msichannel=1).to(device)
    elif 'DCTransformer' in method :
        from .DCTransformer.net import DCT
        from .DCTransformer.Config import opt
        model =DCT(opt.hsi_channel,opt.msi_channel, opt.sf).to(device)
    elif 'HyMS' in method:
        from .HyMS.config import args
        from .HyMS.main_gpu import HyMS

        model =HyMS(args)
        opt= args
    elif 'UDALN' in method:
        from .UDALN.net import udaln
        from .UDALN.config import args as opt

        # sp_range = [list(range(30)),list(range(13,50)),list(range(41,84)),list(range(68,128))]
        sp_range = np.array([[0,30],[13, 50], [41, 84], [68, 127]])


        model = udaln(opt,sp_range)
    elif 'FeafusFormer' in method:
        from .FeafusFormer.net import Feafusformer
        from .FeafusFormer.config import opt
        sp_range = np.array([range(4)])
        model = Feafusformer(opt,sp_range,device)
    else:
        print(f'opt.Method {method} is not defined !!!!')

    return model, opt
