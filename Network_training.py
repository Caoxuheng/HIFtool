from Networks import model_generator
from fusion_mode import ModeSelection
import scipy.io as sio
import os

if __name__ =='__main__':
    # ========================Select fusion mode=====================
    case_lst = ['model','unsupervised','supervised']
    case = case_lst[1]
    Fusion  = ModeSelection(case)

    # ========================Build Network==========================
    Method = 'FeafusFormer'
    model, opt = model_generator(Method,'cuda')

    # ========================Dataset Setting========================
    dataset_name = 'chikusei'
    model_folder = Method + '/' + dataset_name + '/'
    if not os.path.isdir(Method):
        os.mkdir(Method)

    if case== 'model':
        '''
        Model-based methods come with the requirements of [Spectral response function] and [point spread function]
        '''
        import cv2
        from utils import fspecial

        # Load [Spectral response function] and [point spread function]
        srf = sio.loadmat('Dataloader_tool/srflib/chikuseisrf.mat')['R']
        psf =fspecial('gaussian',opt.sf//2+1,opt.sf*0.886)
        # Config model
        model.equip(srf,psf)
        Fusion(model,opt,model_folder,dataset_name,srf)
    else:
        import cv2
        import torch
        from torch.utils.data import DataLoader
        if case =='unsupervised':
            blind = True
            if blind is True:
                mat_save_path = 'E:/Multispectral Image Dataset\QB/test/Test(HxWxC)_qb_data8.mat'
            else:
                srf = sio.loadmat('Dataloader_tool/srflib/chikusei_128_4.mat')['R']
                psf= cv2.getGaussianKernel(opt.sf//2 * 2 + 1, sigma=opt.sf*0.866)
                sp_matrix = psf @ psf.T
                model.equip(srf,sp_matrix)

            Fusion(model,model_folder=model_folder,blind=True,mat_save_path= mat_save_path ,opt=opt,dataset_name=dataset_name,srf=None)

        else:
            # Training Setting
            Batch_size = 2
            end_epoch = 2000
            ckpt_step = 50
            lr = 1e-4
            # Scheme Setting
            General = 1
            Specific = 0
            # Resume
            resume = False
            start = 0

            if General:
                from Dataloader_tool import ChikuseiDataset

                Train_data = ChikuseiDataset(f'Multispectral Image Dataset\{dataset_name}\{dataset_name}.mat',type='train')
                Val_data =  ChikuseiDataset(f'Multispectral Image Dataset\{dataset_name}\{dataset_name}.mat',type='eval')


                training_data_loader = DataLoader(dataset=Train_data, num_workers=0, batch_size=Batch_size, shuffle=True,
                                              pin_memory=True, drop_last=False)
                validate_data_loader = DataLoader(dataset=Val_data, num_workers=0, batch_size=1, shuffle=True,
                                                  pin_memory=True, drop_last=True)
                optimizer = torch.optim.Adam(model.parameters(),lr=lr)
                bestepoch = Fusion(model,training_data_loader,validate_data_loader,model_folder='PretrainModel/'+model_folder,optimizer=optimizer,lr=lr,start_epoch=start,end_epoch=end_epoch,ckpt_step=ckpt_step,RESUME=resume)


