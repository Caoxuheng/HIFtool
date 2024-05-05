import matplotlib.pyplot as plt
import imgvision as iv
from Networks import model_generator
from utils import save_checkpoint,PSNR_GPU,reshuffle
import time
import torch
from torch import nn
import numpy as np

def Eva(model, validate_data_loader,model_folder,start_epoch=100,name='recon'):
    PLoss=nn.L1Loss()

    path_checkpoint = model_folder+"{}.pth".format(start_epoch)
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])

    print('Network is Successfully Loaded from %s' % (path_checkpoint))


    epoch_train_loss, epoch_val_loss = [], []
    psnr=[]

    # ============Epoch Train=============== #

    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(validate_data_loader, 1):
            GT,  LRHSI, HRMSI = batch['hrhsi'].cuda(), batch['lrhsi'].cuda(), batch['hrmsi'].cuda()
            GT /=GT.max()
            LRHSI /=LRHSI.max()
            HRMSI /=HRMSI.max()


            # GPU显存不够请使用改代码： 将图像分为t^2块，sf为上采样参数
            #output_HRHSI = torch.empty_like(GT)
            #t=4
            #sf=4
            #n_ = GT.shape[2]
            #for i in range(t):
            #    for j in range(t):
            #        n = n_//t
            #        output_HRHSI[:,:,i*n:(i+1)*n,j*n:(j+1)*n] = model(LRHSI[:,:,i*n//sf:(i+1)*n//sf,j*n//sf:(j+1)*n//sf],
            #                                                          HRMSI[:,:,i*n:(i+1)*n,j*n:(j+1)*n])



            output_HRHSI = model(LRHSI, HRMSI)




            Pixelwise_Loss = PLoss(output_HRHSI, GT)
            MyVloss = Pixelwise_Loss
            epoch_val_loss.append(MyVloss.item())
            psnr.append(PSNR_GPU(output_HRHSI, GT).item())
    v_loss = np.nanmean(np.array(epoch_val_loss))
    psnr = np.nanmean(np.array( psnr))
    print('             validate loss: {:.7f}'.format(v_loss))
    print('             PSNR loss: {:.7f}'.format(psnr ))

    output_HRHSI =output_HRHSI[0].detach().cpu().permute(1,2,0).numpy()
    print(output_HRHSI.shape)


    GT = GT[0].detach().cpu().permute(1, 2, 0).numpy()
    iv.spectra_metric(output_HRHSI,GT).Evaluation()
    rgb = np.concatenate(
        [output_HRHSI[:, :, 2][:, :, None], output_HRHSI[:, :, 1][:, :, None], output_HRHSI[:, :, 0][:, :, None]],
        axis=2)
    from PIL  import  Image
    RGB = np.clip(rgb,0,1)
    RGB = np.uint8(RGB*255)
    RGB = Image.fromarray(RGB)
    RGB.save(model_folder+name+'.png')
    np.save(model_folder+name,output_HRHSI)
    return  output_HRHSI




if __name__=='__main__':

    from torch.utils.data import DataLoader
    from Dataloader_tool import QBDataset

    # Build Network
    Method = 'PSRT'
    model, opt = model_generator(Method,'cuda')

    # Dataset Setting
    dataset_name = 'QB'
    model_folder = Method + '/' + dataset_name + '/'
    # Training Setting
    Batch_size = 1
    start =1000

    # Train_data = QBDataset('Multispectral Image Dataset\QB/test/Test(HxWxC)_qb_data8.mat', type='test')

    Test_data = QBDataset('Multispectral Image Dataset\QB/test/Test(HxWxC)_qb_data_fr1.mat', type='test')

    training_data_loader = DataLoader(dataset=Test_data, num_workers=0, batch_size=Batch_size, shuffle=False,
                                      pin_memory=True, drop_last=False)
    output_HRHSI = Eva(model,training_data_loader,model_folder=model_folder,start_epoch=start,name=Method)
