import matplotlib.pyplot as plt
import imgvision as iv
from Networks import model_generator
from utils import save_checkpoint,PSNR_GPU,reshuffle
import time
import torch
from torch import nn
import numpy as np

def Eva(model, validate_data_loader,model_folder,start_epoch=100,name='recon'):
    for iteration, batch in enumerate(validate_data_loader, 1):
        GT,  LRHSI, HRMSI = batch[‘hrhsi’].cuda(), batch[‘lrhsi’].cuda(), batch[‘hrmsi’].cuda()
        output_HRHSI = model(LRHSI, HRMSI,GT,iteration)
        output_HRHSI = output_HRHSI.detach().cpu().permute(1, 2, 0).numpy()
        GT = GT[0].detach().cpu().permute(1, 2, 0).numpy()
        iv.spectra_metric(output_HRHSI, GT).Evaluation()
        np.save(model_folder+f'specific{iteration}',output_HRHSI)

    return  output_HRHSI

def Eva_2(model, validate_data_loader,model_folder,start_epoch=100,name='recon'):
    PLoss=nn.L1Loss()
    #
    path_checkpoint = model_folder+"{}.pth".format(start_epoch)
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    #
    # print('Network is Successfully Loaded from %s' % (path_checkpoint))


    epoch_train_loss, epoch_val_loss = [], []
    psnr=[]

    # ============Epoch Train=============== #

    model.eval()

    for iteration, batch in enumerate(validate_data_loader, 1):

        GT,  LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        output_HRHSI = model(LRHSI, HRMSI)
        output_HRHSI = output_HRHSI[0].detach().cpu().permute(1, 2, 0).numpy()
        GT = GT[0].detach().cpu().permute(1, 2, 0).numpy()
        iv.spectra_metric(output_HRHSI, GT).Evaluation()
        np.save(f'UTAL/{iteration}',output_HRHSI)

    return  output_HRHSI



if __name__=='__main__':

    from torch.utils.data import DataLoader
    from Dataloader_tool import Large_dataset

    # Build Network
    Method = 'UTAL_specific'
    model, opt = model_generator(Method,'cuda')

    # Dataset Setting
    dataset_name = 'chikusei'
    model_folder =  'UTAL/'+ dataset_name +'/'
    # Training Setting
    Batch_size = 1
    start =2000
    patch_size= 512

    Test_data = Large_dataset(opt, patch_size, dataset_name, type='test')

    training_data_loader = DataLoader(dataset=Test_data, num_workers=0, batch_size=Batch_size, shuffle=False,
                                      pin_memory=True, drop_last=False)
    output_HRHSI = Eva(model,training_data_loader,model_folder=model_folder,start_epoch=start,name=Method)

