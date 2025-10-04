from Networks import model_generator
from utils import save_checkpoint,PSNR_GPU,reshuffle
import imgvision as iv
import torch
from torch import nn
import numpy as np


def Eva(opt,model, validate_data_loader,model_folder,start_epoch=100,t=4,name='recon'):

    path_checkpoint = model_folder+"{}.pth".format(start_epoch)
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])

    print('Network is Successfully Loaded from %s' % (path_checkpoint))

    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(validate_data_loader, 1):
            GT,  LRHSI, HRMSI = batch[0], batch[1].cuda(), batch[2].cuda()
            filename = validate_data_loader.dataset.test_name
            # GPU显存不够请使用改代码： 将图像分为t^2块，sf为上采样参数
            # output_HRHSI = torch.empty_like(GT)

            # n_ = GT.shape[2]
            # for i in range(t):
            #    for j in range(t):
            #        n = n_//t
            #        output_HRHSI[:,:,i*n:(i+1)*n,j*n:(j+1)*n] = model(LRHSI[:,:,i*n//opt.sf:(i+1)*n//opt.sf,j*n//opt.sf:(j+1)*n//opt.sf],
            #                                                          HRMSI[:,:,i*n:(i+1)*n,j*n:(j+1)*n])

            output_HRHSI = model(LRHSI, HRMSI)[0].detach().cpu().permute(1, 2, 0).numpy()
            GT = GT[0].permute(1, 2, 0).numpy()
            iv.spectra_metric(output_HRHSI,GT).Evaluation()
            np.save(model_folder + str(filename[iteration-1]), output_HRHSI)




if __name__=='__main__':

    from torch.utils.data import DataLoader

    # Build Network
    Method = 'PSRT'

    model, opt = model_generator(Method,'cuda')

    dataset_name = 'CAVE'
    model_folder = Method + '/' + dataset_name + '/'
    patch_size= 512

    # Scheme Setting
    bestepoch= 1000

    from Dataloader_tool import Large_dataset
    Val_data = Large_dataset(opt,  patch_size,type='test')

    validate_data_loader = DataLoader(dataset=Val_data, num_workers=0, batch_size=1, shuffle=False,
                                          pin_memory=True, drop_last=False)

    Eva( opt,model,validate_data_loader,  model_folder, bestepoch)


