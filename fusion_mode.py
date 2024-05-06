import torch, os
from torch import nn
import imgvision as iv
import numpy as np

def ModeSelection(Mode:str):
    if 'unsupervised' in Mode:
        from utils import SpaDown, getInputImgs
        import scipy.io as sio
        def Unsupervisedfusion(model, opt, model_folder, dataset_name,srf):
            if not os.path.isdir(model_folder):
                os.mkdir(model_folder)
            Spatialdown = SpaDown(opt.sf)
            GT_mat = np.load(f'Multispectral Image Dataset\{dataset_name}\GT.npy')
            sio.savemat(f'Multispectral Image Dataset\{dataset_name}\GT.mat',{'HSI':GT_mat})
            GT =   sio.loadmat(f'Multispectral Image Dataset\{dataset_name}\GT.mat')['HSI'][:256,:256]
            LRHSI, HRMSI, GT = getInputImgs(GT, dataset_name,Spatialdown,srf)
            LRHSI, HRMSI = LRHSI.cuda(), HRMSI.cuda()
            Re = model(LRHSI, HRMSI, torch.FloatTensor(GT).T.unsqueeze(0).cuda())
            iv.spectra_metric(Re, GT).Evaluation()
            np.save(model_folder, Re)
            # save_checkpoint(save_folder, model, optimizer, lr, idx)
        fusion = Unsupervisedfusion
    elif 'supervised' in Mode:
        from utils import PSNR_GPU, save_checkpoint
        def Supervisedfusion(model, training_data_loader, validate_data_loader, model_folder, optimizer, lr,
                             start_epoch=0,
                             end_epoch=2000, ckpt_step=50, RESUME=False):
            PLoss = nn.L1Loss()
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100,
                                                           gamma=0.5)
            print('Start training...')

            if RESUME:
                path_checkpoint = model_folder + "{}.pth".format(start_epoch)
                checkpoint = torch.load(path_checkpoint)

                model.load_state_dict(checkpoint['net'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer.param_groups[0]['lr'] = checkpoint["lr"] * 1.5
                start_epoch = checkpoint['epoch']
                print('Network is Successfully Loaded from %s' % (path_checkpoint))
            best_epoch = 0
            for epoch in range(start_epoch, end_epoch, 1):

                epoch += 1
                epoch_train_loss, epoch_val_loss = [], []
                psnr = []
                psnr_train = []
                # ============Epoch Train=============== #
                model.train()
                for iteration, batch in enumerate(training_data_loader, 1):
                    GT, LRHSI, HRMSI = batch['hrhsi'].cuda(), batch['lrhsi'], batch['hrmsi']
                    optimizer.zero_grad()  # fixed
                    output_HRHSI = model(LRHSI.cuda(), HRMSI.cuda())
                    Pixelwise_Loss = PLoss(output_HRHSI, GT)
                    epoch_train_loss.append(Pixelwise_Loss.item())
                    Pixelwise_Loss.backward()  # fixed
                    optimizer.step()  # fixed
                    psnr_train.append(PSNR_GPU(output_HRHSI, GT).item())
                    if iteration % 25 == 0:
                        print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                           Pixelwise_Loss.item()))

                lr_scheduler.step()
                t_loss = np.nanmean(np.array(epoch_train_loss))
                psnr_ = np.nanmean(np.array(psnr_train))
                print('Epoch: {}/{} \t training loss: {:.7f}\tpsnr:{:.2f}'.format(end_epoch, epoch, t_loss, psnr_))
                # ============Epoch Validate=============== #
                if epoch % ckpt_step == 0:
                    model.eval()
                    with torch.no_grad():
                        for iteration, batch in enumerate(validate_data_loader, 1):
                            GT, LRHSI, HRMSI = batch['hrhsi'].cuda(), batch['lrhsi'], batch['hrmsi']

                            output_HRHSI = model(LRHSI.cuda(), HRMSI.cuda())

                            Pixelwise_Loss = PLoss(output_HRHSI, GT)
                            MyVloss = Pixelwise_Loss

                            epoch_val_loss.append(MyVloss.item())
                            psnr.append(PSNR_GPU(output_HRHSI, GT).item())
                    v_loss = np.nanmean(np.array(epoch_val_loss))
                    psnr = np.nanmean(np.array(psnr))
                    best_epoch = epoch
                    save_checkpoint(model_folder, model, optimizer, lr, epoch)

                    print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
                    print('             validate loss: {:.7f}'.format(v_loss))
                    print('             PSNR loss: {:.7f}'.format(psnr))
            return best_epoch
        fusion = Supervisedfusion
    elif 'model' in Mode:
        import cv2
        def modelfusion(model, opt, model_folder, dataset_name, srf):
            save_folder = model_folder + '/'
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            for ax, idx in enumerate([1]):
                base = sio.loadmat(f'Multispectral Image Dataset/{dataset_name}/{dataset_name}.mat')
                GT = base['HSI']
                LRHSI = cv2.GaussianBlur(GT, ksize=[opt.sf // 2 + 1] * 2, sigmaX=opt.sf * 0.866,
                                         sigmaY=opt.sf * 0.866)[opt.sf // 2::opt.sf, opt.sf // 2::opt.sf]
                HRMSI = GT @ srf

                Re = model(LRHSI, HRMSI, dataset_name)
                np.save(save_folder + str(idx), Re)
                iv.spectra_metric(GT, Re, scale=opt.sf).Evaluation()
        fusion = modelfusion
    print(f'Fusion Mode:\033[1;33m {Mode} \033[0m')
    return fusion



