def modelfusion(model,opt,model_folder,dataset_name):

    import os
    import imgvision as iv

    save_folder =model_folder+ '/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    for ax, idx in enumerate([1]):
        base = sio.loadmat(f'Multispectral Image Dataset/{dataset_name}/{dataset_name}.mat')
        GT = base['HSI']
        srf = sio.loadmat('Dataloader_tool/srflib/chikuseisrf.mat')['R']

        
        LRHSI = cv2.GaussianBlur(GT, ksize=[32 // 2  + 1] * 2, sigmaX=32 * 0.866,
                                 sigmaY=32 * 0.866)[32 // 2::32, 32 // 2::32]

        #  Generate HRMSI
        HRMSI = GT @ srf
        Re = model(LRHSI,HRMSI,dataset_name)
        np.save(save_folder+str(idx), Re)
        iv.spectra_metric(GT,Re,scale=opt.sf).Evaluation()

def train(model,training_data_loader, validate_data_loader,model_folder,optimizer,lr,start_epoch=0,end_epoch=2000,ckpt_step=50,RESUME=False):
    PLoss=nn.L1Loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100,
                                                   gamma=0.5)
    print('Start training...')

    if RESUME:
        path_checkpoint = model_folder+"{}.pth".format(start_epoch)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr']=checkpoint["lr"]*1.5
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    best_epoch=0
    for epoch in range(start_epoch, end_epoch, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        psnr=[]
        psnr_train=[]
        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            GT, LRHSI, HRMSI = batch['hrhsi'].cuda(),batch['lrhsi'],batch['hrmsi']
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
        print('Epoch: {}/{} \t training loss: {:.7f}\tpsnr:{:.2f}'.format(end_epoch, epoch, t_loss,psnr_ ))
        # ============Epoch Validate=============== #
        if epoch % ckpt_step== 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT, LRHSI, HRMSI = batch['hrhsi'].cuda(),batch['lrhsi'],batch['hrmsi']

                    output_HRHSI = model(LRHSI.cuda(), HRMSI.cuda())

                    Pixelwise_Loss = PLoss(output_HRHSI, GT)
                    MyVloss = Pixelwise_Loss

                    epoch_val_loss.append(MyVloss.item())
                    psnr.append(PSNR_GPU(output_HRHSI, GT).item())
            v_loss = np.nanmean(np.array(epoch_val_loss))
            psnr = np.nanmean(np.array( psnr))
            best_epoch=epoch
            save_checkpoint(model_folder, model, optimizer, lr, epoch)

            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             PSNR loss: {:.7f}'.format(psnr ))
    return best_epoch

def unsupervisedfusion(model,opt,model_folder,dataset_name):

    import os
    import imgvision as iv
    from utils import SpaDown,getInputImgs
    import scipy.io as sio

    save_folder =model_folder+ dataset_name +'/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    R = torch.tensor(sio.loadmat('Dataloader_tool/srflib/chikusei_128_4.mat')['R']).float().cuda()



    Spatialdown = SpaDown(opt.sf)
    LRHSI, HRMSI, GT = getInputImgs(opt,dataset_name, 0, Spatialdown)
    LRHSI,HRMSI = LRHSI.cuda(), HRMSI.cuda()
    Re = model(LRHSI,HRMSI,torch.FloatTensor(GT).T.unsqueeze(0).cuda())
    iv.spectra_metric(Re,GT).Evaluation()
    np.save(save_folder, Re)
    # save_checkpoint(save_folder, model, optimizer, lr, idx)

if __name__=='__main__':

    from Networks import model_generator
    import numpy as np


    # Build Network
    case = ['model','unsupervised','supervised']
    case = case[2]
    Method = 'DCTransformer'
    model, opt = model_generator(Method)
    # Dataset Setting
    dataset_name = 'chikusei'
    model_folder = Method + '/' + dataset_name + '/'

    if case== 'model':
        import scipy.io as sio
        srf = sio.loadmat('Dataloader_tool/srflib/chikusei_128_4.mat')['R']
        model.equip(srf)
        
        modelfusion(model,opt,model_folder,dataset_name,srf)
    else:
        import scipy.io as sio
        from utils import save_checkpoint, PSNR_GPU
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        if case =='unsupervised':
            srf = sio.loadmat('Dataloader_tool/srflib/chikusei_128_4.mat')['R']
            import cv2
            psf= cv2.getGaussianKernel(opt.sf//2 * 2 + 1, sigma=opt.sf*0.666)
            sp_matrix = psf @ psf.T
            model.equip(srf,sp_matrix)
            unsupervisedfusion(model,opt,model_folder,dataset_name)
        else:
            # Training Setting
            Batch_size = 2
            end_epoch = 2000
            ckpt_step = 50
            lr = 1e-4
            # Scheme Setting
            General = 1
            Specific = 0
            resume = False
            start = 0
            bestepoch = 1450

            if General:
                from Dataloader_tool import ChikuseiDataset

                Train_data = ChikuseiDataset(f'Multispectral Image Dataset\{dataset_name}\{dataset_name}.mat',type='train')
                Val_data =  ChikuseiDataset(f'Multispectral Image Dataset\{dataset_name}\{dataset_name}.mat',type='eval')


                training_data_loader = DataLoader(dataset=Train_data, num_workers=0, batch_size=Batch_size, shuffle=True,
                                              pin_memory=True, drop_last=False)
                validate_data_loader = DataLoader(dataset=Val_data, num_workers=0, batch_size=1, shuffle=True,
                                                  pin_memory=True, drop_last=True)
                optimizer = torch.optim.Adam(model.parameters(),lr=lr)
                bestepoch = train(model,training_data_loader,validate_data_loader,model_folder=model_folder,optimizer=optimizer,lr=lr,start_epoch=start,end_epoch=end_epoch,ckpt_step=ckpt_step,RESUME=resume)

  
