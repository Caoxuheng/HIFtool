from Networks import model_generator
from utils import save_checkpoint,PSNR_GPU,reshuffle
import time
import torch
from torch import nn
import numpy as np

def train(model,training_data_loader, validate_data_loader,model_folder,optimizer,lr,start_epoch=0,end_epoch=2000,ckpt_step=50,RESUME=False,meta=False):
    PLoss=nn.L1Loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100,
                                                   gamma=0.5)
    print('Start training...')
    import os 
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder,exist_ok=True)
    if RESUME:
        path_checkpoint = model_folder+"{}.pth".format(start_epoch)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr']=checkpoint["lr"]*1.5
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    time_s = time.time()
    best=0
    best_epoch=0
    for epoch in range(start_epoch, end_epoch, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        psnr=[]
        psnr_train=[]
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            GT, LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            optimizer.zero_grad()  # fixed
            output_HRHSI = model(LRHSI, HRMSI)
            Pixelwise_Loss = PLoss(output_HRHSI, GT)
            Myloss = Pixelwise_Loss
            epoch_train_loss.append(Myloss.item())  # save all losses into a vector for one epoch
            Myloss.backward()  # fixed
            optimizer.step()  # fixed
            psnr_train.append(PSNR_GPU(output_HRHSI, GT).item())
            if iteration % 25 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Myloss.item()))

        lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        psnr_ = np.nanmean(np.array(psnr_train))
        print('Epoch: {}/{} \t training loss: {:.7f}\tpsnr:{:.2f}'.format(end_epoch, epoch, t_loss,psnr_ ))  # print loss for each epoch

        # ============Epoch Validate=============== #
        if epoch % ckpt_step== 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT,  LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                    output_HRHSI = model(LRHSI, HRMSI)
                    Pixelwise_Loss = PLoss(output_HRHSI, GT)
                    MyVloss = Pixelwise_Loss
                    epoch_val_loss.append(MyVloss.item())
                    psnr.append(PSNR_GPU(output_HRHSI, GT).item())
            v_loss = np.nanmean(np.array(epoch_val_loss))
            psnr = np.nanmean(np.array( psnr))

            if psnr>best:
                best=psnr
                best_epoch=epoch
                save_checkpoint(model_folder, model, optimizer, lr, epoch)

            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             PSNR loss: {:.7f}'.format(psnr ))

    return best_epoch


def meta_train(model, training_data_loader):
    # ============Epoch Train=============== #
    for iteration, batch in enumerate(training_data_loader, 1):
        GT, LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        model(GT,iteration)



def evalu_model_specific(model,opt,model_folder,test_epoch,end_epoch,dataset_name):
    import imgvision as iv
    import os
    from einops import rearrange
    from utils import SpaDown,getInputImgs,SpeDown

    patch_size = 512
    save_folder =model_folder+ dataset_name +'/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    path_checkpoint = model_folder + "{}.pth".format(test_epoch)
    checkpoint = torch.load(path_checkpoint)


    R = torch.tensor(np.load(opt.srfpath)).float().cuda()
    if dataset_name.lower()=='cave':
        lst =[1,2,7,8,9,10,11,15,19,23,26,27]
        size = 512
    elif dataset_name.lower() =='harvard':
        lst=[ 74, 62, 71, 70, 57, 66, 75, 2, 35, 34, 23, 59, 29, 65, 28, 8, 43, 63, 42, 19, 13, 33,
                             20,36, 9]
        size = 1024
 
    for ax, idx in enumerate(lst):
        idx-=1
        mx=0
        model.load_state_dict(checkpoint['net'])
        model = model.cuda()
        Spatialdown = SpaDown(opt.sf, predefine=None)
        base = sio.loadmat(f'Multispectral Image Dataset/{dataset_name}/GT.mat')
        GT = base['HSI']
        LRHSI, HRMSI, GT_ = getInputImgs(GT, dataset_name, Spatialdown,np.load(opt.srfpath))
        GT = torch.FloatTensor(GT_).T.unsqueeze(0).cuda()
        LRHSI,HRMSI = LRHSI.cuda(), HRMSI.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr = 3e-5)
        PixelLoss=nn.L1Loss()
        Spatialdown = Spatialdown.cuda()
        LR,HR,GTs=[],[],[]

        for i in range(size//patch_size):
            for j in range(size//patch_size):
                LR.append(LRHSI[:,:,i*patch_size//32:(i+1)*patch_size//32,j*patch_size//32:(j+1)*patch_size//32])
                HR.append(HRMSI[:, :, i * patch_size:(i + 1)  * patch_size ,j * patch_size:(j + 1)  * patch_size])
                GTs.append(GT[:, :, i * patch_size:(i + 1)  * patch_size, j * patch_size:(j + 1)  * patch_size])

        LR = torch.cat(LR,dim=0)
        HR = torch.cat(HR, dim=0)
        GTs= torch.cat(GTs, dim=0)

        for epoch in range(end_epoch):
            for b in range(2):
                LR_,HR_ = LR[b*2:(b+1)*2],HR[b*2:(b+1)*2]
                # print(LR_.shape,HR_.shape)
                optimizer.zero_grad()
                Pred = model(LR_,HR_)
                Predlr = Spatialdown(rearrange(Pred,'(s b) c h w -> s (b c) h w', s=1))
                Predlr = rearrange(Predlr,' s (b c) h w -> (s b) c h w ',c=LR.shape[1])
                Prehr = SpeDown(Pred,R)
                loss = PixelLoss(Predlr,LR_)+PixelLoss(Prehr,HR_)
                loss.backward()
                optimizer.step()
            if epoch%10==0:
                model.eval()
                with torch.no_grad():
                    Pred = model(LR,HR)
                pr = PSNR_GPU(GTs, Pred).detach().cpu().numpy()
                print('\r{} Epoch: {}\tLoss: {:.4f}\tPSNR: {:.3f}\t Best PSNR:{:.3f}'.format(idx,epoch,loss.detach().cpu().numpy(),pr,mx), end='')
                if pr>mx and epoch>40:
                    Re = reshuffle(Pred,patch_size)
                    # iv.spectra_metric(Re,GT_).Evaluation()
                    # plt.imshow(iv.spectra().space(Re))
                    # plt.show()
                    np.save(save_folder+str(idx), Re)
                    mx=pr
                    save_checkpoint(save_folder, model, optimizer, lr, idx)




if __name__=='__main__':

    from torch.utils.data import DataLoader

    # Build Network
    Method = 'CaFormer_3'

    model, opt = model_generator(Method,'cuda')

    dataset_name = 'CAVE'
    model_folder = Method + '/' + dataset_name + '/'

    # Training Setting
    Batch_size =4
    end_epoch = 2000
    ckpt_step = 50
    lr = 1e-4
    patch_size = 256 #Crop Ps

    # Scheme Setting
    General=False
    bestepoch=700
    meta = False
    Specific=True

    resume=False
    start=0

    if General:
        from Dataloader_tool import Large_dataset
        Train_data = Large_dataset(opt, patch_size,dataset_name ,type='train')
        Val_data = Large_dataset(opt,  patch_size,dataset_name ,type='eval')

        training_data_loader = DataLoader(dataset=Train_data, num_workers=0, batch_size=Batch_size, shuffle=True,
                                      pin_memory=True, drop_last=False)
        validate_data_loader = DataLoader(dataset=Val_data, num_workers=0, batch_size=Batch_size, shuffle=True,
                                          pin_memory=True, drop_last=True)
        if meta:
            Train_data = Large_dataset(opt, patch_size, dataset_name, type='test')

            training_data_loader = DataLoader(dataset=Train_data, num_workers=0, batch_size=Batch_size, shuffle=True,
                                              pin_memory=True, drop_last=False)

            meta_train(model, training_data_loader)

        else:
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            bestepoch = train(model,training_data_loader,validate_data_loader,model_folder=model_folder,optimizer=optimizer,lr=lr,start_epoch=start,end_epoch=end_epoch,ckpt_step=ckpt_step,RESUME=resume,meta = meta)

    if Specific:
        evalu_model_specific(model, opt, model_folder, bestepoch, end_epoch, dataset_name)

