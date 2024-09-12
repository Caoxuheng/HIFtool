class ZSL():
    def __init__(self,args):
        self.args = args
        self.step=0
        self.model = ZSL_cnn(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.equip_degradation(args)
    def equip_degradation(self,args):
        pass


    def get_traindata(self,lhsi,hmsi,U):
        augument = [0]
        HSI_aug = []
        MSI_aug = []
        HSI_aug.append(lhsi)
        MSI_aug.append(hmsi)
        train_hrhs,train_hrms,train_lrhs=[],[],[]

        for j in augument:
            HSI = cv2.flip(lhsi, j)
            HSI_aug.append(HSI)
        for j in range(len(HSI_aug)):
            HSI = HSI_aug[j]
            HSI_Abun = np.tensordot(U.T, HSI, axes=([1], [0]))
            HSI_LR_Abun = self.spadown(HSI_Abun)
            MSI_LR = self.spedown(HSI)

            for j in range(0, HSI_Abun.shape[1] - self.args.args.training_size + 1, 1):
                for k in range(0, HSI_Abun.shape[2] - self.args.training_size + 1, 1):
                    temp_hrhs = HSI[:, j:j + self.args.training_size, k:k + self.args.training_size]
                    temp_hrms = MSI_LR[:, j:j + self.args.training_size, k:k + self.args.training_size]
                    temp_lrhs = HSI_LR_Abun[:, int(j / self.args.sf):int((j + self.args.training_size) / self.args.sf),
                                int(k / self.args.sf):int((k + self.args.training_size) / self.args.sf)]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)

            train_hrhs = torch.Tensor(train_hrhs).cuda()
            train_lrhs = torch.Tensor(train_lrhs).cuda()
            train_hrms = torch.Tensor(train_hrms).cuda()
            train_data = HSI_MSI_Data(train_hrhs, train_hrms, train_lrhs)
            self.train_loader = data.DataLoader(dataset=train_data, batch_size=self.args.BATCH_SIZE, shuffle=True)


    def __call__(self, lhsi, hmsi, GT):

        lhsi_np = lhsi[0].detach().cpu().numpy()
        U0, S, V = np.linalg.svd(np.dot(lhsi_np, lhsi_np.T))
        U0 = U0[:, 0:int(p)]
        U = U0
        U22 = torch.Tensor(U0)
        L1loss = nn.L1Loss()

        maxiteration = 2 * math.ceil(((GT.shape[1] / self.args.sf - self.args.training_size) // 1 + 1) * (
                    (GT.shape[2] / self.args.sf - self.args.training_size) // 1 + 1) / self.args.BATCH_SIZE) * 400

        warm_iter = np.floor(maxiteration / 40)
        for step in range(400):
            for step1, (lhsi_data, hmsi_data, a3) in enumerate(self.train_loader):
                lr = warm_lr_scheduler(self.optimizer, self.args.init_lr1, self.args.init_lr2, warm_iter, step, lr_decay_iter=1, max_iter=maxiteration,
                                   power= self.args.decay_power)

                output, Xre = self.model(lhsi_data, hmsi_data,U22)
                loss = L1loss(self.spadown(Xre),lhsi_data)+L1loss(self.spedown(Xre),hmsi_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return  Xre[0].detach().cpu().numpy().T
