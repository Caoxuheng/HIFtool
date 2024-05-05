import matplotlib.pyplot as plt
from h5py import File
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import torch
import cv2
def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data

class WDCMDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=3*1, patch_w=3, h_stride=1, w_stride=1, ratio=32, type='train'):
        super(WDCMDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel = 191
        self.msi_channel = 13
        self.rows = 32
        self.cols = 9
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=patch_h, patch_w=patch_w, ratio=ratio)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=patch_h, patch_w=patch_w, ratio=ratio)

    def getData(self):
        srf = sio.loadmat('Dataloader_tool/srflib/Sentinel2A_191_13.mat')['R']
        hrhsi = sio.loadmat(self.mat_save_path )['S'][:40*32,:32*9]

        hrhsi = normalize(hrhsi)
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio-1,self.ratio-1],sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666,borderType=cv2.BORDER_REPLICATE)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi@ srf

        return hrhsi, lrhsi, hrmsi

    def generateTrain(self, patch_h, patch_w, ratio):
        num = len(list(range(0, self.rows - patch_h, self.h_stride))) * len(
            list(range(0, self.cols - patch_w, self.w_stride)))

        label_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = hrhsi[:self.rows*ratio, :, :]
        lrhsi = lrhsi[:self.rows, :, :]
        hrmsi = hrmsi[:self.rows*ratio, :, :]


        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h , self.h_stride):
            for y in range(0, self.cols - patch_w , self.w_stride):
                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 191), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 13), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 191), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = hrhsi[self.rows * ratio:(self.rows+patch_h) * ratio, :patch_w*ratio, :]
        lrhsi = lrhsi[self.rows:self.rows+patch_h, :patch_w, :]
        hrmsi = hrmsi[self.rows * ratio:(self.rows+patch_h) * ratio, :patch_w*ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 191), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 13), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 191), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData()

        hrhsi = hrhsi[(self.rows + patch_h) * ratio:, :128, :]
        lrhsi = lrhsi[self.rows + patch_h:,:4]
        hrmsi = hrmsi[(self.rows + patch_h) * ratio:, :128, :]

        # sio.savemat('WDCM_test.mat',{'GT':hrhsi,'HSI':lrhsi,'MSI':hrmsi})
        # print('save')
        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        index =index//2
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]*2

class ChikuseiDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=4*1, patch_w=4*1, h_stride=2, w_stride=2, ratio=32, type='train'):
        super(ChikuseiDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel=128
        self.msi_channel=4
        self.rows = 41
        self.cols = 41
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if type !='test':
            self.hrhsi, self.lrhsi, self.hrmsi = self.getData()

        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio, s_h=8, s_w=8)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=self.patch_h, patch_w=self.patch_w, ratio=ratio)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=16*1, patch_w=16*1, ratio=ratio)

    def getData(self):
        srf = sio.loadmat('Dataloader_tool/srflib/chikusei_128_4.mat')['R']
        hrhsi = np.array(File(self.mat_save_path )['chikusei']).transpose([1,2,0])
        hrhsi[300:812,300:812] =hrhsi[812:1324,812:1324]
        # hrhsi = np.random.rand(2335, 2517, 128)

        # (2335, 2517, 128)
        # hrhsi = normalize(hrhsi)
        hrhsi /= hrhsi.max()
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi@ srf

        return hrhsi, lrhsi, hrmsi

    def generateTrain(self, patch_h, patch_w, ratio, s_h, s_w):
        # print(len(list(range(0, self.rows - patch_h , self.h_stride) )))
        # print(len(list(range(0, self.cols - patch_w , self.w_stride))))
        num = len(list(range(0, self.rows - patch_h , self.h_stride) ))*len(list(range(0, self.cols - patch_w , self.w_stride)))
        # print(num)
        label_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((num , patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        # hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = self.hrhsi[:self.rows*ratio, :, :]
        lrhsi = self.lrhsi[:self.rows, :, :]
        hrmsi = self.hrmsi[:self.rows*ratio, :, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h , self.h_stride):
            for y in range(0, self.cols - patch_w , self.w_stride):

                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        # hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = self.hrhsi[self.rows * ratio:(self.rows+patch_w) * ratio, :patch_w*ratio, :]
        lrhsi = self.lrhsi[self.rows:self.rows+patch_w, :patch_w, :]
        hrmsi = self.hrmsi[self.rows * ratio:(self.rows+patch_w) * ratio, :patch_w*ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0
        # double(HSI), double(MSI), F, par.fft_B, downsampling_scale, S, para, 15

        srf = sio.loadmat('Dataloader_tool/srflib/chikusei_128_4.mat')['R']
        hrhsi = np.load('Chikusei/GT.npy')
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi, ksize=[self.ratio//2 * 2 + 1] * 2, sigmaX=self.ratio * 0.666,
                                 sigmaY=self.ratio * 0.666)[self.ratio // 2::self.ratio, self.ratio // 2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi @ srf

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]

class PaviaDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=3*1, patch_w=3*1, h_stride=1, w_stride=1, ratio=32, type='train'):
        super(PaviaDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel=93
        self.msi_channel=4
        self.rows = 13
        self.cols =13
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type =='test':
            self.hrhsi, self.lrhsi, self.hrmsi = self.getData_()
        else:
            print('train')
            self.hrhsi, self.lrhsi, self.hrmsi = self.getData()
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio, s_h=8, s_w=8)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=self.patch_h, patch_w=self.patch_w, ratio=ratio)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=10*1, patch_w=10*1, ratio=ratio)

    def getData(self):
        srf = sio.loadmat('Dataloader_tool/srflib/chikusei_srf.mat')['R']
        hrhsi = sio.loadmat(self.mat_save_path)['GT']

        # hrhsi = np.random.rand(2335, 2517, 128)

        # (2335, 2517, 128)
        # hrhsi = normalize(hrhsi)
        hrhsi /= hrhsi.max()
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi@ srf

        return hrhsi, lrhsi, hrmsi
    def getData_(self):
        srf = sio.loadmat('Dataloader_tool/srflib/chikusei_srf.mat')['R']
        hrhsi = sio.loadmat(self.mat_save_path )['HSI']
        import  cv2
        hrhsi = cv2.copyMakeBorder(hrhsi,0,64,0,64,cv2.BORDER_REFLECT)
        # hrhsi = np.random.rand(2335, 2517, 128)

        # (2335, 2517, 128)
        # hrhsi = normalize(hrhsi)
        # hrhsi /= hrhsi.max()
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi@ srf

        return hrhsi, lrhsi, hrmsi
    def generateTrain(self, patch_h, patch_w, ratio, s_h, s_w):
        num = len(list(range(0, self.rows - patch_h , self.h_stride) ))*len(list(range(0, self.cols - patch_w , self.w_stride)))
        # print(num)
        label_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((num , patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        # hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = self.hrhsi[:self.rows*ratio, :, :]
        lrhsi = self.lrhsi[:self.rows, :, :]
        hrmsi = self.hrmsi[:self.rows*ratio, :, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h , self.h_stride):
            for y in range(0, self.cols - patch_w , self.w_stride):

                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0


        hrhsi = self.hrhsi[self.rows * ratio:(self.rows+patch_w) * ratio, :patch_w*ratio, :]
        lrhsi = self.lrhsi[self.rows:self.rows+patch_w, :patch_w, :]
        hrmsi = self.hrmsi[self.rows * ratio:(self.rows+patch_w) * ratio, :patch_w*ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData_()

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch = hrhsi[None]
        hrmsi_patch = hrmsi[None]
        lrhsi_patch = lrhsi[None]

        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        index =index//2
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]*2

class XionganDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=4*1, patch_w=4, h_stride=1, w_stride=1, ratio=32, type='train'):
        super(XionganDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel = 93
        self.msi_channel = 4
        self.rows = 60
        self.cols = 32
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=patch_h, patch_w=patch_w, ratio=ratio)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=patch_h, patch_w=patch_w, ratio=ratio)

    def getData(self):
        srf = sio.loadmat('Dataloader_tool/srflib/chikuseisrf.mat')['R']
        hrhsi = np.load(self.mat_save_path )[:117*32,:32*49]

        # hrhsi = normalize(hrhsi)
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio-1,self.ratio-1],sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666,borderType=cv2.BORDER_REPLICATE)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi@ srf

        return hrhsi, lrhsi, hrmsi

    def getData_(self):
        srf = sio.loadmat('Dataloader_tool/srflib/chikuseisrf.mat')['R']
        hrhsi = np.load(self.mat_save_path)[1750:1750+512,650:650+512]

        # hrhsi = normalize(hrhsi)
        #  Generate LRHSI
        lrhsi = cv2.GaussianBlur(hrhsi,ksize=[self.ratio-1,self.ratio-1],sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666,borderType=cv2.BORDER_REPLICATE)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
        #  Generate HRMSI
        hrmsi = hrhsi@ srf

        return hrhsi, lrhsi, hrmsi
    def generateTrain(self, patch_h, patch_w, ratio):
        num = len(list(range(0, self.rows - patch_h, self.h_stride))) * len(
            list(range(0, self.cols - patch_w, self.w_stride)))

        label_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((num, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = hrhsi[:self.rows*ratio, :, :]
        lrhsi = lrhsi[:self.rows, :, :]
        hrmsi = hrmsi[:self.rows*ratio, :, :]


        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h , self.h_stride):
            for y in range(0, self.cols - patch_w , self.w_stride):
                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.hsi_channel), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, self.msi_channel), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData()
        hrhsi = hrhsi[self.rows * ratio:(self.rows+patch_h) * ratio, :patch_w*ratio, :]
        lrhsi = lrhsi[self.rows:self.rows+patch_h, :patch_w, :]
        hrmsi = hrmsi[self.rows * ratio:(self.rows+patch_h) * ratio, :patch_w*ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio):
        hrhsi, lrhsi, hrmsi = self.getData_()
        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        return lrhsi[None], hrmsi[None], hrhsi[None]

    def __getitem__(self, index):
        index =index
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]

class QBDataset(Dataset):
    def __init__(self, mat_save_path,ratio=4, type='train'):
        super(QBDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.hsi_channel = 4
        self.msi_channel = 1

        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain()
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateTrain()
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest()

    def getData1(self):
        from h5py import File
        hrhsi=File(self.mat_save_path)


        return  np.array(hrhsi["lms"]),np.array(hrhsi["ms"]), np.array(hrhsi["pan"])
    def getData(self):
        try:
            hrhsi = sio.loadmat(self.mat_save_path)
            return  hrhsi["lms"],hrhsi["ms"], hrhsi["pan"]
        except:
            from h5py import File
            hrhsi = File(self.mat_save_path)

            return  np.array(hrhsi["lms"]).T,np.array(hrhsi["ms"]).T, np.array(hrhsi["pan"]).T
    def generateTrain(self):
        hrhsi, lrhsi, hrmsi = self.getData1()

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        return lrhsi/lrhsi.max(), hrmsi/hrmsi.max(), hrhsi/hrhsi.max()



    def generateTest(self):
        hrhsi, lrhsi, hrmsi = self.getData()

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        return lrhsi[None],  hrmsi[None,:,:,None],hrhsi[None]

    def __getitem__(self, index):

        if self.type =='test':
            hrhsi = np.transpose(self.label[index], (2,0,1))
            hrmsi= np.transpose(self.msi_data[index], (2,0,1))
            lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        else:
            hrhsi = self.label[index]
            hrmsi = self.msi_data[index]
            lrhsi = self.hsi_data[index]
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]
