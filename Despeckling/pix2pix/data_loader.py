# -*- coding: utf-8 -*-
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from scipy.special import gamma
from scipy.integrate import quad
from numba import jit, float64

@jit(float64(float64,float64,float64))
def _p_function(alpha, L, F):
    return alpha*(F**(L-1))*(np.exp(-L*F))

class noise_simul():
    def __init__(self, img, L=4):
        '''
        img: numpy array, H*W*1
        !!!
        Parameter need to be considered: L, max_n, length
        '''
        self.img = img
        self.img_size=img.size
        self.img_shape=img.shape
        self.L = L
        self.alpha = (1/gamma(L))*(L**L)
        self.max_n = 5 # maximum of simulated SAR noise，p_function(max_n) should be nearly 0
        self.s = self.integral() #integration of pdf, should be nearly 1
        self.max_r = self.max_p()
        self.length =300000 # L=4的时候，这个长度的随机仿真可以满足65536的随机序列需求。如果调整L值，这个值也需要调整
    
    def p_function(self,F):
        return _p_function(self.alpha, self.L, F)
    def max_p(self):
        f1=np.linspace(0,self.max_n,1000)
        r=list(map(self.p_function,f1))
        return max(r)
    def integral(self):
        s=quad(self.p_function,0,self.max_n)
        return s[0]
    def simul_noise(self):
        noises=np.array([])
        while noises.size<self.img_size:
            tn=np.random.rand(self.length)*self.max_n
            ptn=np.array(list(map(self.p_function,tn)))
            rtn=np.random.rand(self.length)*self.max_r
            noises=np.append(noises,tn[np.where(rtn<=ptn)])
        noises=noises[:self.img_size]
        noises=noises.reshape(self.img_shape)
        return noises

    def __call__(self):
        # img = plt.rgb2gray(self.img) # 理论上这里调用的时候已经gray了
        noises=self.simul_noise()
        noises=noises/np.max(noises)
        img_with_speck = self.img*noises
        return img_with_speck
        

class DataLoader():
    def __init__(self, dataset_dir, img_res=(256, 256)):
        self.img_res = img_res
        self.dataset_dir = dataset_dir
        
    def load_data(self, batch_size=3, is_testing=False): #测试用，随机采样一组数据返回
        data_type = "train" if not is_testing else "test"
        listfile = data_type+'set.txt'
        image_names = [l.strip().split(None, 1)[0] for l in open(os.path.join(self.dataset_dir, listfile)).readlines()]
        # image_names.sort() # 理论上txt文件里已经sort过了

        batch_images = np.random.choice(image_names, size=batch_size)
        

        imgs_o = [] #original image
        imgs_n = [] #noised image
        
        for img_name in batch_images:
            img_o = img_to_array(load_img(os.path.join(self.dataset_dir,img_name),color_mode='grayscale'))
            noise_simulation=noise_simul(img_o)
            img_n=noise_simulation()
            # img_n=img_n[:,:,0]

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_o = np.fliplr(img_o)
                img_n = np.fliplr(img_n)

            imgs_o.append(img_o)
            imgs_n.append(img_n)
        # 归一化
        imgs_o = np.array(imgs_o)/127.5 - 1.
        imgs_n = np.array(imgs_n)/127.5 - 1.
        return imgs_n,imgs_o

    def load_batch(self, batch_size=1, is_testing=False): #训练用
        data_type = "train" if not is_testing else "test"
        listfile = data_type+'set.txt'
        image_names = [l.strip().split(None, 1)[0] for l in open(os.path.join(self.dataset_dir, listfile)).readlines()]
        np.random.shuffle(image_names)
        self.n_batches = int(len(image_names) / batch_size)

        for i in range(self.n_batches-1):
            batch_o = image_names[i*batch_size:(i+1)*batch_size]
            imgs_o, imgs_n = [], []
            for img_o in batch_o:
                img_o = img_to_array(load_img(os.path.join(self.dataset_dir,img_o),color_mode='grayscale'))
                noise_simulation=noise_simul(img_o)
                img_n=noise_simulation()

                if not is_testing and np.random.random() > 0.5:
                        img_o = np.fliplr(img_o)
                        img_n = np.fliplr(img_n)

                imgs_o.append(img_o)
                imgs_n.append(img_n)

            imgs_o = np.array(imgs_o)/127.5 - 1.
            imgs_n = np.array(imgs_n)/127.5 - 1.
            # print(imgs_sar.shape)
            yield imgs_n,imgs_o

