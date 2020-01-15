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
            img_n = noise_simul(img_o)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_o = np.fliplr(img_o)
                img_n = np.fliplr(img_n)

            imgs_o.append(img_o)
            imgs_n.append(img_n)

        imgs_o = np.array(imgs_o)/127.5 - 1.
        imgs_n = np.array(imgs_n)/127.5 - 1.
        return imgs_n,imgs_o

    def load_batch(self, batch_size=1, is_testing=False): #训练用
        data_type = "train" if not is_testing else "test"
        path_sar = os.path.join('/home1/dataset/sar-op/subset_spring_90/',data_type,'sar','s1_90')
        sar_list=os.listdir(path_sar)
        sar_list.sort()
        path_op=os.path.join('/home1/dataset/sar-op/subset_spring_90/',data_type,'op','s2_90')
        op_list=os.listdir(path_op)
        op_list.sort()
        self.n_batches = int(len(sar_list) / batch_size)

        for i in range(self.n_batches-1):
            batch_sar = sar_list[i*batch_size:(i+1)*batch_size]
            batch_op = op_list[i*batch_size:(i+1)*batch_size]
            imgs_sar, imgs_op = [], []
            for (img_sar,img_op) in zip(batch_sar,batch_op):
                img_sar = img_to_array(load_img(os.path.join(path_sar,img_sar),color_mode='grayscale'))
                img_op=img_to_array(load_img(os.path.join(path_op,img_op),color_mode='rgb'))


                if not is_testing and np.random.random() > 0.5:
                        img_sar = np.fliplr(img_sar)
                        img_op = np.fliplr(img_op)

                imgs_sar.append(img_sar)
                imgs_op.append(img_op)

            imgs_sar = np.array(imgs_sar)/127.5 - 1.
            imgs_op = np.array(imgs_op)/127.5 - 1.
            # print(imgs_sar.shape)
            yield imgs_op,imgs_sar


    # def imread(self, path):
    #     return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    os.makedirs('test_dataloder/', exist_ok=True)
    r, c = 3,2
    datalo=DataLoader()
    imgs_A, imgs_B = datalo.load_data(batch_size=3, is_testing=True)

    # gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5
    imgs_B =0.5 * imgs_B + 0.5
    imgs_A=0.5 * imgs_A + 0.5
    print('sample',imgs_B.shape,imgs_A.shape)
    gen_imgs = [imgs_B,imgs_A]

    titles = ['Condition', 'Original']
    fig, axs = plt.subplots(r, c)
    for i in range(r): #batch
        for j in range(c):
            if j ==0:
                axs[i,j].imshow(gen_imgs[j][i][:,:,0],cmap='gray')
            else:
                axs[i,j].imshow(gen_imgs[j][i])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
    fig.savefig("test_dataloder/testload.png")
    plt.close()
        