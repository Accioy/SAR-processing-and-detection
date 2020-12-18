import cv2
import os
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from data_loader import DataLoader
import scipy.misc
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model=load_model('generator195.h5')
datadir="sar_test/"
test_dataloader=DataLoader(datadir,img_res=(256, 256))
imgs_sar = test_dataloader.load_sar()
for i in range(0,len(imgs_sar),4):
    #batch=imgs_n[i:i+4]
    batch=imgs_sar[i:i+4]
    fake_A = model.predict_on_batch(batch)
    fake_A = fake_A*0.5+0.5
    for j in range(len(fake_A)):
        # scipy.misc.imsave('out/'+str(i)+str(j)+'.jpg', fake_A[i])
        cv2.imwrite('out/'+str(i)+str(j)+'.jpg',fake_A[j]*255)