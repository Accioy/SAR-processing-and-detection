from data_loader import noise_simul
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np

img_name='airplane45.tif'
img = img_to_array(load_img(img_name,color_mode='grayscale'))
noise_simul=noise_simul(img)
img_n=noise_simul()
img_n=img_n[:,:,0]
img_n=np.round(img_n)
img_n=img_n.astype(np.int32)
plt.imshow(img_n,cmap='gray')
plt.show()
plt.savefig('noised.jpg')

plt.imshow(img_n*2,cmap='gray')
plt.show()
plt.savefig('noised2.jpg')

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The results are the same.
# I think it is because the float value are all mapped to 0-1 whatever the original range is.