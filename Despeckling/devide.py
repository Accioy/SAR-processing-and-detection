from PIL import Image
import numpy as np
import os



root=os.path.abspath('./')
f_list_sar=os.listdir('sar/')
f_list_op=os.listdir('op/')
f_list_sar.sort()
f_list_op.sort()

sar_images=[]
op_images=[]
for f,g in zip(f_list_sar,f_list_op):
    sar_image=Image.open('sar/'+f)
    sar_images.append(sar_image)
    op_image=Image.open('op/'+g)
    op_image=op_image.convert('L')
    op_image.show()
    op_images.append(op_image)
    

for s,o in zip(sar_images,op_images):
    s_np=np.asarray(s)
    o_np=np.asarray(o)
    noise=s_np/o_np
    noise_im=Image.fromarray(noise)
    noise_im.show()



