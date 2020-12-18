from data_loader import DataLoader
import matplotlib.pyplot as plt

'''
unit test for dataloader and noise simulator
'''

datadir='C:\\travail\\dataset\\UCMerced_LandUse\\Images'

datalo=DataLoader(datadir)

### test for load_data

# imgs_n, imgs_o = datalo.load_data(batch_size=3, is_testing=True)


# imgs_n =0.5 * imgs_n + 0.5
# imgs_o=0.5 * imgs_o + 0.5
# print('sample',imgs_n.shape,imgs_o.shape)
# gen_imgs = [imgs_n,imgs_o]

# titles = ['n', 'o']
# r, c = 3,2
# fig, axs = plt.subplots(r, c, figsize=(20, 30))
# for i in range(r): #batch
#     for j in range(c):
#         if j==0:
#             axs[i,j].imshow(gen_imgs[j][i][:,:,0]*2,cmap='gray')
#         else:
#             axs[i,j].imshow(gen_imgs[j][i][:,:,0],cmap='gray')
#         axs[i, j].set_title(titles[j])
#         axs[i,j].axis('off')
# fig.savefig("testload.png")
# plt.close()
        
### test for load_batch
batch_i=0
for (imgs_n, imgs_o) in datalo.load_batch(batch_size=3, is_testing=False):
    batch_i+=1
    imgs_n =0.5 * imgs_n + 0.5
    imgs_o=0.5 * imgs_o + 0.5
    print('sample',imgs_n.shape,imgs_o.shape)
    gen_imgs = [imgs_n,imgs_o]
    titles = ['n', 'o']
    r, c = 3,2
    fig, axs = plt.subplots(r, c, figsize=(20, 30))
    for i in range(r): #batch
        for j in range(c):
            axs[i,j].imshow(gen_imgs[j][i][:,:,0],cmap='gray')
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
    fig.savefig("testload/testload%d.png" % batch_i)
    plt.close()
    if batch_i>20:
        break