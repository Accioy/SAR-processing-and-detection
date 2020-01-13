import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
import numpy as np
import time
import random
from numba import jit, float64

## parameters, load an image ############

# number of looks
global L
L=4.0

# img name
img_name='airplane45.tif'

# img resolution
img = plt.imread(img_name)
res = img.shape
res = res[0:2]

# max of simulated noise
max_n=5

## speckle probability density function ##########

global alpha
alpha=(1/gamma(L))*(L**L)
@jit(float64(float64))
def p_function(F):
    return alpha*(F**(L-1))*(np.exp(-L*F))

r=[]
f1=np.linspace(0,max_n,1024)
for i in f1:
    r1=p_function(i)
    r.append(r1)

max_r=max(r)
tic=time.time()
## Generate the noise sequence of the probability density function ###################

# 方法1 ######################
# n=0
# N=res[0]*res[1]
# noises=[]
# while n<N:
#     t=random.uniform(0,max_n)     #生成[0,max_n]均匀分布随机数
#     pt=p(t,L)             #计算对应密度函数值f(t)
#     rt=random.uniform(0,max(r))   #生成[0,m]均匀分布随机数，m取概率密度函数的上确界
#     if rt<=pt:           #如果随机数r小于f(t)，接纳该t并加入序列noises中
#         n=n+1
#         noises.append(t)

# 优化 ############################
length=300000

N=res[0]*res[1]
noises=np.array([])
# LL=[L for i in range(length)]

# while len(noises)<N:
tn=np.random.rand(length)*max_n
ptn=np.array(list(map(p_function,tn))) #p: density function
rtn=np.random.rand(length)*max_r
noises=np.append(noises,tn[np.where(rtn<=ptn)])


toc=time.time()
print(toc-tic)
print(noises.shape)
## compare the sequence distribution with the probability density function ##

s=quad(p_function,0,np.inf)
print(s)

plt.plot(f1,r,'r')
plt.hist(noises,bins=100,density=True)
plt.show()

## Image pre-process ######
# img = plt.rgb2gray(img)
# img = plt.im2double(img)
# ## add noise #############
# noises=noises/max(noises)
# noises=np.reshape(noises,res)
# img_with_speck = img*noises
# plt.imshow(img_with_speck*2)
# plt.show()
