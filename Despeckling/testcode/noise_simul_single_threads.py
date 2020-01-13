import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
import numpy as np
import time
import random
import threading
## parameters, load an image ############
# number of looks
L=4.0
# img name
img_name='airplane45.tif'
# img resolution
img = plt.imread(img_name)
res = img.shape
res = res[0:2]
# max of simulated noise
global max_n
max_n=5

## speckle probability density function ##########

def p(F,L):
    return (1/gamma(L))*(L**L)*(F**(L-1))*(np.exp(-L*F))

r=[]
f1=np.linspace(0,max_n,1024)
for i in f1:
    r1=p(i,L)
    r.append(r1)

global max_r
max_r=max(r)
tic=time.time()
## Generate the noise sequence of the probability density function ###################

global lengh
lengh=10000

N=res[0]*res[1]
noises=np.array([])
global LL
LL=[L for i in range(lengh)]

class MyThread(threading.Thread):
    def __init__(self, func, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        
    def run(self):
        self.result = self.func()

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def func_for_multithread():
    tn=np.random.rand(lengh)*max_n
    ptn=np.array(list(map(p,tn,LL))) #p: density function
    rtn=np.random.rand(lengh)*max_r
    result=tn[np.where(rtn<=ptn)]
    # print('thread %s ended.' % threading.current_thread().name)
    return result

threads = []
for i in range(30):
    t=MyThread(func_for_multithread,name='th '+str(i))
    threads.append(t)
for i in range(30):
    threads[i].start()
for i in range(30):
    threads[i].join()

noises=np.concatenate([threads[i].get_result() for i in range(30)])


toc=time.time()
print(noises.shape)
## compare the sequence distribution with the probability density function ##

s=quad(p,0,np.inf,args=(L))
print(s)
print(toc-tic)
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
