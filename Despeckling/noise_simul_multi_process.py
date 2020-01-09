import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
import numpy as np
import time
import random
from multiprocessing import Process, Queue, set_start_method, Pool

def p(F,L):
    return (1/gamma(L))*(L**L)*(F**(L-1))*(np.exp(-L*F))

def func_for_multiprocess(length,max_n,max_r,LL,q):
    tn=np.random.rand(length)*max_n

    ptn=np.array(list(map(p,tn,LL))) #p: density function
    rtn=np.random.rand(length)*max_r
    result=tn[np.where(rtn<=ptn)]
    q.put(result)
    # print('fin')



if __name__=='__main__':
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
    max_n=5

    ## speckle probability density function ##########

    r=[]
    f1=np.linspace(0,max_n,1024)
    for i in f1:
        r1=p(i,L)
        r.append(r1)

    max_r=max(r)
    tic=time.time()
 

    # 继续优化版本 ##########################
    length=10000

    N=res[0]*res[1]
    noises=np.array([])
    LL=[L for i in range(length)]

    # set_start_method('spawn',True)
    q = Queue()
    res=[]
    for i in range(3):
        p=Process(target=func_for_multiprocess, args=(length,max_n,max_r,LL,q))
        res.append(p)
        
    for p in res:
        p.start()
    for p in res:
        p.join()
    print('waiting...')
    noises=np.concatenate([q.get() for p in res])



    toc=time.time()
    print(noises.shape)
    ## compare the sequence distribution with the probability density function ##

    # s=quad(p,0,np.inf,args=(L))
    # print(s)
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
