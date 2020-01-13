# use pool to create subprocesses
# use return to return the result of a process

from multiprocessing import Process, Queue, set_start_method, Pool
import os, time, random

def task(i):
    time.sleep(i/1000)
    return i
    


if __name__=='__main__':
    set_start_method('spawn',True)

    my_list=[12,345,9,3,24,123,23,43,545,1231,234,5432,452]
    L=len(my_list)
    p=Pool(L)
    res=[]
    for i in range(L):
        res.append(p.apply_async(task, args=(my_list[i],)))
    print('waiting...')
    p.close()
    p.join()
    for r in res:
        print(r.get())
    