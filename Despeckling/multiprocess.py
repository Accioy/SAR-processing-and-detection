from multiprocessing import Process, Queue
import os, time, random


def write(i,q):
    time.sleep(i/1000)
    q.put(i)
    


if __name__=='__main__':
    my_list=[12,345,9,3,24,123,23,43,545,12431,234,5432,452]
    q = Queue()
    L=len(my_list)
    jobs=[]
    for i in range(L):
        p = Process(target=write, args=(my_list[i],q,))
        jobs.append(p)
    for p in jobs:
        p.start()
    for p in jobs:
        p.join()
    r = [q.get() for i in range(L)]
    print(r)