# return results with queue

import multiprocessing as mp
import time
def foo(i,q):
    time.sleep(10/(i+1))
    q.put('hello'+str(i))
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    res=[]
    for i in range(10):
        p=mp.Process(target=foo,args=(i,q))
        res.append(p)
        p.start()

    for p in res:
        print(q.get())
    for p in res:
        p.join()
