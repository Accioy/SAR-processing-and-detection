import time, threading

# 新线程执行的代码:

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

def loop():
    print('thread %s is running...' % threading.current_thread().name)
    a=0
    a+=3
    print('thread %s ended.' % threading.current_thread().name)
    return a
print('thread %s is running...' % threading.current_thread().name)
t1 = MyThread(loop,name='thread1')
t2 = MyThread(loop,name='thread2')
t1.start()
t2.start()
t1.join()
t2.join()
r = t1.get_result()+t2.get_result()
print('thread %s ended.' % threading.current_thread().name)
print(t1.get_result(),t2.get_result(),r)