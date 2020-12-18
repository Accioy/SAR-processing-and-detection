import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
filename = 'nohup_old.out'
p=[]
with open(filename,'r') as f:
    while True:
        l=f.readline()
        if not l:
            break
        if l[:4]=='mAP:':
            p.append(float(l[4:].strip()))
print(len(p))
print(p)
x = np.arange(len(p))
xnew = np.linspace(0,len(p),int(len(p)*0.3))
smooth = make_interp_spline(x,p)(xnew)
plt.plot(x,p,xnew,smooth)
plt.show()
