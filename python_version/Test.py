### Testing LassoFISTA in python
### Jeremy L Hour
### 21 juin 2016


### Load libraries
import os
import time
import numpy as np
from scipy.linalg import inv, solve, det
import math
import scipy.stats

os.chdir("/home/coder/LassoFISTA/python_version/")

### Load user-defined func
from LassoFISTA import LassoFISTA, prox, LeastSq, LeastSqgrad, LassoObj
from DataSim import DataSim


### 1. Test of DataSim code
X, y, d, b0, gamma0 = DataSim(n=100,p=50,Ry=.5,Rd=.2,rho=.5)
beta_hat = np.linalg.lstsq(X,y)[0] # solution with py func
beta_hatman = inv( X.T.dot(X) ).dot(X.T.dot(y)) # manual solution

### Test on Lalonde dataset
f = open("/home/coder/LassoFISTA/dataset/d.txt","r")
header1 = f.readline()
d = []
for line in f:
  line = line.strip()
  columns = line.split()
  d.append(int(columns[1]))
f.close()
d=np.array(d)

f = open("/home/coder/LassoFISTA/dataset/y.txt","r")
header1 = f.readline()
y = []
for line in f:
  line = line.strip()
  columns = line.split()
  y.append(float(columns[1]))
f.close()
y=np.array(y)
y=y.astype(float)
  
f = open("/home/coder/LassoFISTA/dataset/X.txt","r")
header1 = f.readline()
X = []
for line in f:
  line = line.strip()
  columns = line.split()
  X.append(np.delete(columns,0))
f.close()
X=np.array(X)
X=X.astype(float)

n, p = X.shape
c, g = 2, .1/np.log(max(n,p))
Lambda= c*scipy.stats.norm.ppf(1-.5*g/p)/np.sqrt(n)

start_time = time.time()
betaL, valL, lossL, l1normL, nbIterL, cvFL = LassoFISTA(y,X,Lambda,nopen=np.array([0]),tol=1e-6,maxIter=1e6,trace=True)
print("--- %s seconds ---" % (time.time() - start_time))
