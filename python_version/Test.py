import numpy as np
from scipy.linalg import inv, solve, det
import random as rd
import math
import scipy.stats

def DataSim(n=2000,p=50,Ry=.5,Rd=.2,rho=.5):
  ### Covariate correlation coefficients
  Sigma = np.zeros(p*p)
  Sigma.shape = (p,p)
  
  for k in range(0,p):
    for j in range(0,p):
      Sigma[k,j] = rho**abs(k-j)

  ### Treatment effect
  a = 0
  
  ### Treatment variable coefficient
  gamma = np.zeros(p)
    
  for j in range(1,math.floor(p/2)):
    gamma[j] = 1*(-1)**(j) / j**2
    
  ### Outcome equation coefficients
  b = gamma
    
  for j in range(math.floor(p/2)+1,p):
    b[j] = (-1)**(j+1) / (p-j+1)**2
  
  ### Adjustment to match R.squared
  c = math.sqrt((1/gamma.dot(Sigma).dot(gamma))*(Rd/(1-Rd)))
  gamma = c*gamma
  
  c = math.sqrt((1/b.dot(Sigma).dot(b))*(Ry/(1-Ry)))
  b = c*b

  
  # Simulate covariates
  X = np.random.multivariate_normal(np.zeros(p), Sigma, n)
  
  # Simulate treatment
  d = np.random.uniform(size=n) < 1/(1+np.exp(-X.dot(gamma)))
  d = d.astype(int)
  
  # Simulate outcome
  y = a*d + X.dot(b) + np.random.normal(0,1,n)

  # Add the intercept
  X = np.c_[ np.ones((n,1)), X ]
  
  return X, y, d, b, gamma

### Running the code

X, y, d, b0, gamma0 = DataSim(n=100,p=50,Ry=.5,Rd=.2,rho=.5)
beta_hat = np.linalg.lstsq(X,y)[0] # solution with py func
beta_hatman = inv( X.T.dot(X) ).dot(X.T.dot(y)) # manual solution


### FISTA algo

#################################
#################################
### Define auxiliary functions###
#################################
#################################

def prox(x,delta,nopen):
  # nopen is a vector of indices, starts from 0
  y = np.maximum(abs(x)-delta,np.zeros(x.shape[0])) * np.sign(x)
  y[nopen] = x[nopen] # Do not penalize these variables
  return y

def LeastSq(mu,y,X):
  f = y - X.dot(mu)
  return np.mean(f**2)

def LeastSqgrad(mu,y,X):
  df = -2*(y - X.dot(mu)).dot(X) / X.shape[0]
  return df

def LassoObj(beta,y,X,delta,nopen):
  if nopen.shape[0]>0:
    f = LeastSq(beta,y,X) + delta*sum(abs(np.delete(beta, nopen)))
  else:
    f = LeastSq(beta,y,X) + delta*sum(abs(beta))
  return f


def LassoFISTA(y,X,Lambda,W=np.ones(X.shape[0]),betaInit=np.zeros(X.shape[1]),
                        nopen=np.array([0]),
                        tol=1e-8,maxIter=1000,trace=False):
  # Observation weighting
  y = np.sqrt(W)*y
  X = np.diag(np.sqrt(W)).dot(X)
  
  ### Set Algo. Values
  eta = 1/max(2* np.linalg.eigvals(X.T.dot(X))/X.shape[0])
  theta = 1
  thetaO = theta
  beta = betaInit
  v = beta
  cv = 0
  
  k = 0
  while True:
    k += 1
    
    thetaO = theta
    theta = (1+np.sqrt(1+4*thetaO**2))/2
    delta = (1-thetaO)/theta
    
    betaO = beta
    beta = prox(v - eta*LeastSqgrad(v,y,X), Lambda*eta,nopen)
    
    v = (1-delta)*beta + delta*betaO
    
    # Show objective function value
    if trace & (k%100 == 0):
       print("Objective Func. Value at iteration",k,":",LassoObj(beta,y,X,Lambda,nopen))
    
    # Break if diverges
    if abs(LassoObj(beta,y,X,Lambda,nopen)-LassoObj(betaO,y,X,Lambda,nopen)) < tol or k > maxIter:
      break
    
    if k > maxIter:
      print("Max. number of iterations reach in Lasso minimization.")
      cv = -555

  value=LassoObj(beta,y,X,Lambda,nopen)
  loss=LeastSq(beta,y,X)
  l1norm=sum(abs(beta))
  nbIter=k
  convergenceFISTA=cv
  
  return beta, value, loss, l1norm, nbIter, convergenceFISTA

betaL, valL, lossL, l1normL, nbIterL, cvFL = LassoFISTA(y,X,.5,W=np.ones(X.shape[0]),
                                          betaInit=np.zeros(X.shape[1]),nopen=np.array([0]),tol=1e-8,maxIter=1000,trace=True)


### Test on Lalonde dataset
f = open("E:\LassoFISTA\dataset\d.txt","r")
header1 = f.readline()
d = []
for line in f:
  line = line.strip()
  columns = line.split()
  d.append(int(columns[1]))
f.close()
d=np.array(d)

f = open("E:\LassoFISTA\dataset\y.txt","r")
header1 = f.readline()
y = []
for line in f:
  line = line.strip()
  columns = line.split()
  y.append(float(columns[1]))
f.close()
y=np.array(y)
y=y.astype(float)
  
f = open("E:\LassoFISTA\dataset\X.txt","r")
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

import time
start_time = time.time()

betaL, valL, lossL, l1normL, nbIterL, cvFL = LassoFISTA(y,X,Lambda,W=1-d,
                                          betaInit=np.zeros(X.shape[1]),nopen=np.array([0]),tol=1e-6,maxIter=1e6,trace=True)
print("--- %s seconds ---" % (time.time() - start_time))
