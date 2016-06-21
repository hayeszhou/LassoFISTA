### FISTA algo

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
