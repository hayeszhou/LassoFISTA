### group lasso test
### 04/06/2020
### Jérémy L'Hour

library('MASS')
source("functions/group_lasso.R")

### SIMULATE DATA
set.seed(999)
rho = .5
p = 50
n = 10000

Sigma = matrix(0,nrow=p, ncol=p)
for(k in 1:p){
  for(j in 1:p){
    Sigma[k,j] = rho^abs(k-j)
  }
}

X = mvrnorm(n = n, mu=rep(0,p), Sigma)
even = 1:p %% 2 == 0
X[,even] = ifelse(X[,even] > 0,1,0)

y_1 =  1*X[,1] + rnorm(n)
y_2 =  -0.6*X[,1] + rnorm(n)

Y = cbind(y_1,y_2)

### Test group Lasso
sample_size = c(50,100,200,500,1000,5000,8000,10000)
beta = matrix(ncol=2,nrow=length(sample_size))

for(i in 1:length(sample_size)){
  m = sample_size[i]
  gamma_pen = .1/log(max(p,m))
  lambda = 1.1*qnorm(1-.5*gamma_pen/p)/sqrt(m) 
  res = group_lasso(X=X[1:m,],y=Y[1:m,],lambda=lambda,trace=TRUE)
  
  beta[i,] = res$beta[1,]
  
}

plot(sample_size,beta[,1], type="line", col="red", ylim=c(-1,1))
lines(sample_size,beta[,2],col="blue")

