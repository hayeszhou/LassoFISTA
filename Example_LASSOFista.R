### Lasso FISTA: main example
### 15 fevrier 2016, edited 24 fevrier
### J L'Hour

### Set working directory
rm(list=ls())
set.seed(30031989)
setwd("//ulysse/users/JL.HOUR/1A_These/LassoFISTA") 


### 0. Settings

### Load packages
library("MASS")
library("reshape2")
library("ggplot2")

### Load user-defined functions
source("functions/DataSim.R") 
source("functions/LassoFISTA.R")
source("functions/RidgeFISTA.R")


#####################
#####################
#####################
## SIMULATED  DATA ##
#####################
#####################
#####################


### 1. Simulate data
dataset <- DataSim(n=500,p=1000,Ry=.5,Rd=.2, rho=.5)
y <- dataset$y
X <- dataset$X
d <- dataset$d
p <- ncol(X)
n <- nrow(X)



c <- 1.1
g <- .1/log(max(p,n))
lambda <- c*qnorm(1-.5*g/p)/sqrt(n)

fit <- LassoFISTA(betaInit=rep(0,p),y,X,W=rep(1,nrow(X)),
                       nopen=NULL,lambda,
                       tol=1e-6,maxIter=1000,trace=T)
fit2 <- LassoFISTA(betaInit=rep(0,p),y,X,W=rep(1,nrow(X)),
                  nopen=1,lambda,
                  tol=1e-8,maxIter=1000,trace=T)
fit3 <- LassoFISTA(betaInit=rep(0,p),y,X,W=rep(1,nrow(X)),
                   nopen=c(1,2,p),lambda,
                   tol=1e-8,maxIter=1000,trace=T)
# Function value
c(fit$value,fit2$value,fit3$value)


# L1-error
c(sum(abs(fit$beta-c(0,dataset$b))),
  sum(abs(fit2$beta-c(0,dataset$b))),
  sum(abs(fit3$beta-c(0,dataset$b))))

# L2-error
c(sum((fit$beta-c(0,dataset$b))^2),
  sum((fit2$beta-c(0,dataset$b))^2),
  sum((fit3$beta-c(0,dataset$b)))^2)


### Ridge test
fit <- RidgeFISTA(betaInit=rep(0,p),y,X,W=rep(1,nrow(X)),
                  nopen=NULL,lambda,
                  tol=1e-6,maxIter=1000,trace=T)

analsol <- solve(t(X)%*%X + n*lambda*diag(p)) %*% (t(X)%*%y)

RidgeObj(analsol,y,X,lambda,NULL)

### No penalty on intercept
Ridgefit <- RidgeFISTA(betaInit=rep(0,p),y,X,W=rep(1,nrow(X)),
                  nopen=1,lambda,
                  tol=1e-6,maxIter=1000,trace=T)

Lassofit <- LassoFISTA(betaInit=rep(0,p),y,X,W=rep(1,nrow(X)),
                  nopen=1,lambda,
                  tol=1e-6,maxIter=1000,trace=T)

# L1-error
c(sum(abs(Ridgefit$beta-c(0,dataset$b))),
  sum(abs(Lassofit$beta-c(0,dataset$b))))

# L2-error
c(sum((Ridgefit$beta-c(0,dataset$b))^2),
  sum((Lassofit$beta-c(0,dataset$b))^2))