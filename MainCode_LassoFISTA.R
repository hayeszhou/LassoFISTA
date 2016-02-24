### Lasso FISTA: main example
### 15 fevrier 2016
### J L'Hour

### Set working directory
setwd("//ulysse/users/JL.HOUR/1A_These/Lasso_FISTA") 

rm(list=ls())
set.seed(30031987)


### 0. Settings

### Load packages


### Load user-defined functions
source("functions/DataSim_LassoTest.R") 
source("functions/LassoFISTA.R")

library("MASS")
library("reshape2")
library("ggplot2")

### 1. Lalonde Dataset
load("dataset/LalondeData_Unscaled.R")

X <- data$X
W_Lasso <- as.vector(data$W)
y <- data$y
y <- (y-mean(y))/sd(y) # Need to renormalize y
d <- data$d

n <- nrow(X)
p <- ncol(X)

eta <- 1/max(2*eigen(t(X)%*%X)$values/nrow(X))

# Overall penalty
c <- 2
g <- .1/log(max(p,n))
lambda <- c*qnorm(1-.5*g/p)/sqrt(n)


W_Lasso <- W_Lasso*sum(d)
m_y <- c(t(W_Lasso)%*%y/sum(W_Lasso))

psi0 <- as.vector(sqrt( t(W_Lasso*(y-m_y)^2) %*% (diag(sqrt(W_Lasso))%*%X)^2 / n ))


Lassofit_vanilla <- LassoFISTA(betaInit=rep(0,p),y,X,W=(1-d),
                           nopen=1,lambda,psi=rep(1,p),
                           tol=1e-6,maxIter=1e6,trace=T)

Lassofit_pen <- LassoFISTA(betaInit=rep(0,p),y,X,W=W_Lasso,
                          nopen=1,lambda,psi=rep(1,p),
                          tol=1e-6,maxIter=1e6,trace=T)

Lassofit_penpen <- LassoFISTA(betaInit=rep(0,p),y,X,W=W_Lasso,
                           nopen=1,lambda,psi=psi0,
                           tol=1e-6,maxIter=1e6,trace=T)

coefdata <- data.frame(cbind(Lassofit_vanilla$beta,Lassofit_pen$beta,Lassofit_penpen$beta))
coefdata[,"id"] <- 1:p


coefdatamelt=melt(coefdata, id='id')
ggplot(coefdatamelt,aes(id,value, group=variable, color=variable))+geom_point()

### Nombre de variable selectionné par chacun des modeles
signaldata <- data.frame(cbind(Lassofit_vanilla$beta!=0,
                               Lassofit_pen$beta!=0,
                               Lassofit_penpen$beta!=0))
signaldata[,"id"] <- 1:p


signaldatamelt=melt(signaldata, id='id')
ggplot(signaldatamelt,aes(id,value, group=variable, color=variable))+geom_point()


### Correlation entre les predictions des trois modeles pour les non traités
predictdata <- data.frame(cbind(X%*%Lassofit_vanilla$beta,
                             X%*%Lassofit_pen$beta,
                             X%*%Lassofit_penpen$beta))
cor(predictdata,predictdata)


### Treatment eval
eps <- sd(data$y)*(y - X%*%Lassofit_penpen$beta)
pi <- mean(d)
theta <- sum(d*eps)/sum(d) - sum(W_Lasso*sum(d)*eps)/sum(W_Lasso*sum(d))


#####################
#####################
#####################
## SIMULATING DATA ##
#####################
#####################
#####################


### 2. Simulate data
dataset <- DataSim(n=200,p=500,Ry=.05,Rd=.2, rho=.995)
y <- dataset$y
X <- dataset$X
d <- dataset$d
p <- ncol(X)
n <- nrow(X)

## Valeur du pas obtenu
eta <- 1/max(2*eigen(t(X)%*%X)$values/nrow(X))

c <- 1.1
# Overall penalty level
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

# Check zero mean over residuals
eps <- y - X%*%fit2$beta 
mean(eps)

# L1-error
c(sum(abs(fit$beta-c(0,dataset$b))),
  sum(abs(fit2$beta-c(0,dataset$b))),
  sum(abs(fit3$beta-c(0,dataset$b))))

# L2-error
c(sum((fit$beta-c(0,dataset$b))^2),
  sum((fit2$beta-c(0,dataset$b))^2),
  sum((fit3$beta-c(0,dataset$b)))^2)


#### With specific penalty levels (trying to reproduce Lalonde's dataset features)
W_Lasso <- rcauchy(n, location = 0, scale = 2)
W_Lasso <- (1-d)* W_Lasso * (W_Lasso >0)
W_Lasso <- W_Lasso/sum(W_Lasso) * sum(d)
plot(W_Lasso)

m_y <- c(t(W_Lasso)%*%y/sum(W_Lasso))
psi0 <- as.vector(sqrt( t(W_Lasso*(y-m_y)^2) %*% (diag(sqrt(W_Lasso))%*%X)^2 / n ))


penload_fit <- LassoFISTA(betaInit=rep(0,p),y,X,W=W_Lasso,
                  nopen=NULL,lambda, psi=psi0,
                  tol=1e-6,maxIter=1000,trace=T)
penload_fit2 <- LassoFISTA(betaInit=rep(0,p),y,X,W=W_Lasso,
                   nopen=1,lambda,psi=psi0,
                   tol=1e-6,maxIter=1000,trace=T)
penload_fit3 <- LassoFISTA(betaInit=rep(0,p),y,X,W=W_Lasso,
                   nopen=c(1,2,p),lambda,psi=psi0,
                   tol=1e-6,maxIter=1000,trace=T)
# Function value
c(penload_fit$value,penload_fit2$value,penload_fit3$value)

# L1-error
c(sum(abs(penload_fit$beta-c(0,dataset$b))),
  sum(abs(penload_fit2$beta-c(0,dataset$b))),
  sum(abs(penload_fit3$beta-c(0,dataset$b))))

# L2-error
c(sum((penload_fit$beta-c(0,dataset$b))^2),
  sum((penload_fit2$beta-c(0,dataset$b))^2),
  sum((penload_fit3$beta-c(0,dataset$b)))^2)



#####################
#####################
#####################
## ORT func Test   ##
#####################
#####################
#####################

### Load user-defined functions
source("functions/DataSim_LassoTest.R") 
source("functions/LassoFISTA.R")
source("functions/ORTReg_Test.R")

library("MASS")
library("reshape2")
library("ggplot2")

load("dataset/LalondeData_Unscaled.R")

X <- data$X
W_Lasso <- as.vector(data$W)
y <- data$y
d <- data$d

n <- nrow(X)
p <- ncol(X)

beta0 <- c(log(sum(d)/sum(1-d)),rep(0,p-1))

W0 = (1-d)*exp(X%*%beta0)


Lalondefit <- ORTReg_Test(y,d,X,beta0,method="WLSLasso",
                        c=5*sd(y), nopenset=c(1),
                        maxIterPen=1e3,maxIterLasso=1e6,tolLasso=1e-6,PostLasso=T,trace=T)

### Treatment eval
eps <- y - X%*%Lalondefit$muLasso
c(t(W0)%*%eps/sum(W0))

pi <- mean(d)
theta <- sum(d*eps)/sum(d) - sum(W_Lasso*sum(d)*eps)/sum(W_Lasso*sum(d))
print(theta)