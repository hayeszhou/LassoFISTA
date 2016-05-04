### High-dimension Calibration for Treatment Effect
### 28 decembre 2015
### J L'Hour

### Set working directory
setwd("R:/Simulations/BEAST") # R pour Jeremy, Z pour Marianne

rm(list=ls())


### 0. Settings

### Load packages
library("hdm")

set.seed(1)
n = 100 #sample size
p = 100 # number of variables
s = 3 # nubmer of variables with non-zero coefficients
X = matrix(rnorm(n*p), ncol=p)
beta = c(rep(5,s), rep(0,p-s))
Y = X%*%beta + rnorm(n)
lasso.reg = rlasso(Y~X,post=FALSE) # use lasso, not-Post-lasso
print(lasso.reg, all=FALSE) # can also do summary(lasso.reg, all=FALSE)
##
## Call:
## rlasso(formula = Y ~ X, post = FALSE)
##
## X1 X2 X3 X83
## 4.66808 4.81418 4.77023 -0.02741
yhat.lasso = predict(lasso.reg) #in-sample prediction
Xnew = matrix(rnorm(n*p), ncol=p) # new X
Ynew = Xnew%*%beta + rnorm(n) #new Y
yhat.lasso.new = predict(lasso.reg, newdata=Xnew) #out-of-sample prediction
post.lasso.reg = rlasso(Y~X,post=TRUE) #now use post-lasso
print(post.lasso.reg, all=FALSE) # or use summary(post.lasso.reg, all=FALSE)
##
## Call:
## rlasso(formula = Y ~ X, post = TRUE)
##
## X1 X2 X3
## 4.945 5.049 4.984
yhat.postlasso = predict(post.lasso.reg) #in-sample prediction
yhat.postlasso.new = predict(post.lasso.reg, newdata=Xnew) #out-of-sample prediction
MAE<- apply(cbind(abs(Ynew-yhat.lasso.new), abs(Ynew - yhat.postlasso.new)),2, mean)
names(MAE)<- c("lasso MAE", "Post-lasso MAE")
print(MAE, digits=2) # MAE for Lasso and Post-Lasso
## lasso MAE Post-lasso MAE
## 0.85 0.77