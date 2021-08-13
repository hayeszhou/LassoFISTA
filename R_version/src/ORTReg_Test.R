ORTReg_Test <- function(y,d,X,beta,method="WLSLasso",
                             c=1.1, nopenset=c(1),
                             maxIterPen=100,maxIterLasso=1e6,tolLasso=1e-6,PostLasso=F,trace=F){
  ### Function to compute mu hat
  ### Jeremy L Hour
  ### 11 janvier 2016
  ### EDITED : 19 fevrier 2016
  
  ### INPUTS:
  # y : Outcome variable (not normalized)
  # d : Treatment indicator
  # X : Covariates (not normalized)
  # beta : estimate from the first step
  # method : c("WLSLasso","LinearOutcome")  
  # c : constant for the overall penalty level
  # nopenset : set of indices that should not be penalized
  #            Default is intercept not penalized
  # maxIterPen : maximal number of iterations for penalty convergence
  # maxIterLasso : maximal number of iterations in Lasso procedure
  # tolLasso : tolerance for stopping criterion in Lasso minimization
  # PostLasso : if TRUE computes the PostLasso instead of Lasso
  # trace : if TRUE print convergence info
  
  # The first column of X must be the constant
  
  
  ### Load user-defined functions
  source("functions/LassoFISTA.R")
  
  ### Setting
  d <- as.matrix(d)
  y <- as.matrix(y)
  X <- as.matrix(X)
  
  n <- nrow(X)
  p <- ncol(X)
  
  ### Type of method to compute weights
  if(method == "WLSLasso"){ W=(1-d)*exp(X%*%beta) }
  if(method == "LinearOutcome"){ W=(1-d)*(sum(d)/sum(1-d)) }
  W <- as.vector(W)

  ### First step: Lasso
  
  # Overall penalty level
  g <- .1/log(max(p,n))
  lambda <- c*qnorm(1-.5*g/p)/sqrt(n)
  lambda_tilde <- lambda/sd(y) # Lambda used in the rescaled minimization program
  
  # Rescale the y
  y_tilde <- (y-mean(y))/sd(y)
  
  # Penalty loadings: get a preliminary estimate
  m_y <- c(t(W)%*%y_tilde/sum(W))
  Psi <- as.vector(sqrt( t(W*(y_tilde-m_y)^2) %*% (diag(sqrt(W))%*%X)^2 / n ))
  
  # Estimation parameters
  v <- .01 # Stopping rule
  k <- 0
  mu <- rep(0,p)
  
  
  # Lasso estimate
  repeat{
    k <- k+1
    
    # Compute Lasso estimate
    LassoEstim <- LassoFISTA(betaInit=mu,y_tilde,X,W=W,
                            nopen=nopenset,lambda=lambda_tilde,psi=Psi,
                            tol=tolLasso,maxIter=maxIterLasso,trace=F)
    mu <- LassoEstim$beta
    
    # Adjust the estimation ?
    eps <- y_tilde - X%*%LassoEstim$beta
    de <- c(t(W)%*%eps/sum(W))
    mu[1] <- mu[1] + de
    
    # Update penalty loadings
    PrePsi <- Psi
    Psi <- as.vector(sqrt( t(W*(y_tilde-X%*%mu)^2) %*% (diag(sqrt(W))%*%X)^2 / n ))
    
    # Trace showing
    if(trace & k%%5==0){
      print(paste("Max. pen. loading diff at Lasso Iteration nb.",k,":",max(abs(Psi-PrePsi)))) 
    }
    
    # Stopping rules
          if(k > maxIterPen || max(abs(diag(Psi-PrePsi))) < v || LassoEstim$convergenceFISTA==-555){
            cvg <- LassoEstim$convergenceFISTA
            break
         } 
  }
  
  cvg = 0
  if(k > maxIterPen){
    cvg=-999
    print("Penalty estimation did not converge.")
  }
  
  # Obtain the estimates for the unscaled model
  muLasso <- sd(y)*mu
  muLasso[1] <- muLasso[1] + mean(y) # Get the full intercept
  
  # Get estimated active set
  if(cvg==0){
    SHat <- union(1,which(muLasso != 0)) 
  } else {
    SHat <- NA
  }
  
  ### Second step: Post-Lasso
  muPL <- rep(0,p)
  if(PostLasso==T & cvg==0){
    OrthoReg <- lm(y ~ X[,SHat] - 1, weights=W)
    muPL[SHat] <- OrthoReg$coefficients
    muPL[is.na(muPL)] <- 0
  }
  
  # Return objects
  return(list(method=method,
              lambda=lambda,
              SHat=SHat,
              muPL=muPL,
              muLasso=c(muLasso),
              nbIter=k,
              convergence=cvg
  ))
  
  
}
