DataSim <- function(n=2000,p=50,Ry=.5,Rd=.2,rho=.5){
  ### DGP for Monte-Carlo experiment
  ### Jeremy L Hour and Marianne Blehaut
  ### last edited: 18 fevrier 2016
  ### Specifically changed for test of Lasso function
  
  ### INPUTS:
  ### n : sample size
  ### p : number of variables
  ### Ry : desired R-squared for outcome equation
  ### Rd : desired R-squared for treatment equation
  
  
  ### Covariate correlation coefficients
  Sigma <- matrix(0,nrow=p, ncol=p)
  
  for(k in 1:p){
    for(j in 1:p){
      Sigma[k,j] <- rho^abs(k-j)
    }
  }
  
  ### Treatment effect
  a <- 0
  
  ### Treatment variable coefficient
  gamma <- rep(0,p)
    
  for(j in 1:abs(p/2)){
    gamma[j] <- 1*(-1)^(j) / j^2
  }
    
  ### Outcome equation coefficients
  b <- gamma
    
  for(j in (abs(p/2)+1):p){
    b[j] <- (-1)^(j+1) / (p-j+1)^2
  }
  
  ### Adjustment to match R.squared
  c <- sqrt((1/t(gamma)%*%Sigma%*%gamma)*(Rd/(1-Rd)))
  gamma <- c*gamma
  
  c <- sqrt((1/t(b)%*%Sigma%*%b)*(Ry/(1-Ry)))
  b <- c*b

  
  # Simulate covariates
  X <- mvrnorm(n = n, mu=rep(0,p), Sigma)
  # Simulate treatment
  # logit : d <- as.numeric(runif(n) < 1/(1+exp(-X%*%gamma)))
  d <- as.numeric(runif(n) < dnorm(X%*%gamma))
  # Simulate outcome
  y <- a*d + X%*%b + rnorm(n)

  # Add the intercept
  X <- cbind(rep(1,n),X)
  
  return(list(X=X,
              y=y,
              d=d,
              b=b,
              g=gamma))
}