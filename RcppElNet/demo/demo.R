library(glmnet)
set.seed(42) # Set seed for reproducibility

n <- 100000L  # Number of observations
p <- 20L  # Number of predictors included in model
real_p <- 6L  # Number of true predictors

## Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

system.time({ model <- RcppElNet::fit_model(x, y, 0.5) })
system.time({ pred <- RcppElNet::model_predict(model, newx=x) })
(mse <- mean((y - pred)^2))
FM::pcor(y, pred)
model
