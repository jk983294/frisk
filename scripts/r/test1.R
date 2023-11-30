library(glmnet)

set.seed(42) # Set seed for reproducibility

n <- 100000L  # Number of observations
p <- 1000L  # Number of predictors included in model
real_p <- 15L  # Number of true predictors

## Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

model <- glmnet(x, y, type.measure="mse", alpha = 0.5, family="gaussian")

# See how increasing lambda shrinks the coefficients
# Each line shows coefficients for one variables, for different lambdas.
# The higher the lambda, the more the coefficients are shrunk towards zero.
plot(model, xvar = "lambda")

system.time({ model <- glmnet(x, y, type.measure="mse", alpha = 0.5, family="gaussian") })
predicted <- predict(model, s=model$lambda.1se, newx=x)
(mse <- mean((y - predicted)^2))
