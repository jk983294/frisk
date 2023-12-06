library(glmnet)

set.seed(42) # Set seed for reproducibility

n <- 100000L  # Number of observations
p <- 20L  # Number of predictors included in model
real_p <- 6L  # Number of true predictors

## Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

# use cv to choose best lambda
cv_model <- cv.glmnet(x, y, type.measure="mse", alpha = 0.5, family="gaussian")
plot(cv_model)
best_lambda <- cv_model$lambda.1se
model1 <- glmnet(x, y, lambda = best_lambda, alpha = 0.5, family="gaussian")

# glmnet generate path
model <- glmnet(x, y, lambda = NULL, alpha = 0.5, family="gaussian")

# See how increasing lambda shrinks the coefficients
# Each line shows coefficients for one variables, for different lambdas.
# The higher the lambda, the more the coefficients are shrunk towards zero.
plot(model, xvar = "lambda")
print(model)
coef(model, s=0.1)

system.time({ model <- glmnet(x, y, type.measure="mse", alpha = 0.5, family="gaussian") })
predicted <- predict(model, newx=x)
(mse <- mean((y - predicted)^2))

