library(glmnet)

##############################################################
## (p - real_p) useless variables in the model, only real_p that are useful.
##############################################################

set.seed(42)  # Set seed for reproducibility

n <- 10000L  # Number of observations
p <- 1000L  # Number of predictors included in model
real_p <- 15L  # Number of true predictors

## Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

## Split data into training and testing datasets.
## 2/3rds of the data will be used for Training and 1/3 of the
## data will be used for Testing.
train_rows <- sample(1:n, .66*n)
x_train <- x[train_rows, ]
x_test <- x[-train_rows, ]

y_train <- y[train_rows]
y_test <- y[-train_rows]

## Now we will use 10-fold Cross Validation to determine the
## optimal value for lambda for...

################################
##
## alpha = 0, Ridge Regression
## coef L2 penalty
################################
ridge_model <- cv.glmnet(x_train, y_train, type.measure="mse", alpha=0, family="gaussian")
ridge_pred <- predict(ridge_model, s=ridge_model$lambda.1se, newx=x_test)
## s = is the "size" of the penalty that we want to use, and
##     thus, corresponds to lambda. (I believe that glmnet creators
##     decided to use 's' instead of lambda just in case they
##     eventually coded up a version that let you specify the
##     individual lambdas, but I'm not sure.)
##
##     In this case, we set 's' to "lambda.1se", which is
##     the value for lambda that results in the simplest model
##     such that the cross validation error is within one
##     standard error of the minimum.
##
##     If we wanted to to specify the lambda that results in the
##     model with the minimum cross validation error, not a model
##     within one SE of of the minimum, we would
##     set 's' to "lambda.min".
##
##     Choice of lambda.1se vs lambda.min boils down to this...
##     Statistically speaking, the cross validation error for
##     lambda.1se is indistinguishable from the cross validation error
##     for lambda.min, since they are within 1 SE of each other.
##     So we can pick the simpler model without
##     much risk of severely hindering the ability to accurately
##     predict values for 'y' given values for 'x'.
##
##     All that said, lambda.1se only makes the model simpler when
##     alpha != 0, since we need some Lasso regression mixed in
##     to remove variables from the model. However, to keep things
##     consistent when we compare different alphas, it makes sense
##     to use lambda.1se all the time.
##
## newx = is the Testing Dataset

## Lastly, let's calculate the Mean Squared Error (MSE) for the model
## created for alpha = 0.
## The MSE is the mean of the sum of the squared difference between
## the predicted 'y' values and the true 'y' values in the
## Testing dataset...
(ridge_mse <- mean((y_test - ridge_pred)^2))

################################
##
## alpha = 1, Lasso Regression
## coef L1 penalty
################################
lasso_model <- cv.glmnet(x_train, y_train, type.measure="mse", alpha=1, family="gaussian")
lasso_pred <- predict(lasso_model, s=lasso_model$lambda.1se, newx=x_test)
(lasso_mse <- mean((y_test - lasso_pred)^2))

################################
##
## alpha = 0.5, a 50/50 mixture of Ridge and Lasso Regression
## coef L2 penalty + coef L1 penalty
################################
net_model <- cv.glmnet(x_train, y_train, type.measure="mse", alpha=0.5, family="gaussian")
net_pred <- predict(net_model, s=net_model$lambda.1se, newx=x_test)
(net_mse <- mean((y_test - net_pred)^2))

################################
##
## However, the best thing to do is just try a bunch of different
## values for alpha rather than guess which one will be best.
##
## The following loop uses 10-fold Cross Validation to determine the
## optimal value for lambda for alpha = 0, 0.1, ... , 0.9, 1.0
## using the Training dataset.
##
## NOTE, on my dinky laptop, this takes about 2 minutes to run
##
################################

list.of.fits <- list()
for (i in 0:10) {
  ## Here's what's going on in this loop...
  ## We are testing alpha = i/10. This means we are testing
  ## alpha = 0/10 = 0 on the first iteration, alpha = 1/10 = 0.1 on
  ## the second iteration etc.

  ## First, make a variable name that we can use later to refer
  ## to the model optimized for a specific alpha.
  ## For example, when alpha = 0, we will be able to refer to
  ## that model with the variable name "alpha0".
  model_name <- paste0("model", i/10)

  ## Now fit a model (i.e. optimize lambda) and store it in a list that
  ## uses the variable name we just created as the reference.
  list.of.fits[[model_name]] <- cv.glmnet(x_train, y_train, type.measure="mse", alpha=i/10, family="gaussian")
}

## Now we see which alpha (0, 0.1, ... , 0.9, 1) does the best job
## predicting the values in the Testing dataset.
results <- data.frame()
for (i in 0:10) {
  model_name <- paste0("model", i/10)

  ## Use each model to predict 'y' given the Testing dataset
  predicted <- predict(list.of.fits[[model_name]], s=list.of.fits[[model_name]]$lambda.1se, newx=x_test)

  ## Calculate the Mean Squared Error...
  mse <- mean((y_test - predicted)^2)

  ## Store the results
  temp <- data.frame(alpha=i/10, mse=mse, model_name=model_name)
  results <- rbind(results, temp)
}

## View the results
results