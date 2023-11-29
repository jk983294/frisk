import numpy as np
from glmnet import ElasticNet

n = 100000  # Number of observations
p = 1000  # Number of predictors included in model
real_p = 15  # Number of true predictors
X = np.random.rand(n, p)
y = np.add.reduce(X[:, 0:real_p], 1)

m = ElasticNet(alpha=0.5)
m = m.fit(X, y)
p = m.predict(X)