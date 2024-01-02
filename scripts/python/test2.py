import numpy as np
from sklearn.linear_model import lars_path_gram

n = 500000  # Number of observations
p = 1000  # Number of predictors included in model
real_p = 100  # Number of true predictors
X = np.random.normal(size=(n, p))
x_mean_ = X.mean(axis=1)
x_sd_ = X.std(axis=1)

# add noise
# for c in range(real_p):
#     if c % 2 == 0:
#         for r in range(n):
#             X[r, c] = X[r, c] * 100

multiplier = np.array([1. if i % 2 == 0 else 100. for i in range(p)])
X = X * multiplier.T
x_mean_ = X.mean(axis=0)
x_sd_ = X.std(axis=0)
print("x_mean_", x_mean_)
print("x_sd_", x_sd_)
y = np.add.reduce(X[:, 0:real_p], 1)
y_sd = np.std(y)
y = y + np.random.normal(loc=0., scale=y_sd * 10, size=n)
y_mean = y.mean()
print(y_sd, y_mean)

# calc moment
Xy = np.dot(X.T, y)
gram = np.dot(X.T, X)

# # adjust
# for r in range(p):
#     for c in range(p):
#         gram[r, c] = gram[r, c] - n * x_mean_[r] * x_mean_[c]
#
# for i in range(p):
#     Xy[i] = Xy[i] - n * y_mean * x_mean_[i]

path = lars_path_gram(Xy, gram, n_samples=5, max_iter=real_p)
selected = path[1]
selected_noise = [i for i in selected if i >= 100]
print(len(selected), sorted(selected))
print(len(selected_noise), sorted(selected_noise))
coef = path[2][:, real_p]
# print(coef)
pred = np.dot(X, coef)
mse = np.mean((y - pred)**2)
pcor = np.corrcoef(y, pred)
print('mse ', mse, "pcor", pcor[0][1])
print('end')
