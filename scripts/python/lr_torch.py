import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

n = 10000  # Number of observations
p = 100  # Number of predictors included in model
real_p = 20  # Number of true predictors
X_numpy = np.random.rand(n, p)
y_numpy = np.add.reduce(X_numpy[:, 0:real_p], 1)

epochs = 100
batch_size = n // 10
X = Variable(torch.from_numpy(X_numpy)).float()
y = Variable(torch.from_numpy(y_numpy.reshape(-1, 1))).float()
dataset_train = TensorDataset(X, y)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
train_len = len(dataloader_train)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, x_dim, fit_intercept=True):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(x_dim, 1, bias=fit_intercept)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def get_coefs(self):
        coefs = self.linear.weight.detach().numpy().reshape(-1)
        intercept = 0
        if self.linear.bias:
            intercept = self.linear.bias.detach().numpy()
        return coefs, intercept


def train_model(model, dt_loader):
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(our_model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for i, data in enumerate(dt_loader, 0):
            # Get inputs
            inputs, targets = data
            # Forward pass: Compute predicted y by passing x to the model
            pred_y = model(inputs)

            # Compute and print loss
            loss = criterion(pred_y, targets)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == train_len - 1:
                print('epoch {} loss {}'.format(epoch, loss.item()))
    print('Training process has finished.')


def predict(model, x_val):
    with torch.no_grad():
        pred_y = model(x_val)
    return pred_y.numpy().reshape(-1)


# our model
our_model = LinearRegressionModel(x_dim=p, fit_intercept=False)
train_model(our_model, dataloader_train)
pred_y = predict(our_model, X)
mse = ((pred_y - y_numpy)**2).mean(axis=0)
print("predict mse", mse)
coefs, intercept = our_model.get_coefs()
print("coefs", coefs, intercept)
