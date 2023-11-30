import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

n = 10000  # Number of observations
p = 100  # Number of predictors included in model
real_p = 20  # Number of true predictors
X_numpy = np.random.rand(n, p)
y_numpy = np.add.reduce(X_numpy[:, 0:real_p], 1)


class ElNetModel(torch.nn.Module):
    def __init__(self, x_dim, lambda_, alpha, fit_intercept=True):
        super(ElNetModel, self).__init__()
        self.linear = torch.nn.Linear(x_dim, 1, bias=fit_intercept)
        self.lambda_ = lambda_
        self.alpha = alpha

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def loss(self, mse_loss):
        l1_norm = self.linear.weight.abs().sum()
        l2_norm = self.linear.weight.pow(2).sum()
        return mse_loss + self.lambda_ * (self.alpha * l1_norm + 0.5 * (1 - self.alpha) * l2_norm)

    def get_coefs(self):
        coefs = self.linear.weight.detach().numpy().reshape(-1)
        intercept = 0
        if self.linear.bias:
            intercept = self.linear.bias.detach().numpy()
        return coefs, intercept

    def fit(self, X_numpy, y_numpy):
        epochs = 100
        batch_size = n // 10
        X = Variable(torch.from_numpy(X_numpy)).float()
        y = Variable(torch.from_numpy(y_numpy.reshape(-1, 1))).float()
        dataset_train = TensorDataset(X, y)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        train_len = len(dataloader_train)

        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01)
        for epoch in range(epochs):
            for i, data in enumerate(dataloader_train, 0):
                # Get inputs
                inputs, targets = data
                # Forward pass: Compute predicted y by passing x to the model
                pred_y = self(inputs)

                # Compute and print loss
                mse_loss = criterion(pred_y, targets)
                loss = self.loss(mse_loss)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == train_len - 1:
                    print('epoch {} loss {}'.format(epoch, loss.item()))
        print('Training process has finished.')

    def predict(self, x_val):
        with torch.no_grad():
            pred_y = self(Variable(torch.from_numpy(x_val)).float())
        return pred_y.numpy().reshape(-1)


# our model
my_model = ElNetModel(x_dim=p, lambda_=0.1, alpha=0.5, fit_intercept=False)
my_model.fit(X_numpy, y_numpy)
pred_y = my_model.predict(X_numpy)
mse = ((pred_y - y_numpy)**2).mean(axis=0)
print("predict mse", mse)
coefs, intercept = my_model.get_coefs()
print("coefs", coefs, intercept)
