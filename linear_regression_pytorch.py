import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='talk', palette='colorblind')


class Linear(nn.Module):

    def __init__(self, hidden_dim):
        super(Linear, self).__init__()
        self.w = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim), requires_grad=True
        )

    def forward(self, x):
        return torch.matmul(x, self.w)


hidden_dim = 10

w_true = torch.randn(hidden_dim, hidden_dim).data
linear_model = Linear(hidden_dim)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-3)

n_epoch = 10000
losses = np.zeros(n_epoch,)
w_diff = np.zeros(n_epoch,)
for i in range(n_epoch):
    x = torch.randn(1, hidden_dim)
    y = torch.matmul(x, w_true)
    y_hat = linear_model.forward(x)
    loss = nn.functional.mse_loss(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses[i] = loss.item()
    w_diff[i] = torch.norm(linear_model.w - w_true).item()

f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(losses)
ax.plot(w_diff)
ax.set_title('Learning curve')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
sns.despine()
