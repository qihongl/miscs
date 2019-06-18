import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', context='talk', palette='colorblind')


def kl(P, Q):
    """(P * (P / Q).log()).sum()

    Parameters
    ----------
    P : type
        Description of parameter `P`.
    Q : type
        Description of parameter `Q`.

    Returns
    -------
    type
        Description of returned object.

    """
    assert P.sum() == 1 == Q.sum()
    return F.kl_div(Q.log(), P, None, None, 'sum')


# parametrically vary two Bernoulli
n_points = 50
qs = ps = np.linspace(0, 1, n_points)
p = .1
q = .2

# compute kl distance matrix
kls = np.zeros((n_points, n_points))
for i, p in enumerate(ps):
    for j, q in enumerate(qs):
        P = torch.Tensor([p, 1-p])
        Q = torch.Tensor([q, 1-q])
        kls[i, j] = kl(P, Q).item()

# plot
vmin = np.min(kls)
n_ticks = 2
ticklabels = np.linspace(0, 1, n_ticks)
ticks = np.linspace(0, n_points, n_ticks)
kls[kls == np.inf] = None

f, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.heatmap(kls, cmap='viridis', square=True, vmin=0, ax=ax)
ax.set_title('KL(P,Q), where P,Q are Bernoulli')
ax.set_xlabel('p')
ax.set_ylabel('q', rotation=0)
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, rotation=0)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels, rotation=0)
f.tight_layout()
