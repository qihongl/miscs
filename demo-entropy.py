import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', context='talk', palette='colorblind')


def entropy(d_, base=2):
    """calculate entropy given a distribution

    Parameters
    ----------
    d_ : list/ 1d array
        a probability distribution
    base : int
        the log base

    Returns
    -------
    float
        entropy of d_

    """
    assert np.sum(d_) == 1
    return - np.sum([pi * np.log(pi)/np.log(base) for pi in d_])


# generate some distributions [p, 1-p]
n_points = 100
xs = np.linspace(0, 1, num=n_points)
ys = 1 - xs

# compute entropy
ents = np.zeros(n_points, )
for i, (x, y) in enumerate(zip(xs, ys)):
    ents[i] = entropy([x, y])

# plot
f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(ents)
ax.set_title('Entropy of Bernoulli(p)')
ax.set_xlabel('p')
ax.set_ylabel('Entropy')
ax.set_xticks([0, n_points//2, n_points])
ax.set_xticklabels([0, .5, 1])
sns.despine()
f.tight_layout()
