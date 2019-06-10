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
n_points = 1000
xs = np.linspace(0, 1, num=n_points)
ys = 1 - xs

# compute entropy
engs = np.zeros(n_points, )
ents = np.zeros(n_points, )
for i, (x, y) in enumerate(zip(xs, ys)):
    ents[i] = entropy([x, y])
    engs[i] = x * y

# plot
f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(ents, label='entropy')
ax.plot(engs, label='energy')
ax.set_title('Uncertainty of Bernoulli(p)')
ax.set_xlabel('p')
ax.set_ylabel('Uncertainty')
xticks = np.linspace(0, 1, num=5)
ax.set_xticks(xticks * n_points)
ax.set_xticklabels(xticks)
f.legend(bbox_to_anchor=(.75, .65))
f.tight_layout()
sns.despine()
