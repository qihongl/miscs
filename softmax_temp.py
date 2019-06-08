import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax

sns.set(style='white', context='talk', palette='colorblind')

x = [.4, .3, .3]
betas = [0, .1, .3, 1, 3, 10]

# preproc
n_xs = len(x)
n_bs = len(betas)
x = np.array(x)

b_pal = sns.color_palette('RdBu', n_colors=n_bs)
f, ax = plt.subplots(1, 1, figsize=(8, 5))

for i, beta in enumerate(betas):
    y = softmax(x/beta)
    ax.plot(y, color=b_pal[i])

ax.plot(x, color='k')

f.legend([b for b in betas]+['raw'], title='temp')
ax.set_title('Softmax with different temperature')
ax.set_ylabel('Prob')
ax.set_xlabel('Actions')
ax.set_xticks(range(len(x)))
ax.set_xticklabels(range(len(x)))
sns.despine()
f.tight_layout()
