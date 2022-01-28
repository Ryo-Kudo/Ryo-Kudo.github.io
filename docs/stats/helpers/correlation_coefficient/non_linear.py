import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def show():
    x = np.linspace(-1, 1, 9)
    y = x ** 2
    r = stats.pearsonr(x, y)[0]

    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(x, y)
    plt.setp(ax, title='$y=x^{2}$の相関係数' + '$r={r}$'.format(r=r),
             xlabel='$X$', ylabel='$Y$', xticks=(), yticks=())

    plt.show()
