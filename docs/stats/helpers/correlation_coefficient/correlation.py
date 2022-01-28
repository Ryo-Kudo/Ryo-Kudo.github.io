import numpy as np
from scipy import stats
from ipywidgets import interact, IntSlider, FloatSlider
import matplotlib.pyplot as plt

def plot(r, random_seed):
    n = 100
    np.random.seed(random_seed)
    x = np.random.uniform(size=n)
    e = np.random.uniform(size=n)
    y = r * x + np.sqrt(1 - r ** 2) * e

    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(x - x.mean(), y - y.mean(), marker='.')
    plt.setp(ax, title='相関係数 $r_{xy}=' + '{r:.2f}$'.format(r=stats.pearsonr(
                                                                x, y)[0]),
             xlabel='$x$', ylabel='$y$', xticks=(), yticks=())

    plt.show()

def show():
    style = dict(description_width='7em')
    r = FloatSlider(value=0, min=-1, max=1, step=0.1,
                    description='相関係数 (目安)', continuous_update=False,
                    readout_format='.1f', style=style)
    seed = IntSlider(value=1, min=1, max=256, description='乱数',
                     continuous_update=False, style=style)
    interact(plot, r=r, random_seed=seed)
