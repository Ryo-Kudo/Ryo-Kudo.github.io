import numpy as np
import matplotlib.pyplot as plt


def show():
    eps = np.finfo(np.float).eps
    x = np.linspace(eps, 1 - eps, 30)
    gini = 1 - (x ** 2 + (1 - x) ** 2)
    entropy = - (x * np.log(x) + (1 - x) * np.log(1 - x))
    _, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), sharex=True, sharey=True)
    kwargs = dict(xlabel='片方のクラスの割合', xlim=(0, 1), ylim=(0, 0.75))
    ax1.plot(x, gini)
    kwargs['title'] = 'ジニ不純度'
    plt.setp(ax1, **kwargs)
    ax2.plot(x, entropy)
    kwargs['title'] = 'エントロピー'
    plt.setp(ax2, **kwargs)
    plt.show()
