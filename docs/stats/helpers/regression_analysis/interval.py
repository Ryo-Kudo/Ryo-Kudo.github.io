import numpy as np
from sklearn.datasets import make_regression
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt

x, y = make_regression(n_samples=30, n_features=1, n_informative=1,
                       noise=20.0, random_state=1)
X = sm.add_constant(x)

def plot_conf_itvl(ax, xlim, fitted_model, alpha, color, transparency):
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = fitted_model.predict(sm.add_constant(xx))
    mean_x = x.mean()
    n = len(x)
    dof = fitted_model.df_resid
    from scipy import stats
    t = stats.t.ppf(1 - alpha / 2, df=fitted_model.df_resid)
    sq_err = (fitted_model.resid ** 2).sum()
    conf = t * np.sqrt((sq_err / dof) * (1.0 / n + (xx - mean_x) ** 2 /
                       ((xx ** 2).sum() - n * mean_x ** 2)))
    upper = yy + abs(conf)
    lower = yy - abs(conf)
    ax.fill_between(xx, lower, upper, color=color, alpha=transparency,
                    label='信頼区間')

def plot_pred_itvl(ax, xlim, fitted_model, alpha, color, transparency):
    std, lower, upper = wls_prediction_std(fitted_model, sm.add_constant(xlim),
                                           alpha=alpha)
    ax.fill_between(xlim, lower, upper, color=color, alpha=transparency,
                    label='予測区間')

def plot(conf_alpha, pred_alpha):
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(x, y)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    fitted_model = sm.OLS(y, X).fit()

    plot_conf_itvl(ax, xlim, fitted_model, conf_alpha, 'blue', 0.4)
    plot_pred_itvl(ax, xlim, fitted_model, pred_alpha, 'blue', 0.1)

    ax.legend()
    plt.setp(ax, xlim=xlim, ylim=ylim, xticks=(), yticks=())
    plt.show()

def show():
    style=dict(description_width='7em')
    conf_alpha = FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01,
                             description='信頼区間の係数', readout_format='.2f',
                             continuous_update=False, style=style)
    pred_alpha = FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01,
                             description='予測区間の係数', readout_format='.2f',
                             continuous_update=False, style=style)
    interact(plot, conf_alpha=conf_alpha, pred_alpha=pred_alpha)
