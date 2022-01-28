import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import statsmodels.api as sm
from ipywidgets import interact, IntSlider

X, y = make_regression(n_samples=1000, n_features=2, n_informative=2,
                       noise=20.0, random_state=1)

def r2(k):
    if k > 2:
        np.random.seed(0)
        X_large = np.hstack((X, np.random.normal(size=(1000, k - 2))))
    else:
        X_large = X
    df = pd.DataFrame(X_large, columns=['X{}'.format(i+1) for i in range(k)])
    result = sm.OLS(endog=y, exog=df).fit()
    print(df.tail())
    print()
    print('決定係数 (R^2)                     :{0:.3f}'.format(result.rsquared))
    print('自由度調整済み決定係数 (Adjusted R^2):{0:.3f}'.format(result.rsquared_adj))
    print()
    print(result.summary())

def show():
    k = IntSlider(value=2, min=2, max=100, continuous_update=False,
                  description='変数の数')
    interact(r2, k=k)
