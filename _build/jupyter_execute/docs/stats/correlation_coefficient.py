#!/usr/bin/env python
# coding: utf-8

# # 相関係数 (correlation coefficient)
# ---
# 変数同士の線形な関係の度合いを数値化したものである。

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 相関係数
# ---
# - 相関係数は、 $-1\leqq r_{xy}\leqq 1$ を満たし、$1$に近いほど正の相関が強く、$-1$に近いほど負の相関が強いことを意味する。
# - 通常、$|r_{xy}|<0.2$のときは無相関と見なす。

# In[3]:


from helpers.correlation_coefficient import correlation
correlation.show()


# $n$ 組のデータ$(x_1,\ y_1), (x_2,\ y_2),\dots,\ (x_n,\ y_n)$の相関係数$r_{xy}$を以下のように計算する。
# 
# $$
#     \begin{align*}
#         r_{xy} = \dfrac{S_{xy}}{\sqrt{S_{xx} S_{yy}}}
#     \end{align*}
# $$
# 
# ここで
# 
# $$
#     \begin{align*}
#         S_{xy} &= \Sigma ^n _{i=1} (x_i-\bar{x})(y_i-\bar{y}) = \Sigma x_i y_i - \dfrac{(\Sigma x_i)(\Sigma y_i)}{n} \\
#         S_{xx} &= \Sigma ^n _{i=1} (x_i-\bar{x})^2 = \Sigma {x_i}^2 - \dfrac{(\Sigma x_i)^2}{n} \\
#         S_{yy} &= \Sigma ^n _{i=1} (y_i-\bar{y})^2 = \Sigma {y_i}^2 - \dfrac{(\Sigma y_i)^2}{n} 
#     \end{align*}
# $$

# - 相関係数はあくまで**線形な関係のみを捉える**ため、変数同士に関係があっても、その関係が**非線形な場合は捉えられない**。

# In[4]:


from helpers.correlation_coefficient import non_linear
non_linear.show()


# ## 相関行列
# ---
# - 変数が多く全ての変数の散布図行列を表示しにくい場合には、代わりに相関係数を用いた相関行列を利用することがある。
# - Python: pandas.DataFrame.corr を用いる。

# In[5]:


swiss = pd.read_csv('./data/swiss.csv', index_col=0)
swiss.tail()


# In[6]:


#関数の説明を表示
help(pd.DataFrame.corr)


# In[7]:


cor = swiss.corr()
cor


# - 多くの場合、数値だけで表にするより、ヒートマップにした方が見やすい。
#     - seaborn.heatmap を用いる。
#     - vmin, vmax は設定しなければならない。設定しない場合、データ中の最小値・最大値を自動で使用してしまうので誤解を生む。
#     - cmap は中央が薄く、最小や最大に近づくほど色が濃いものを使用するとわかりやすい。

# In[8]:


# 関数の説明を表示
help(sns.heatmap)


# In[9]:


sns.heatmap(cor, vmin=-1, vmax=1, cmap='seismic', annot=True, square=True)
plt.show()


# - 変数が多い場合、相関関係が強い順にソートし、不要な変数を除外することで見やすくなることもある。

# In[13]:


# Firtilityと相関が強い3個に絞る
k = 4
cols = cor.nlargest(k, 'Fertility')['Fertility'].index
cm = np.corrcoef(swiss[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, vmin=-1, vmax=1, cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size':10}, cmap='Spectral', 
                yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()


# ## 散布図行列と相関行列
# ---
# - 散布図行列も相関行列も対角を挟んで半分は重複して情報を含んでいる。
# - 半分を散布図行列に、残り半分を相関行列にすることで、限られたスペースに無駄なく情報を詰めることができる。
# - 多変数データを可視化するとき、まず最初にこのグラフを作成するべきだが、Pythonの主なパッケージには該当するものがないので、以下のような処理を自作しておき、使いたい時にコピペできるようにしておくと便利である。
#     - *Rにはデータの可視化パッケージがたくさんあるので、この点はRが有利である……。*

# In[28]:


def corr_scatter_matrix(data, size=1.5, cmap='seismic', **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import seaborn as sns

    def corr(x, y, color, **kwargs):
        cor = np.corrcoef(x, y)[0, 1]
        norm = Normalize(vmin=kwargs.pop('vmin') if 'vmin' in kwargs else -1,
                         vmax=kwargs.pop('vmax') if 'vmax' in kwargs else 1)
        cmap = plt.get_cmap(kwargs.pop('cmap') if 'cmap' in kwargs else None)
        sm = ScalarMappable(norm, cmap)
        ax = plt.gca()
        ax.text(0.5, 0.5, '{:.2f}'.format(cor), transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center',
                **kwargs)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.setp(ax, facecolor=sm.to_rgba(cor))

    matrix = sns.PairGrid(data, height=size)
    matrix.map_upper(corr, cmap=cmap, **kwargs)
    matrix.map_lower(plt.scatter)
    matrix.map_diag(sns.histplot)
    plt.show()  


# In[29]:


corr_scatter_matrix(swiss)

