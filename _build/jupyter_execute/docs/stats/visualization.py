#!/usr/bin/env python
# coding: utf-8

# # グラフの表示 (visualization)
# ---
# データをわかりやすくグラフに表示することができると、データの理解や説明に役立つ。

# In[2]:


import numpy as np
import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## グラフの意味

# ### ヒストグラム
# ---
# - 縦軸に度数 (頻度)、横軸に階級 (値の範囲) をとったグラフである。
# - データ分布の形や偏りがないかなどを確認するのに利用する。
#     - 山が複数ある場合には、性質の異なる複数のグループが混在している可能性がある。
#     - 外れ値をどこに設定するか、区間の数をいくつに設定するかなどで印象が全く異なるので、複数表示して確認する。

# In[3]:


#関数の説明を表示
help(pd.DataFrame.hist)


# In[4]:


isl = pd.read_csv('./data/islands.csv', header=None, index_col=0, names=[''])
isl.tail(10)


# In[5]:


isl.hist()


# ### 箱ひげ図
# ---
# - 中央値・四分位点・外れ値を一度に可視化できるグラフである。
#     - 四角 (箱) の上下間 = 四分位範囲 (25%点 ~ 75%点)
#     - 四角 (箱) の中の線 = 中央値
#     - 上下に伸びた線 (ひげ) = 箱の上 (下) から箱の高さの1.5倍を延長した範囲 (データの最小値・最大値は超えない)
#     - 上下の点 = 外れ値
# - pd.DataFrame.boxplot を用いて描画する。

# In[6]:


#関数の説明を表示
help(pd.DataFrame.boxplot)


# In[7]:


ins = pd.read_csv('./data/InsectSprays.csv')
ins.tail()


# In[8]:


ins.boxplot(by='spray')


# ### 棒グラフ (bar plot)
# ---
# - pd.DataFrame.plot.bar を用いて描画する。

# In[9]:


#関数の説明を表示
help(pd.DataFrame.plot.bar)


# In[10]:


va = pd.read_csv('./data/VADeaths.csv', index_col=0)
va


# In[11]:


va.plot.bar()


# In[12]:


va.plot.bar(stacked=True)


# ### モザイク図 (mosaic plot)
# ---
# - 数値の大きさを面性で表すプロットである。
#     - クロス集計表の視覚化などに利用される。
# - statsmodels.graphics.mosaicplot.mosaic を用いる。

# In[14]:


#関数の説明を表示
help(mosaic)


# In[15]:


titanic = pd.read_csv('./data/titanic.csv')
titanic.tail()


# In[16]:


mosaic(titanic, ['Class', 'Survived'])
plt.show()


# In[17]:


mosaic(titanic, ['Sex', 'Survived'])
plt.show()


# ## （参考）その他のグラフについて
# ---
# - 上記のコードは随時追加していく予定である。
# - 上記以外のものについては、[pandasのドキュメント](https://pandas.pydata.org/pandas-docs/stable/visualization.html)や[seabornのチュートリアル](https://seaborn.pydata.org/tutorial/distributions.html)を参照すると良い。

# In[ ]:




