#!/usr/bin/env python
# coding: utf-8

# # 回帰分析 (regression analysis)
# ---
# - 1つの変数 $Y$ を他の変数 $X$ で説明しようとする分析である。
# - $X$ が1変数だけの場合を**単回帰**分析、2変数以上の場合を**重回帰**分析という。
# - 線形回帰: $y=a+b_{1}x_{1}+\dots+b_{k}x_{k}$ ( $k$ は変数の数) を扱う。

# In[2]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 用語の整理
# ---
# - 相関分析
#     - 変数間の関係の有無を調べる
# - 回帰分析
#     - ひとつの変数を他の変数で説明する
#     - 変数 $Y$ : 従属変数（または目的変数）
#     - 変数 $X$ : 独立変数（または説明変数）
# - **単回帰分析**
#     - 回帰分析のうち、独立変数$X$ がひとつのもの
# - **重回帰分析**
#     - 回帰分析のうち、独立変数$X$ が二つ以上のもの

# ## データの構造式と最小二乗法
# ---
# データの構造式(重回帰モデル)
# 
# $$
#     Y_{i}=a+b_{1}X_{1i}+\dots+b_{k}X_{ki}+\epsilon_{i} \, (i=1,\ 2,\dots,\ n) \\
# $$
# 
# - 独立変数: $X\ (X_{1},\ X_{2},\dots,\ X_{n})$
#     - ここで、$i$が$1$のみであれば、単回帰モデルとなる。
# - 従属変数: $Y\ (Y_{1},\ Y_{2},\dots,\ Y_{n})$
# - $\epsilon_{i}$ は回帰方程式では説明しきれない誤差であり、互いに独立に $N (0, \sigma^2)$に従う。
#     - 誤差が小さくなるように、$a,\ b_1,\dots,\ b_k$を定めることができれば、独立変数で従属変数をうまく説明できていると言える。
#     - 誤差$\epsilon$の二乗をの合計を最小化するように回帰係数を最適化する。（最小二乗法）[^1] <br> ${\displaystyle S=\sum ^{n}_{i=1} \epsilon ^{2}_{i} =\sum ^{n}_{i=1}\{Y_{i} -( a+b_{1i} X_{1i} +\dots +b_{ki} X_{ki})\}^{2}}$
# 
# 
# [^1]: 最小二乗法にも複数の種類がある。今回は、Ordinary Least Squares(OLS)を取り上げた。
# 
# 
# 

# ## 単回帰分析 (simple regression analysis)

# ### 回帰方程式の評価
# ---
# - 観測値 $Y_i$ と予測 $\hat{Y}$ との差 (残差 residual) を $e_i=Y_i-\hat{Y}_i$ とすると、以下で定義される**決定係数** $R^2$で回帰方程式が上手く当てはまっているかを評価する。
# - $ R ^{2} =1-\frac{{\displaystyle \sum ^{n}_{i=1} e ^{2}_{i}} }{{\displaystyle \sum ^{n}_{i=1}\left( Y_{i} -\overline{Y}\right)^{2}} } $
# - 決定係数は、**最小二乗法を使って求められた単回帰方程式の場合**には $X, Y$ 間の相関係数 $r$ との間に $R^2=r^2$ という関係が成り立つ。

# ### 回帰係数の標本分布
# ---
# - **誤差項$\epsilon$が正規分布に従う**とき、回帰係数の推定量 $\hat{b}$ も正規分布に従う。
# - より正確には、誤差項 $\epsilon_1,\ \epsilon_2,\dots,\ \epsilon_n$ が独立で共通の正規分布 $N(0, \sigma^2)$ に従うとすると、<br>回帰係数の推定量 $\hat{b}$ の分布は平均 $b$ 、分散 $
# {\displaystyle \frac
#     {\sigma ^{2}}
#     {{\displaystyle \sum ^{n}_{i=1}\left( X_{i} -\overline{X}\right)^{2}} }
# }
# $ の正規分布 $
# N\left(b,\ {\displaystyle \frac
#     {\sigma ^{2}}
#     {{\displaystyle \sum ^{n}_{i=1}\left( X_{i} -\overline{X}\right)^{2}} }
# } \right)
# $ に従う。

# ### 回帰係数の検定
# ---
# - 各独立変数が、従属変数を説明できているかを確かめるために、回帰係数を検定する必要がある。
#     - 検定結果が統計的に優位でない場合、変数を変形することや、取り除くことが必要である。
# - 回帰係数 $b$ の推定量 $\hat{b}$ についての検定は、次の$t$値を利用して行う。
#     - $
# t={\displaystyle
#     \frac
#         {\hat{b} -b}
#         {\frac
#             {{\displaystyle
#                 \sqrt{s^{2}}
#             } }
#             {\sqrt{{\displaystyle
#                 \sum ^{n}_{i=1}\left( X_{i} -\overline{X}\right)^{2}
#             } }}
#         }
# }
# $
# , ただし
# $
# s^{2} ={\displaystyle
#     \frac
#         {{\displaystyle
#             \sum ^{n}_{i=1} e^{2}_{i}
#         } }
#         {n-2}
# }
# $
#     - $\hat{b}$は自由度 $n-2$ の $t$ 分布 $t(n-2)$ に従う。
#     - $s^2$ は誤差項 $\epsilon$ の分散 $\sigma^2$ の不偏推定量である。

# ### 標準誤差 (standard error)
# ---
# - 回帰係数 (の推定量) $\hat{b}$ の標準偏差 (の推定量) は以下の通りである。
# $$
# SE =
# \frac
#     {{\displaystyle
#         \sqrt{s^{2}}
#     } }
#     {\sqrt{{\displaystyle
#         \sum ^{n}_{i=1}\left( X_{i} -\overline{X}\right)^{2}
#     } }}
# $$

# ## 重回帰分析 (multiple regression analysis)
# ---
# - 基本的には単回帰分析と同じだが、回帰方程式の評価では**自由度調整済み決定係数**(Adjusted $R^2$)を用いる。

# ### 回帰方程式の評価
# ---
# - 自由度調整済み決定係数$R_a^2$ <br> $ R_a^2 = {\displaystyle
#     1-\frac
#         {n-1}
#         {n-k-1}
#     \left( 1-R ^{2}\right)
# } $
# - $R^2$は単回帰と同様の決定係数である。<br> $
# R ^{2} =1-\frac
#     {{\displaystyle
#         \sum ^{n}_{i=1} e ^{2}_{i}
#     } }
#     {{\displaystyle
#         \sum ^{n}_{i=1}\left( Y_{i} -\overline{Y}\right)^{2}
#     } }
# $
# - 決定係数$R^2$は説明変数の数 $k$ が増えるだけで数値が上昇してしまうため、 自由度調整済み決定係数を用いる。
# - 単回帰と同様に、決定係数は、最小二乗法を使って求められた回帰方程式の場合には $X, Y$ 間の重相関係数 $R$ との間に $R^2=R^2$(決定係数=重相関係数の二乗) という関係が成り立つ。

# In[3]:


from helpers.regression_analysis import r2
r2.show()


# ## 回帰方程式による区間予測
# ---
# - 標本分布が正規分布なので、正規分布の区間推定と同様に行えば良い。
# - パッケージを使う場合は信頼区間 (confidence interval) と予測区間 (prediction interval) の違いに気をつける。
#     - 信頼区間
#         - 標本抽出を繰り返して母数 (母平均や母回帰係数など) を推定すると一定割合 (95%など) の予測値が収まると考えられる区間
#         - 例： 回帰係数 $a,\ b$ の推定量 $\hat{a},\ \hat{b}$ の推定区間
#     - 予測区間
#         - 標本抽出を繰り返すと一定割合 (95%など) の標本が入ると考えられる区間
#         - 例: $\epsilon$ の分散 $\sigma ^2$ の推定量 $s^2$ の推定区間

# In[4]:


from helpers.regression_analysis import interval
interval.show()


# ## Pythonでの実行方法
# ---
# - statsmodels.regression.linear_model.OLS を用いる。
# - $F$ 統計量 (F-statistic) は全ての係数が0であるかどうか（回帰が成立しているかどうか）の検定統計量であり、まず確認するべき項目である。
#     - Prob (F-statistic)は$F$ 統計量の$p$ 値であり、それが有意水準以下であれば、回帰は成立している。
# - summary にあるそれぞれの変数の P>|t| が $t$ 検定の $p$ 値である。
#     - $p$ 値が有意水準以下であれば、その$\hat{b}$は統計的に優位である。

# In[4]:


help(sm.OLS)


# In[5]:


boston = pd.read_csv('./data/boston.csv')
boston.tail()


# - 回帰モデルが$a=0$となるような特別な制約がない場合は、まず切片を考慮した回帰分析を行う。
#     - 以下で、$a \neq 0$と仮定して回帰を行なった結果、constは統計的に優位であったため、$a \neq 0$の仮定は正しかったと言える。

# In[10]:


Y = boston['中央価格'] # 従属変数
X = boston.loc[:, '犯罪率':'低所得層割合'] # 独立変数
X = sm.add_constant(X) # a=0の場合、この行は不要
model = sm.OLS(Y, X)
fit = model.fit()
fit.summary()


# In[11]:


# （参考: 切片が不要の場合）
Y = boston['中央価格']
X = boston.loc[:, '犯罪率':'低所得層割合']
# X = sm.add_constant(X) # a=0の場合、この行は不要
model = sm.OLS(Y, X)
fit = model.fit()
fit.summary()


# In[ ]:





# In[7]:


help(smf.ols)


# In[8]:


# （参考： formula形式）
import statsmodels.formula.api as smf
model2 = smf.ols(formula='中央価格~{}'.format('+'.join(boston.columns[:-1])), data=boston)
fit2 = model2.fit()
fit2.summary()

