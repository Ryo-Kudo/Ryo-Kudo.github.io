#!/usr/bin/env python
# coding: utf-8

# # ロジットモデル (logit model)
# ---
# - ロジットモデルは、**ダミー変数などの2値変数 (0, 1の値をとる) を目的変数**として、回帰分析を適用する手法のひとつである。
#     - ロジスティック回帰（Logistic Regression）やロジット回帰（Logit　Regression）とも言われる。
# - モデルの出力値をどちらかのカテゴリに所属する確率と見なすことで、**カテゴリ予測**を可能にする。
#     - 以下の例のように、各データがカテゴリA, Bのどちらに所属するかを$A=0, B=1$の2値で表し、予測を行う。
# <!--
# <table class="background-bright border text-center" style="text-align: center">
#     <tr class="background-dark">
#         <th></th>
#         <th>$x_1$</th>
#         <th>$x_2$</th>
#         <th>$\dots$</th>
#         <th>$x_k$</th>
#         <th class="border-right-double" style="min-width: 11em;">$y\ (A=1,\ B=0)$</th>
#         <th>$\hat{y}$</th>
#         <th>予測結果</th>
#     </tr>
#     <tr>
#         <td>$1$</td>
#         <td class="text-right">1.0</td>
#         <td class="text-right">-2.0</td>
#         <td>$\dots$</td>
#         <td class="text-right">3.0</td>
#         <td class="border-right-double">1</td>
#         <td>0.98 (A:98%, B: 2%)</td>
#         <td>A</td>
#     </tr>
#     <tr>
#         <td>$2$</td>
#         <td class="text-right">-2.5</td>
#         <td class="text-right">1.3</td>
#         <td>$\dots$</td>
#         <td class="text-right">1.1</td>
#         <td class="border-right-double">0</td>
#         <td>0.52 (A:52%, B:48%)</td>
#         <td>A</td>
#     </tr>
#     <tr>
#         <td colspan="1">$\vdots$</td>
#         <td colspan="1">$\vdots$</td>
#         <td colspan="1">$\vdots$</td>
#         <td colspan="1"></td>
#         <td colspan="1">$\vdots$</td>
#         <td colspan="1">$\vdots$</td>
#         <td colspan="1">$\vdots$</td>
#         <td colspan="1">$\vdots$</td>
#     </tr>
#     <tr>
#         <td>$n$</td>
#         <td class="text-right">3.3</td>
#         <td class="text-right">0.9</td>
#         <td>$\dots$</td>
#         <td class="text-right">-0.5</td>
#         <td class="border-right-double">0</td>
#         <td>0.36 (A:36%, B:64%)</td>
#         <td>B</td>
#     </tr>
# </table>
# -->

# ![ロジットモデルの例](./image/logit_01.png)

# In[3]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 線形回帰との関係
# ---
# - ロジットモデルでは、 $z=a+b_{1}x_{1}+b_{2}x_{2}+\dots+b_{k}x_{k}$ とすると 出力（予測結果）$\hat{y}$ は以下の式から求める。
#     - $z$は、線形回帰 (単回帰・重回帰) の出力である。
#     - 線形回帰の出力にロジスティック関数 $\Lambda (x)={\displaystyle \frac{e^{x}}{1+e^{x}}}$ を適用したものが$\hat{y}$である。
# 
# $$
# \hat{y} ={\displaystyle \frac{e^{z}}{1+e^{z}}} \left( ={\displaystyle \frac{1}{1+e^{-z}}} \right)
# $$
# 
# 
# - ロジスティック関数は、以下を満たす累積分布関数である。
#     - $-\infty<x<\infty$ の区間で $0<\Lambda(x)<1$
#     - $\Lambda (x)$ を微分した導関数 $\Lambda '(x)=\Lambda(x)\left(1-\Lambda(x)\right)$ は、 $\Lambda'(x)>0$ (確率密度の条件)
# 
# 
# - ロジスティック関数が表す確率分布はロジスティック分布と呼ばれる。

# ![ロジスティック分布](./image/logit_02.png)

# - 線形回帰の出力を確率分布に変換する累積分布関数は、あらゆる実数の入力 ( $-\infty<x<\infty$ ) に対応する確率を出力すれば何でも良い。
# - しかし、一般的にはロジスティック分布 (ロジットモデル) か標準正規分布 (プロビットモデル) が使われる。

# ## 背景 (読み飛ばし可)
# ※数学好きな人向け

# ### 回帰式の意味
# ---
# $y_{i}=1$ となる確率 $p_{i}( X_{i})$ とそれ以外の確率 $1-p_{i}( X_{i})$ の比 (オッズ比) $O_{i}( X_{i})$ を
# 
# $$
# O_{i}( X_{i})={\displaystyle \frac{p_{i}( X_{i})}{1-p_{i}( X_{i})}}
# $$
# 
# として、これを $p_{i}( X_{i})$ について整理すると
# 
# $$
# p_{i}( X_{i}) ={\displaystyle \frac{O_{i}( X_{i})}{1+O_{i}( X_{i})}} 
# $$
# 
# となり、 $p_{i}( X_{i})$ の予測 $\hat{y_{i}}={\displaystyle \frac{e^{z_{i}}}{1+e^{z_{i}}}}$ と一致する。
# 
# 
# 
# つまり、 $e^{z_{i}}$ は $y_{i}=1$ となる確率のオッズ比であり、ロジスティック回帰で求めた $z_{i}$ は対数オッズ比 $logO_{i}( X_{i})$ と解釈できる。
# <!--
# <table class="border text-center background-bright">
#     <tr class="background-dark">
#         <th></th>
#         <th>誤差項の分布</th>
#         <th>特徴</th>
#     </tr>
#     <tr>
#         <th class="background-dark border-bottom">ロジットモデル<br />(ロジスティック回帰)</th>
#         <td>ロジスティック分布</td>
#         <td class="text-left">累積分布関数が計算しやすい<br />回帰式の係数の意味が解釈しやすい (オッズ比)</td>
#     </tr>
#     <tr>
#         <th class="background-dark border-bottom">プロビットモデル</th>
#         <td>標準正規分布</td>
#         <td class="text-left">回帰分析の考え方と親和的である</td>
#     </tr>
# </table>
# -->

# ![ロジットモデルとプロピットモデルの比較](./image/logit_03.png)

# ### 誤差の分布
# ---
# 線形回帰と同じように誤差項 $\epsilon$ を使って、 $A_{i} =a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik} +\epsilon _{i}$ とすると、
# 
# $$
# y_{i} =\left\{\begin{aligned}
#     1 &  & A_{i} +\epsilon _{i} =a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik} +\epsilon _{i}  >0 & , & F( A_{i} +\epsilon _{i})  >0.5\\
#     0 &  & A_{i} +\epsilon _{i} =a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik} +\epsilon _{i} \leqq 0 & , & F( A_{i} +\epsilon _{i}) \leqq 0.5
# \end{aligned}\right.
# $$
# 
# このとき、ロジットモデルは誤差項 $\epsilon$ の分布にロジスティック分布を仮定している。 (プロビットモデルは正規分布)
# 
# ロジットモデルやプロビットモデルのように誤差項の分布とそれに対応する関数を用いて $y$ を $x$ と $\epsilon $ の線型結合に分解して分析する手法を一般化線形モデル (generalized linear model) と呼ぶ。

# ## パラメーターの求め方
# ---
# - 通常、最尤法 (maximum likelihood method)、最尤推定 (maximum likelihood estimation, MLE) を用いる。
# - 最尤法とは、関数 $F(x)$ から元のデータ $(X_i,\ y_i)\ (i=1,\ 2,\dots,\ n)$ が再現される確率 (尤度) を最大化するようにパラメーターを決める方法である。
# 
# $$
# {\displaystyle 尤度=p( y_{1} |x_{1}) \cdot p( y_{2} |x_{2}) \dotsc p( y_{n} |x_{n}) =\prod ^{n}_{i=1} p( y_{i} |x_{i})}
# $$
# 
# - 累積分布関数を $F(x)$ とし、あるデータ $X_{i}\ (x_{1} ,\ x_{2} ,\dots,\ x_{n} )$ が与えられたときに $y_{i} =1$ となる確率を $P(y_{i} =1\ |\ X_{i} )=F(a+b_{1} x_{i1} +b_{2} x_{i2}+\dots +b_{k} x_{ik})$ と表すと、 $P(y_{i} =0\ |\ X_{i})=1-P(y_{i} =1\ |\ X_{i} )$ なので、尤度は以下の尤度関数 $L$ で表わすことができる。
# 
# $$
# L( a,\ b_{1} ,\ b_{2} ,\dots ,\ b_{k}) ={\displaystyle \prod _{\{i\ |\ y_{i} =1\}} F( a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik}) \cdot \prod _{\{i\ |\ y_{i} =0\}}[ 1-F( a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik})]} 
# $$
# 
# - 確率の積の形は、値が小さすぎて、コンピュータで計算しにくいため、対数をとる。
# 
# $$
# log\ L( a,\ b_{1} ,\ b_{2} ,\dots ,\ b_{k}) ={\displaystyle \sum _{\{i\ |\ y_{i} =1\}} log\ F( a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik}) +\sum _{\{i\ |\ y_{i} =0\}}[ 1-F( a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik})]} 
# $$
# 
# - $y_{i} =1$ のとき、 $1-y_{i} =0$、 $y_{i} =0$ のとき、 $1-y_{i} =1$ であるため、上式は下式のようにまとめられる。
# 
# $$
# log\ L( a,\ b_{1} ,\ b_{2} ,\dots ,\ b_{k}) ={\displaystyle \sum ^{n}_{i=1}\{y_{i} \ log\ F( a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik}) +( 1-y_{i}) log\ [ 1-F( a+b_{1} x_{i1} +b_{2} x_{i2} +\dots +b_{k} x_{ik})]\}} 
# $$
# 
# - この対数尤度 $log\ L$ を最大化するパラメーター $a,\ b_{1} ,\ b_{2} ,\dots ,\ b_{k}$ を母数の推定値とするのが最尤法。

# ## パラメーターの標本分布
# ---
# - パラメーターの標本分布を求めるのは困難なので、通常は中心極限定理によって標準分布に近似する。

# ## パラメーターの検定
# ---
# - 自由度 $n-(k+1)$ の $t$ 分布を利用して $t$ 検定を行う。

# ## Pythonでの実行方法
# ---
# - statsmodels.discrete.discrete_model.Logitを用いる。

# In[5]:


# 関数の情報を確認
help(sm.Logit)


# In[7]:


spector = pd.read_csv('./data/spector.csv')
x = sm.add_constant(spector.iloc[:, :3])
y = spector['評価向上']
spector.tail()


# In[10]:


model1 = sm.Logit(y, x)
fit1 = model1.fit()
fit1.summary()


# - 評価向上に値するか否かをロジスティック回帰した結果、（$p = 0.05$）
#     - 定数項: 統計的に優位
#     - GPA: 統計的に優位
#     - 試験成績: **統計的に優位でない**
#     - プログラム参加: 統計的に優位

# - この場合、統計的に優位でない独立変数を減らし、再度回帰を行うことで、合理的な予測値を得る。
# 
# ```
# # 2列目の変数を除き、独立変数に格納する。
# spector = pd.read_csv('./data/spector.csv')
# x = sm.add_constant(spector.iloc[:, [1, 3]]) # 1, 3列のみ
# y = spector['評価向上']
# spector.tail()
# ```
# 
# ```
# # ロジットモデルの実行
# model2 = sm.Logit(y, x)
# fit2 = model2.fit()
# fit2.summary()
# ```
# 
# - 今回は、このままモデルを実行すると"Perfect separation detected, results not available"のエラーとなるため、これ以上の続行は不可能である。
#     - 独立変数"プログラム参加"の値と目的変数"評価向上"の値がほとんど一致してしまっているために生じたエラーである。
#     - サンプルデータのデータ数や独立変数が少なすぎるため、エラーが発生した。
# - *後日、もっとわかりやすい別データに差し替え予定*
# 

# In[ ]:





# ### （参考）その他の記述方法

# In[12]:


# （参考） GLM (Generalized Linear Models) を使う記述方法
# ロジットモデルは、リンク関数にLogit()を用いる一般化回帰モデル(GLM)の一種であるので sm.GLMを使って以下のようにも書くことができる。（当然、結果は一緒である。）
    # 関数の詳細は以下のコマンドで確認できる。
    # help(sm.GLM)
# families.Binomial()は yの値が二項分布であるという意味である。
model2 = sm.GLM(y, x, family=sm.families.Binomial()) 
fit2 = model2.fit()
fit2.summary()


# In[14]:


# （参考） formula 形式: Rと同じように記述できる方式
formula = '評価向上~{}'.format('+'.join(spector.columns[:3]))
model3 = smf.logit(formula, data=spector)
fit3 = model3.fit()
fit3.summary()


# In[15]:


# （参考） formula形式かつ、GLMを利用した書き方
model4 = smf.glm(formula, data=spector, family=sm.families.Binomial())
fit4 = model4.fit()
fit4.summary()

