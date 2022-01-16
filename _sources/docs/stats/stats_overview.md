# Stats Overview

（2022.Jan 日々、追記中…）

ビジネスデータからデータマイニングをする機会や、初めての人にやり方を教える機会が多い。データ分析には数多くの手法があるが、その内の基本的なものについて、**重要な点を短時間で復習するための情報**を、本ページに記述する。

*（例えば、正規分布、中心極限定理、信頼区間、検定などの話はここでは触れない。統計学の基礎については、良い書籍やwebがたくさんあるのでそちらを見た方が良い。）*

## Data Mining とは？

"データマイニングとは、変数間の一貫した規則性や関係性を求め、データを探索するための分析プロセスである。さらに、その発見した規則性を新しいデータセットに適用することで、その発見の正確さを検証する。データマイニングの最終的な目的は、予測である。" (https://www.tibco.com/reference-center/what-is-data-mining)

具体的には、データマイニングで作成した予測モデルは、企業や個人の意思決定に活用される。特に、マーケティングの課題解決でのニーズが高い。

統計学やデータマイニングは、MBAでも必修科目に入るほど、有効なビジネス課題解決手法として認識されている。学ばない手はない。

## Methods

<!--
:::{figure-md} stat_methods
![Methods](/assets/media/stat_01.png)

Methods / toolkit
:::
-->


| Regression (従属変数が連続) | Classification (従属変数がカテゴリー) | 
|----------|-------------|
| Linear Regression | Logistic Regression |
| - | Multinominal Logistic Regression |
| Regression Tree | Classification Tree |
| Random Forest | Random Forest |
| Artificial Neural Network | Artificial Neural Network |
| Lasso Regression | Linear Discriminant Analysis |
| ... | ... |  


### 線形回帰, Linear Regression

$$
y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \cdots + \beta_k x_{ki} + \epsilon = \hat{y_i} + \epsilon_i
$$

- 従属連続変数 (Dependent continuous variable): $y$
- 独立変数 (Independent / explanatory variables): $x_1, x_2, \cdots , x_k$
- Goal: 以下のSSEを最小にするような$\beta_0, \beta_1, \cdots , \beta_k$ のセットを見つける。
    - SSE: $\Sigma ^n _{i=1} \epsilon ^2 _i = \Sigma ^n _{i=1} (y_i - \hat{y_i})^2 $

- Example: $\hat{y} = 10 + 2x_1 + 5x_2$

```
# Python コードを追記予定
```

- 注意点
    - 多重共線性, Multicolinerarity
    - 内生性・同時性, Endogeneity / Simultaneity

### ロジスティック回帰, Logistic Regression 

$$
P(Y=1) = f(\beta_0 + \beta_1 x_{1} + \beta_2 x_{2} + \cdots + \beta_k x_{k})
$$

$$
where \cdots  f(z) = \dfrac{1}{1+e^{-z}}
$$

- $f(z)$ はロジスティック関数
- Goal: （大雑把に言えば）良い予測を実現する $\beta_0, \beta_1, \cdots , \beta_k$ のセットを見つける。


### 回帰木, Regression Tree

### 分類木, Classification Tree

### ランダムフォレスト, Random Forest

:::{figure-md} stat_random_forest
![Random Forest](/assets/media/stat_02.png)

Random Forest
:::

## モデル評価方法 

### OSR$^2$, Out of Sample R$^2$
- $y$が連続値の時に利用できる。

$$
OSR^2 = 1- \dfrac{\Sigma ^n _{i=1} ( y_{test, i} - \hat{y}_{test, i} )^2 }{\Sigma ^n _{i=1} ( y_{test, i} - \bar{y}_{training} )^2}
$$



### Mean Absolute Error
- $y$が連続値の時に利用できる。

$$
MAE = 1- \dfrac{\Sigma ^n _{i=1} | y_{test, i} - \hat{y}_{test, i} | }{n}
$$



### 混合行列, The Confusion Matrix
- $y$がカテゴリ値の時に利用できる。

:::{figure-md} confusion_matrix
![The Confusion Matrix](/assets/media/stat_03.png)

The Confusion Matrix
:::

- Matrixの内容
    - True Positives
        - 予測値$\hat{Y}=1$、実測値$Y=1$
    - True Negatives
        - 予測値$\hat{Y}=0$、実測値$Y=0$
    - False Positives (Type I error)
        - 予測値$\hat{Y}=1$、実測値$Y=0$
    - False Negatives (Type II error)
        - 予測値$\hat{Y}=1$、実測値$Y=0$
- Matrix評価のための重要な3指標
    - Accuracy: $\dfrac{TN + TP}{TOTAL}$  

    - Sensitivity: $\dfrac{TP}{TP + FN} = \dfrac{TP}{Actual \, Positives}$  

    - Specificity: $\dfrac{TN}{TN + FP} = \dfrac{TN}{Actual \, Negatives}$  

- Goal: 基本的には $TP$, $TN$ を可能な限り増やし、$FP$, $FN$ を可能な限り減らす。
    - 実際には、$FP$, $FN$がビジネスに及ぼす影響を理解し、どちらかの増大を許容することもある。


## （参考）

２つの事象に相関関係が見られたからといって、そこに因果関係があるとは限らない。

:::{figure-md} tylervigen
![モッツァレラチーズの消費量と土木工学博士号の授与数](/assets/media/stat_04.png)

モッツァレラチーズの消費量と土木工学博士号の授与数 (http://www.tylervigen.com/spurious-correlations)
:::

- 疑似相関に注意し、考えながら分析を行う必要がある。