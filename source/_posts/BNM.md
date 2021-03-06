---
title: (CVPR 2020) BNM
date: 2022-06-06 17:34:18
mathjax: true
tags:
- Transfer Learning
categories:
- Paper Reading
- Transfer Learning

---



<font color=VioletRed>paper</font>: https://arxiv.org/abs/2003.12237
<font color=VioletRed>code</font>:  https://github.com/cuishuhao/BNM

问题：

- 深度网络的学习在很大程度上依赖于带有人类注释标签的数据。在某些标签不足的情况下，当数据密度较大时，决策边界的性能会下降。**<font color=DarkViolet>一种常见的解决方法是直接最小化香农熵，但熵最小化带来的副作用，即减少预测多样性，大多被忽略。</font>**

如何解决：

- 重新研究了随机选择的数据批的分类输出矩阵的结构。通过理论分析发现，预测的可鉴别性和多样性可以通过$batch$输出矩阵的$Frobenius$范数和秩来分别度量。此外，**<font color=DarkViolet>核范数</font>**是$Frobenius$范数的上界，是矩阵秩的凸近似。因此，**<font color=DarkViolet>为了提高可分辨性和多样性，</font>**我们在输出矩阵上提出了$batch$核范数最大化$(BNM)$。

***

## 方法

### $Measuring \  Discriminability  \ with \  F -norm$

首先随机选择$B$个未标记样本的$batch$中的数据预测输出，分类的个数位$C$，则$batch$预测输出矩阵$A \in \mathbb R^{B \times C}$为
$$
\sum_{j=1}^{C}A_{i,j}=1 \quad \forall i \in 1 \dots B  \\
A_{i,j} \ge 0 \quad \forall i \in 1 \dots B,j \in 1 \dots C
$$
深度方法可以通过训练足够数量的标记样本来获得性能良好的响应矩阵$A$。然而，在标签不足的情况下，标签数据和未标签数据之间的差异可能导致边缘数据分布在任务特定决策边界附近的高密度区域。由于模糊样本容易被误分类，我们着重于通过增加可分辨性来优化未标记样本的预测结果。

事实上，判别性越高，预测的不确定性越小。为了测量不确定度，大多数方法采用香农熵，为了简单起见，通常用熵表示。熵的计算方法如下
$$
H(A)=-\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{C}A_{i,j}log(A_{i,j})
$$
我们可以直接最小化$H(A)$，以减少不确定性和具有更多的辨别能力。注意这里的减少不确定性就是说当$H(A)$达到最小值时，$A_i$的每一行只有一个为1，其他$C−1$项为0，最小值恰好满足$A$的最高预测判别能力，其中每个预测$A_i$都是完全确定的。(说白了就是确定是哪个类就是哪个类，不拖泥带水不模糊)

还有其他函数可以提高预测的辨别能力，作者选择了$Frobenius$范数$\left\| A\right||_{F}$
$$
\left\| A\right||_{F}= \sqrt{\sum_{i=1}^{B}\sum_{j=1}^{C} |A_{i,j}|^{2}}
$$
**<font color=DarkViolet>可以证明了$H(A)$和$\left\| A\right||_{F}$具有严格相反的单调性，并且中$H(A)$的最小值和$\left\| A\right||_{F}$的最大值可以达到相同的值。所以就可以替代！</font>**

其中，根据算术均值和几何均值不等式，可计算出$\left\| A\right||_{F}$的上界为:
$$
\left\| A\right||_{F} \le \sqrt{\sum_{i=1}^{B}(\sum_{j=1}^{C}A_{i,j}) \cdot (\sum_{j=1}^{C}A_{i,j})}=\sqrt{\sum_{i=1}^{B}1 \cdot 1}=\sqrt{B}
$$
所以最小化$H(A)$或者最大化$\left\| A\right||_{F}$都可以提升辨别能力

***

###  $Measuring  \ Diversity \  with \  Matrix \  Rank$

在随机抽取的一批$B$样例中，有些类别的样本占主导地位，而有些类别的样本较少甚至没有，这是正常的。**<font color=DarkViolet>在这种情况下，用熵最小化或$F$ -范数最大化训练的模型倾向于将决策边界附近的样本分类为大多数类别。不断收敛到大多数类别，降低了预测的多样性，不利于整体预测精度。</font>**

为了建立预测多样性的模型，我们首先观察矩阵固定批次的$B$未标记样本。**<font color=DarkViolet>预测中的类别数量平均应该是一个常数。如果这个常数变大，预测方法可以获得更多的多样性。</font>**因此，预测多样性可以用$batch$输出矩阵$A$中预测类别的数量来衡量。

进一步分析$A$中的类别数和预测向量。当$A_i$和$A_k$属于不同类别时，随机选择的两个预测向量$A_i$和$A_k$可能是线性无关的。当$A_i$和$A_k$属于同一类别，且$\left\| A\right||_{F}$在$\sqrt{B}$附近时，$A_i$和$A_k$之间的差异很小。那么$A_i$和$A_k$可以近似地被认为是线性相关的。线性无关向量的最大个数称为矩阵秩。因此，如果$\left\| A\right||_{F}$接近上限$\sqrt{B}$, $rank(A)$可以是$A$中预测类别数量的近似值。

所以！

上述分析可知，当$\left\| A\right||_{F}$在$\sqrt{B}$附近时，预测多样性可以用$rank(A)$近似表示。因此，我们可以最大化$rank(A)$来保持预测的多样性。显然，秩(A)的最大值为$min(B, C)$，当$B≥C$时，最大值为$C$，这就有力地保证了该批的预测多样性达到最大值。然而，当$B < C$时，最大值小于$C$，它仍然强制对批样本的预测应该尽可能多样化，尽管不能保证所有类别将被分配到至少一个样本。因此，使$rank(A)$最大化可以在任何情况下保证多样性。

***

### $Batch \ Nuclear-norm \  Maximization$

对于正规矩阵，矩阵秩的计算是一个$NP-hard$非凸问题，我们不能直接约束矩阵$A$的秩。定理表明，当$\left\| A\right||_{F} \le 1$时，$rank(A)$的凸包是核范数$\left\| A\right||_{\star}$。在我们的情形中，与上定理不同，我们有$\left\| A\right||_{F} \le \sqrt{B}$，这样$rank(A)$的凸包就变成了$\left\| A\right||_{\star} /  \sqrt{B}$，这也与$\left\| A\right||_{\star}$成正比，所以可以忽略比例进行近似。同时，当$\left\| A\right||_{F}$接近上界时，$rank(A)$可以近似表示多样性，如上一节所述。因此，当$\left\| A\right||_{F}$在$\sqrt{B}$附近时，预测多样性可以用$\left\| A\right||_{\star}$来近似表示。同时，最大化$\left\| A\right||_{\star}$可以保证更高的预测多样性。

在一些工作中，$\left\| A\right||_{F}$与$\left\| A\right||_{\star}$之间的范围关系可以表示为
$$
\frac{1}{\sqrt{D}}\left\| A\right||_{\star} \le \left\| A\right||_{F} \le \left\| A\right||_{\star} \le \sqrt{D} \cdot \left\| A\right||_{F}
$$
其中$D=min(B, C)$,这说明了$\left\| A\right||_{\star}$和$\left\| A\right||_{F}$是可以相互绑定的。因此，如果$\left\| A\right||_{\star}$变大，那么$\left\| A\right||_{F}$也会变大。最大化$\left\| A\right||_{F}$可以提高上面所述的可辨别性，所以最大化$\left\| A\right||_{\star}$也有助于提高预测可辨别性。

因为$\left\| A\right||_{F}$的上界是$\sqrt{B}$，所以最大化$\left\| A\right||_{\star}$可以
$$
\left\| A\right||_{\star} \le \sqrt{D} \cdot \left\| A\right||_{F} \le \sqrt{D \cdot B}
$$
**<font color=DarkViolet>所以我们可以发现$\left\| A\right||_{\star}$的影响因素可以分为两部分，分别对应方程中的两个不等式条件。第一个不等式对应多样性，第二个对应可辨别性。多样性越大，$rank(A)$越高，$\left\| A\right||_{\star}$越高。同样，当可分辨性变大时，$\left\| A\right||_{F}$会增加，$\left\| A\right||_{\star}$也会变大。</font>**

基于以上研究结果，最大化$\left\| A\right||_{\star}$可以提高预测的可辨别性和多样性。我们提出了$batch$核范数最大化

这里举了一个例子

![image-20220606193048170](BNM/image-20220606193048170.png)

可以看出$\left\| A\right||_{F}$是为了让值不模糊，也就是不存在什么$[0.1,0.9]$这样的，而用了$\left\| A\right||_{\star}$可以理解为又进行了一次筛选



***

### 合并

在任务中，我们给出了标记域$D_L$和未标记域$D_U$。$D_L=\left\{ (x_i^{L},y_i^{L})_{i=1}^{N_L}\right\}$,其中的标签$y_i^{L}=[y_{i1}^{L},y_{i2}^{L},\dots,y_{iC}^{L}] \in \mathbb R^{C}$，其中只有一个类为1，其余都为0，未标记域$D_U=\left\{ (x_i^{U})_{i=1}^{N_U}\right\}$，分类的结果由深度网络得到$A_i=G(x_i)$,在标记域上随机抽取批大小为$B_L$的$\left\{X_L, Y_L\right\}$， $D_L$上的分类损失可计算为:
$$
\mathcal L_{cls}=\frac{1}{B_L}\left\|Y^Llog(G(X^L))\right\|
$$
在无标记域$D_U$上学习,随机抽样的批大小$B_U$样本$\left\{X^U\right\}$，$D_U$上的分类矩阵可以记为$G(X^{U})$,$BNM$的损失函数可以表示为
$$
\mathcal L_{bnm}=-\frac{1}{B_U}\left\|G(X^U)\right\|_{\star}
$$
其中网络$G$是在$D_L$和$D_U$中共享的，最小化$\mathcal L_{bnm}$可以在不损失多样性的情况下降低决策边界附近的数据密度，比典型的熵最小化方法更有效。为了训练网络，我们同时优化分类损失和$BNM$损失，即$\mathcal L_{cls}$和$\mathcal L_{bnm}$可以同时优化，并与参数λ结合如下:
$$
\mathcal L_{all}= \frac{1}{B_L}\left\|Y^Llog(G(X^L))\right\|_1-\frac{1}{B_U}\left\|G(X^U)\right\|_{\star}
$$
通过加强多样性，$BNM$的可能是牺牲一定程度的多数类别预测命中率，以提高少数类别的预测命中率。属于多数类的样本可能会被错误地分类为少数类，以增加多样性。但对标记训练数据的分类损失会惩罚错误鼓励的批量多样性，因为分类损失同时最小化。

在优化中做到双赢！



# 总结

逻辑分析很不错的一篇迁移学习的文章，看的感觉很扎实，但是数学上的推理还需要深入挖掘，真的觉得线性代数和概率论的基础有些差，都忘得差不多了
