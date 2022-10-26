## FairRec: Fairness-aware News Recommendation with Decomposed Adversarial Learning 读书笔记

提出具有相同sensitive attribute的用户基本会有相同的新闻点击pattern，新闻推荐系统可能捕捉这些模式，而导致像这个sensitive attributes群体用户全部推荐该pattern的新闻，这就造成了推荐结果的偏差。用户无法获取其他感兴趣的推荐内容，这是不公平的。

提出了一种分解的对抗学习方法来学习无偏差的用户嵌入，该方法用于生成公平感知的新闻推荐结果。

### 问题定义

对于具有敏感属性z的目标用户u，我们假设她单击了N篇新闻文章，这些文章表示为集合D，将u的M个候选新闻集合为Dc。

目标用户点击u的黄金点击标签表示为[y1,y2...yM]，模型预测的带你及标签为[ ^y1,^y2，...，^yM ],候选新闻按预测点击标签排序，排名前k的候选新闻视为推荐结果，即为Dr

Dr的公平性，以对sensitive attribute的鉴别能力来衡量，鉴别越准，说明影响越大，越不公平

### FairRec框架

使用新闻和用户模型NRMS（别人提出的）

用多头自注意网络捕获新闻标题中单词上下文表示，使用专心池网络通过对不同的单词的重要性建模来学习新闻表示。将新闻模型中学到的候选新闻Dc的表示表示为ec，

用户模型从u的点击新闻集合D，使用新闻模型来学习D的表示形式，然后使用多头自我注意网络和专心池网络的组合来获得统一的用户表示形式。将此用户模型学习的无偏差用户嵌入表示为ud

点击评分模块根据无偏差用户嵌入ud和候选新闻嵌入ec计算公平感知排名得分。用点积函数通过评估无偏差用户嵌入和候选新闻嵌入之间的相关性来计算公平感知排名得分^y = ud · ec

#### 正交正则化分解的对抗学习

提出对抗性学习用于从有偏见的数据中学习无偏见的深度表示的技术。

可以通过删除相关用户的sensitive attribute的片偏见信息来学习无偏见的用户嵌入。

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\1.png" alt="1" style="zoom: 67%;" />

将用户兴趣模型分解为两个组件，bias-aware，其主要目的是学习偏差感知的用户嵌入，以捕获敏感用户属性上的偏差信息，和一个bias-free，只将用户感兴趣的与属性无关的信息编码为无偏差的用户嵌入。

将sensitive attribute的预测任务应用到bias-aware用户embedding，属性预测器预测用户的属性z

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\2.png" alt="2" style="zoom: 80%;" />

Wb和bb是参数，^z是预测的概率向量，这里的损失为：zij为第i类第j个用户属性的真实概率，^zij为预测的概率

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\3.png" alt="3" style="zoom: 67%;" />

将对抗也应用到bias-free中，用属性判别器根据bias-free的用户嵌入预测用户属性，

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\4.png" alt="4" style="zoom: 80%;" />

Wd和bd是参数，属性判别器的loss类似预测器

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\5.png" alt="5" style="zoom:67%;" />

为了避免鉴别器从bias-free用户embedding中推断用户属性，使用鉴别器负梯度来惩罚模型

bias-free的embedding经过鉴别器，仍然无法完全去除sensitive attribute的用户属性信息，因此提出正交正则化来进一步净化bias-free的用户embedding

让bias-aware的用户embedding和bias-free的用户embedding彼此正交，正则化损失如下：

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\6.png" alt="6" style="zoom:60%;" />

**u**ib是第i个用户的bias-aware embedding，**u**id是第i个用户的bias-free embedding

#### 模型训练

将两部分用户embedding添加一起形成一个用于训练推荐模型的统一用户embedding，用于训练推荐模型

用户点击Dc的概率表示为^y=u·ec，按照前人提出的使用负抽样技术构建标记样本进行新闻推荐模型训练。即对于用户单击的每个候选新闻，选取T个用户没点击过新闻作为负样本，

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\7.png" alt="7" style="zoom: 67%;" />

^yi是第i个被点击的候选新闻，^yi,j是该候选新闻相关联的第j个负面新闻的点击分数，Nc是用于训练的点击候选新闻数量

推荐模型最终损失函数是新闻推荐，属性预测，正交正则化和对抗损失函数的加权求和，λ是控制相应损失的系数。

<img src="C:\Users\wangkui\Desktop\推荐系统\公平性\截图\8.png" alt="8" style="zoom:67%;" />

