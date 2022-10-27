## Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations读书笔记

当前有一类推荐系统利用到了知识图谱，本文自定义了三个指标，将量化后的指标加入到原有的知识图谱推荐系统得到新的推荐结果，近时性、流行度和多样性，三个指标增强了模型的可解释性

解释模型是如何得到推荐或者检索的结果

现有的可解释推荐系统主要是将被推荐的产品和已经体验过的产品之间的关系建模，并据此形成解释类型(用户看了电影A，X导演的的电影A，X还导演过电影B，所以把B推荐给用户)

定义了三个表示解释质量的指标：通过总结文献和用户研究得到（用户认为有价值的属性？？？），确定了用户认可的重要属性，如近时性、流行度和多样性。



### 问题定义

将知识图谱分为一组为一个三元组形式的G={(eh,r,et)}，至少有两种类型的实体存在，一种用户集u∈U⊂E，一种产品集p∈P⊂E，E为实体集，r∈R为关系集

一个路径是实体和关系的交替序列   l={e1,r1,e2,...,en-1,rn,en},而路径中的所有关系定义了路径的模式  πl={r1.r2.r3...r|πl|},路径模式中的最后一个关系r|πl|代表着路径的类型wl

本文是像用户推荐产品，路径为lup，e1是个用户，而en是个产品，每条路径lup都包含三个概念部分，第一，过去交互，e1->e2。第二，实体链，e2->e3.....e|πl|。第三，推荐，即e|πl|+1

用户U对产品的反馈，隐式地从用户的动态中获得抽象成集合(u,p)，或者（u,p,v）三元组显式计算，定义一个用户产品反馈二进制矩阵R，R(u,p)=1说明有用户与产品有交互，否则0。该矩阵用来表示推荐模型的相关性，得到用户和使用他们的产品排名，取top-k,即P~u去推荐。模型对用户有效的推荐用函数CRU:U

取用户到被推荐产品路径里被预测的路径L~up，这些被预测的路径，每一条l都有相应的分数表示有多少概率会取该条l去解释推荐。l^up∈L~up，表示最高概率的那条路径，它将用来解释该推荐。

认为该解释从用户角度去看不一定是很有质量的解释，假设存在用户相关解释属性的一个集合CEX，用C(u):(P~u，L~up)，一个实例解释属性，将附加一个近期的以往交互到该推荐物品的路径中，一个理想的解释θ，最大化目标函数的期望：

<img src="C:\Users\wangkui\Desktop\推荐系统\可解释性\Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations\截图\1.png" alt="1" style="zoom: 67%;" />

CRU根据推荐效用进行优化，并对推荐的产品（以及可解释的路径）进行重新排序，两者结合，推荐程序和解释质量属性的权重相等，以使解释属性最大化，同时考虑推荐效用。

#### EXPLANATION PROPERTY DESIGN

文献研究分析得出将产生的解释概念化为三个属性

以一个路径为例子：ui  *listened* song1 *featuring* artist1 *featuring* song2

##### Recency of the Linking Interaction

用户与所选路径中包含的产品最近产生的交互，即例子中：ui listened song1

能加强推荐模型的效能，对于用户来说，与最近的交互相关的解释在本质上更容易理解

##### Popularity of the Shared Entity

共享实体的流行度，即例子中：artist1

流行度通常抽象成新颖性和熟悉度的概念，即使推荐结果是个不流行的产品，但是解释依然流行的共享实体得到的解释，目的就是让产生的解释更让用户熟悉，用户更容易懂

##### Diversity of the Explanation Type

多样性在这里指的意思是，提供的解释路径的类型越多，最终解释的效果就越好

然后对不同年龄人群用户调查，得出三种属性构想是合理的

#### Operationalization

##### Linking Interaction Recency (LIR)

此属性用于量化自解释路径中的链接交互发生以来的时间。

用户u和他交互的产品集Pu，按时间顺序排序，表示他们的的互动列表，Tu=[(pi,ti)]，ti是发生交互的时间i=1.....|Pu|

将指数加权的移动平均值应用于包含在 𝑇𝑢 中的时间戳，获得用户u的每一个交互的LIR

给定一个交互（pi，ti）∈Tu，该交互的LIR计算为：

<img src="C:\Users\wangkui\Desktop\推荐系统\可解释性\Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations\截图\2.png" alt="2" style="zoom: 50%;" />

βLIR∈[0,1]，随着交互时间衰减，LIR(p1,t1)=t1，每个用户的LIR在[0,1]范围，靠近0意味着交互连接时间距离很远，否则为1

整体的推荐列表的解释的LIR，通过平均，选中的每个推荐产品的解释路径的交互连接的LIR得到的

##### Shared Entity Popularity (SEP)

假设该共享实体在KG中设计的关系数量来代表它受欢迎的程度

给定类型λ的实体列表，根据受欢迎程度，Sλ=[(ei,vi)]，ei∈Eλ是类型λ下的一个实体，vi是一个共享实体所涉及的关系数量（度），衰减应用于Sλ，得到类型λ下的一个实体的SEP：

<img src="C:\Users\wangkui\Desktop\推荐系统\可解释性\Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations\截图\3.png" alt="3" style="zoom: 80%;" />

βSEP和流行度相关的衰减，SEP(e1,v1)=v1，SEP值同样属于[0,1]，约低的流行度就趋向0，反之趋向1

整体的一个推荐列表得解释的SEP，通过平均每个推荐产品的选定路径中共享实体的SEP值得到。

##### Explanation Type Diversity (ETD)

此属性用于量化推荐产品附带的不同类型的解释。

给定top-k清单，推荐给用户u，~Pu表示产品，对应的解释路径为^Lu

用ω^Lu={ωl|l∈^Lu}表示，用户u的推荐列表解释路径集合的类型

用户u的ETD计算：在选中的解释关系中在推荐列表大小k和可能的解释类型的总数，两者最小值，作为唯一类型的数量

<img src="C:\Users\wangkui\Desktop\推荐系统\可解释性\Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations\截图\4.png" alt="4" style="zoom:67%;" />

#### EXPLANATION PROPERTY OPTIMIZATION

重新排列推荐模型输出的推荐列表（和解释），这方法容易拓展包含新属性。

提出两种重新排序的方法

##### Soft Optimization

目标是调整推荐产品的原始解释路径列表，以便所选路径更好地满足目标解释属性。

给定用户u和p是推荐列表里的物品，认为L~up，用户产品路径由推荐模型θ生成，目标是优化一个属性（LIR OR SEP）联合特别的推荐物品的解释。

仅需要计算预测列表中每个解释路径对应的目标属性值，选择值最高的解释路径，soft优化去改变属性的计算方式：CP~iu是在清单top-i位置的属性C的值。还可以通过软优化一起优化多个解释，例如C = {LIR,SEP}，在用户u在i位置，为推荐产品寻找一条路径p∈P~u 的

根据解释属性去对每个推荐产品的解释路径重新排序，但是不包括最初的推荐产品。

<img src="C:\Users\wangkui\Desktop\推荐系统\可解释性\Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations\截图\5.png" alt="5" style="zoom: 67%;" />

##### Weighted Optimization

提高解释质量，对推荐产品和相关解释路径进行重新排序，对于每个用户u，目标是从推荐给u的k个产品确定一个最佳的集合P~u，利用一个最大maximum marginal relevance的方法，将目标属性作为支持度量。

对于每个排序i的位置的推荐产品，我们计算加权求和(1)u和产品的相关性，(2)当前u的推荐清单的范围，将满足目标属性，如果我们再推荐时包含这个产品，通过^liup得到。

权重α分配给目标属性量化，目标属性产品对用户的相关性有多么的重要，计算了所有的加权得分，取最高得分的产品，将该产品添加到第i位置上推荐给u的产品，重复至位置k。集合P~u里，i位置的优化：

<img src="C:\Users\wangkui\Desktop\推荐系统\可解释性\Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations\截图\6.png" alt="6" style="zoom:80%;" />

位置0为空集，α∈[0,1]是表示相关性和目标属性之间的权衡参数。当α=0，则直接输出原始模型输出，不考虑目标属性。当α=1，抛弃推荐模型的输出，将目标属性最大化。

