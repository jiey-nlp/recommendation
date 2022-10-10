## RecGURU:Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation读书笔记

序列存在数据稀疏的问题，而跨域推荐来解决该类问题，但是推荐跨域推荐目前主要基于重叠用户的数据进行迁移学习，在许多实际程序中，存在重叠用户的数量通常不足的问题。

### 问题定义

单一领域内的基于隐式反馈的序列推荐问题。

定义 U={u1,...,u|U|}和 V={v1,...,v|V|}分别表示用户集和物品集。其中 |U|和 |V|分别表示用户和物品的总数量。对于每一个用户ui,∀i∈{1,...,|U|}，其中每一个用户所交互的物品按照时间顺序（行为序列）即可表示为 si={vi1,...,vi|si|}，其中|si|表示序列 si的长度。

具体来说，序列推荐的目的就是判断用户下一个要购买的物品，也即给定一个用户 ui，序列推荐就是通过分析其购买历史记录来预测下一次也就是 |si|+1次会购买什么，也就是预测商品 vi|si|+1最有可能是什么。其过程可以通过一个条件概率来进行表达:

![1](https://user-images.githubusercontent.com/91814991/194822505-02f52161-a7bf-4ff0-9ad4-b0cf39b7b53b.png)

在跨域推荐中，为了提高性能，考虑其他域的用户信息，如果A,B两个域拥有重叠用户，跨域序列推荐会利用B领域的信息来增强对A领域物品的推荐，同理A=>B，用户 ui和其在领域A和B的物品序列分别siA，siB，A中候选物品概率：

![2](https://user-images.githubusercontent.com/91814991/194822537-25d1aff3-59c7-4f56-9076-3112371c0cb9.png)

A,B没有用户重叠，用户在A中有行为，而B中没有，则通过B中的隐含信息来增强领域A中的性能，概率表达：

![3](https://user-images.githubusercontent.com/91814991/194822559-dad32d47-1851-4590-9449-004b59352989.png)

其中 SB={sjB},∀uj∈UB指领域B中所用用户的序列集合， info(·)表示一个用来从领域B中提取任何对A领域有用的信息的提取模型。

### 方法提出

提出RecGURU主要包括两部分:

(1)广义用户表示单元(GURU)以获得每个用户的广义用户表示(GUR)

(2)将GUR作为输入的跨域顺序推荐(CDSRec)单元实现序列推荐任务中的跨域协作。

![4](https://user-images.githubusercontent.com/91814991/194822592-954c250e-5660-443a-a136-4b3c9fb7ef68.png)

将行为序列送入embedding层得到嵌入向量，包含项目信息和序列位置信息，GURU将得到的嵌入向量A,B作为输入，得到ui，uj的潜在的用户表示hia，hib，为了保证用户表示有意义有信息，decoder将重构输入序列，encoder和decoder构成一个自动编码器框架。

为了得到综合了源域和目标域信息的广义表示，让鉴别器无法分辨出两个域的用户表示hia和hib（拉近两个域）

##### Generalized User Representations

在做用户泛化表示时的思路是先对单域内的用户进行建模表示，信息来自其他域，通过域鉴别器的实现正则化，之后得到一般用户表示

##### User Representations in Single Domain

使用一个自动编码器来学习潜在的用户表示，它能够重建用户的原始输入序列。自动编码器可以通过重建任务生成有意义的表示，以提高性能。框架中的自动编码器由嵌入模块、编码器模块和解码器模块组成。

给定用户ui和他的行为序列，得到：（粗体是向量表示）

![5](https://user-images.githubusercontent.com/91814991/194822634-2d3b870d-36d5-485f-a812-9b3a9ff5cfe6.png)

Embed有由一个项目嵌入层和一个位置嵌入层组成，由两个可训练的嵌入矩阵，以合并项目信息和行为序列的序列信息，将项目嵌入和位置嵌入的输出做point-wise相加得到eit

这里的Encoder是一种结构类似于Transformer，它包含了多个相同的transformer层，每个transformer层包含一个多头双向自注意力层和一个position-wise的全连接前馈层。最后[eos]的隐含表示 hi 作为整个序列的总体表示。送入Decoder。

重构decoder操作的的重构损失

![6](https://user-images.githubusercontent.com/91814991/194822652-c3e57b51-66fc-440a-96ba-30c0a49aa112.png)

si是自编码器的输入，s^i是重构的输出，hi是encoder的输出用户的潜在表示，对于任何位置t，如果是原词是【pad】则在重构损失中忽略这个位置

##### Generalizing User Representation Across Domains

自动编码器整合先验知识的常用技术是在重建目标函数中添加额外的惩罚项。为了从两个域中合并知识，添加KL散度进重构损失，测量两个领域中学习到的用户表示分布之间的距离

![8](https://user-images.githubusercontent.com/91814991/194822671-6b082051-41b2-4b2c-9f75-8d0c3a6c4857.png)

ρA，ρB表示AB域的潜在用户表示的分布，为了最小化两个分部之间的KL差异，采用对抗训练。encoder生成潜在用户表示hia~ρa，hib~ρb，分布由域A,B中的编码器参数表征。

为二进制分类任务构建鉴别器，输入是域AB中一个单一潜在表征，输出是表示来源哪个域的预测。采用负对数似然损失作为对抗优化过程的目标函数，表示为

![9](https://user-images.githubusercontent.com/91814991/194822681-303fb51e-73df-48f5-92d0-8503bd212138.png)

I(·)在ui是Ua的时候，值为1，否则为0。f（hi）的值由域鉴别器f(.)计算。

交替更新域鉴别器和编码器，让鉴别器最终无法区分来自A或B域。通过对潜在用户表示的分布约束实现了信息共享，而不依赖重叠用户信息。

添加l2惩罚项，强制让域重叠用户加强显式表示共享，就是让同一用户的表示在不同域应该是相同的或者相近的。
![10](https://user-images.githubusercontent.com/91814991/194822835-8f3a92d9-d155-490f-ad30-3dea721d6a7b.png)


##### Cross-domain Sequential Recommendation

GURU提取的广义的用户表示反映了不同领域的用户的总体偏好，有利于特定领域的推荐任务。

最终预测：将广义用户表示hi和近期项目序列V作为输入序列，输出是最新一次的用户偏好向量qi，|si|

![12](https://user-images.githubusercontent.com/91814991/194822861-d0b507db-b4d3-473a-bf68-4112f073f97a.png)

![13](https://user-images.githubusercontent.com/91814991/194822880-0eba6029-f4a9-49cb-9738-0784bb4b0bb5.png)

Iv是v的项目嵌入，根据偏好得分对候选项目进行排名和推荐。我们用Bayesian Personalized Ranking(BRP)损失去训练推荐模型，给一个来自A域的用户ui，计算在时间顺序t时的项目推荐，损失为

![14](https://user-images.githubusercontent.com/91814991/194822896-7da3490f-6e88-4fe7-8210-8ceba8086edf.png)

v是目标项目，Iv是它对应的嵌入，Ns是一组阴性训练样本

##### Training Strategy

在第一阶段，我们使用重建任务分别对每个域中的自动编码器进行预训练。通过预训练过程，重建损失大大减少，为后续对抗训练提供了一个良好的开端。

第二阶段遵循gan的训练原则，在每次迭代过程中首先优化鉴别器Ldis

![15](https://user-images.githubusercontent.com/91814991/194822912-732e1e1c-3338-4a88-bd16-15f64c784a71.png)

最终重叠用户的LOSS，重构loss，鉴别器loss和l2多任务联合最小化损失：

![16](https://user-images.githubusercontent.com/91814991/194822935-65996692-d739-431f-94dc-a30bb0322f17.png)

第三阶段使用下一个项目推荐任务微调每个单独域中的CDSRec模型。

#### 实验

![17](https://user-images.githubusercontent.com/91814991/194822952-c4ac8365-c865-43ff-9328-8a8dec27b9b7.png)

![18](https://user-images.githubusercontent.com/91814991/194822963-e93aea7c-6cab-4bfa-9123-5c13f418d63e.png)
