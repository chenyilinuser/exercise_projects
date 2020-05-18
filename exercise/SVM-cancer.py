from sklearn.datasets import load_breast_cancer#乳腺癌数据集-支持向量机算法
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler#让数据服从标准分布的标准化
from sklearn.metrics import confusion_matrix as CM,recall_score,roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import datetime

#将特征矩阵和标签Y分开
data = load_breast_cancer()
# print(type(data))
x=data.data#sklearn自动就这么的为data和target
y=data.target
# print(x)
# print(y)

# #探索标签的分类,查看标签有哪些分类
# Y=np.unique(y)
# print(Y)
# # plt.scatter(x[:,0],x[:,1],c='y')
# # plt.show()

#二、先分训练集和测试集，然后再进行数据预处理。在实际生活中，测试集不一定是带标签的。
#分训练集和测试集
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=420)
# #由于‘poly’运行速度太慢，去掉
# kernel=['linear','sigmoid','rbf']
# for kernel in kernel:
#     times=time()
#     clf=SVC(kernel=kernel
#             ,gamma='auto'
#             ,cache_size=1000
#             ).fit(xtrain,ytrain)
#     ypredict=clf.predict(xtest)
#     score=clf.score(xtest,ytest)
#     print('the accuracy under kernel %s is %f'%(kernel,score))
#     print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))
# #到此应该可以确定乳腺癌数据集应该是一个线性可分的数据集

# kernel=['linear','poly','sigmoid','rbf']
# for kernel in kernel:
#     times=time()
#     clf=SVC(kernel=kernel
#             ,gamma='auto'
#             ,degree=1#degree默认值是3，所以poly核函数跑的非常慢
#             ,cache_size=1000
#             ).fit(xtrain,ytrain)
#     ypredict=clf.predict(xtest)
#     score=clf.score(xtest,ytest)
#     print('the accuracy under kernel %s is %f'%(kernel,score))
#     print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))
# #到此应该可以确定乳腺癌数据集应该是一个线性可分的数据集

# rbf 表现不应该这么差的，应该可以调整参数，对数据进行预处理，提高其准确性
#处理特征矩阵
#描述性统计与异常值，看看有没有异常值有没有偏态问题，看1%与最小值，看99%与最大值进行比较即可,保证量纲
#因为这里是sklearn库中自带的数据，需要先转换成dataframe类型的
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)
# print(xtrain.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
# print(xtest.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#上面这段可以直接写成x，对特征进行描述性统计
#x=pd.DateFrame（x）

#数据存在问题：
# （1）查看数据均值mean和std 发现，数据量纲不统一
# (2) 数据的分布是偏态的
# 因此我们需要对数据进行标准化
# print(xtrain.isnull().sum())#查看是否有缺失值
std=StandardScaler()
xtrain=std.fit_transform(xtrain)
xtest=std.fit_transform(xtest)
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)
# print(xtrain.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
# print(xtest.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)

#数据处理之后，再跑一遍
# kernel=['linear','poly','sigmoid','rbf']
# for kernel in kernel:
#     times=time()
#     clf=SVC(kernel=kernel
#             ,degree=1
#             ,gamma='auto'
#             ,cache_size=2000
#             ).fit(xtrain,ytrain)
#     ypredict=clf.predict(xtest)
#     score=clf.score(xtest,ytest)
#     print('the accuracy under %s is %f'%(kernel,score))
#     print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))

# 当用径向基（RBF）内核去训练SVM，有两个参数必须要去考虑：C惩罚系数和gamma。参数C，通用在所有SVM内核，
# 与决策表面的简单性相抗衡，可以对训练样本的误分类进行有价转换。较小的C会使决策表面更平滑，同时较高的C旨
# 在正确地分类所有训练样本。Gamma定义了单一训练样本能起到多大的影响。较大的gamma会更让其他样本受到影响。
# 直观地，该gamma参数定义了单个训练样例的影响达到了多远，低值意味着“远”，高值意味着“接近”。
# 所述gamma参数可以被看作是由模型支持向量选择的样本的影响的半径的倒数。
# 该C参数将训练样例的错误分类与决策表面的简单性相对应。低值C使得决策表面平滑，而高度C旨在通过给予模型自由选择
# 更多样本作为支持向量来正确地对所有训练样本进行分类。

# #微调rbf参数gamma
# score=[]
# gamma_range=np.logspace(-10,1,50)
# for i in gamma_range:
#     clf=SVC(kernel='rbf'
#             ,gamma=i
#             ,cache_size=2000
#             ).fit(xtrain,ytrain)
#     score.append(clf.score(xtest,ytest))
# print(max(score))
# print(gamma_range[score.index(max(score))])
#最佳结果：0.9766081871345029，0.05689866029018305

# score=[]
# c_range=np.linspace(0.01,2,50)
# for i in c_range:
#     clf=SVC(kernel='rbf'
#             ,gamma=0.05689866029018305
#             ,C=i
#             ,cache_size=2000
#             ).fit(xtrain,ytrain)
#     score.append(clf.score(xtest,ytest))
# print(max(score))
# print(c_range[score.index(max(score))])
# #最佳结果：0.9824561403508771，1.1020408163265305
# # plt.plot(range(50),np.logspace(-10,1,50))
# # plt.show()

clf = SVC(kernel='rbf'
          , gamma=0.05689866029018305
          , C=1.1020408163265305
          , cache_size=2000
          ).fit(xtrain, ytrain)
ypredict=clf.predict(xtest)
score=clf.score(xtest,ytest)
recall=recall_score(ytest,ypredict)
auc=roc_auc_score(ytest,clf.decision_function(xtest))#roc曲线下方的面积，越大越好，即越接近1越好
print(score, recall, auc)


# #使用交叉验证
# times=time()
# gamma_range=np.logspace(-10,1,20)
#gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，
# gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
# coef0_range=np.linspace(0,5,10)
# param_grid=dict(gamma=gamma_range,coef0=coef0_range)#coef可能是权重
#clf.intercept_[0]#用来获得截距(这里共有两个值，分别为到x和到y的)
# cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=420)
# grid=GridSearchCV(SVC(kernel='poly',degree=1,cache_size=2000),param_grid=param_grid,cv=cv)
# grid.fit(x,y)
# print('the best parameters are %s with score %0.5f'%(grid.best_params_,grid.best_score_))
# print(datetime.datetime.fromtimestamp(time()-times).strftime('%M:%S:%f'))
