# -*- coding: utf-8 -*-
"""
@Author  : gpf
@License : (C) Copyright 2023
@Time    : 2023/04/18 13:06

"""
#%% 导入相关
import numpy as np
import pandas as pd
import sklearn.tree as st
import sklearn.feature_selection as sf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#%%
# 导入数据集
from sklearn.tree import plot_tree

data = "E:\SoftCup\preprocess_train.csv"

Fault_diagnosis_data = pd.read_csv(data, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108])
# Fault_diagnosis_data = Fault_diagnosis_data.dropna()    # 丢弃含空值的行、列
print(Fault_diagnosis_data.head())

#%%
# 查看数据集信息
Fault_diagnosis_data.info()
print("---------------")
print(Fault_diagnosis_data.keys())
print("---------------")

#%%
#数据预处理
#筛选出有nan和空值的属性
# 统计每列有多少个缺失值(既包括nan也包括空单元格)
# 指定列填充数据

#创建3个列表，分别存放:缺失属性列分别适合用平均值mean、中位数median、众数mode来替换的feature、
feature_mean = ['feature0','feature2','feature4','feature7','feature8','feature11','feature15','feature22','feature23',
                'feature27','feature28','feature30','feature38','feature41','feature42','feature43','feature48','feature49',
                'feature51','feature56','feature58','feature63','feature70','feature71','feature75','feature79','feature81',
                'feature82','feature84','feature85','feature86','feature95','feature96','feature99','feature102','feature106']
feature_median = ['feature3','feature10','feature12','feature17','feature18','feature21','feature24','feature25','feature26',
                'feature29','feature34','feature37','feature40','feature45','feature47','feature50','feature52','feature53',
                'feature55','feature62','feature68','feature69','feature73','feature74','feature83','feature90','feature93',
                'feature98','feature103','feature104']
feature_mode = ['feature1','feature20','feature32','feature54','feature60','feature64','feature65','feature78','feature80',
                'feature88','feature92']
print("feature_mean数组长度：",len(feature_mean))   #36 30 11
print("feature_median数组长度：", len(feature_median))
print("feature_mode数组长度：",len(feature_mode))

#1.
print("初始feature0列空值数量：",Fault_diagnosis_data['feature0'].isna().sum())   #查看未处理的feature0列空值数量
length1 = len(feature_mean)
i = 0
while i < length1:
    Fault_diagnosis_data[feature_mean[i]].fillna(Fault_diagnosis_data[feature_mean[i]].mean(), inplace=True)#使用该列平均值进行空缺值替换
    i += 1
print(Fault_diagnosis_data['feature0'].isna().sum())   #查看使用平均值替换处理后的feature0列空值数量


#2.
print("初始feature3列空值数量：",Fault_diagnosis_data['feature3'].isna().sum())   #查看未处理的feature0列空值数量
length2 = len(feature_median)
j = 0
while j < length2:
    Fault_diagnosis_data[feature_median[j]].fillna(Fault_diagnosis_data[feature_median[j]].median(), inplace=True)
    j += 1
print(Fault_diagnosis_data['feature3'].isna().sum())   #查看使用平均值替换处理后的feature0列空值数量

#3.
print("初始feature1列空值数量：",Fault_diagnosis_data['feature1'].isna().sum())   #查看未处理的feature0列空值数量
length3 = len(feature_mode)
k = 0
while k < length3:
    Fault_diagnosis_data[feature_mode[k]].fillna(Fault_diagnosis_data[feature_mode[k]].mode().iloc[0], inplace=True)
    k += 1
print(Fault_diagnosis_data['feature1'].isna().sum())   #查看使用平均值替换处理后的feature0列空值数量

#%%
#查看经填补空缺值操作之后的数据：
print(Fault_diagnosis_data.isna().sum())   #查看使用平均值替换处理后的feature0列空值数量
# print("所有特征列含空值的数量：",Fault_diagnosis_data.isnull().sum())

#%%
# 载入特征和标签集
X = Fault_diagnosis_data[['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12','feature13','feature14','feature15','feature16','feature17','feature18','feature19','feature20','feature21','feature22','feature23','feature24','feature25','feature26','feature27','feature28','feature29','feature30','feature31','feature32','feature33','feature34','feature35','feature36','feature37','feature38','feature39','feature40','feature41','feature42','feature43','feature44','feature45','feature46','feature47','feature48','feature49','feature50','feature51','feature52','feature53','feature54','feature55','feature56','feature57','feature58','feature59','feature60','feature61','feature62','feature63','feature64','feature65','feature66','feature67','feature68','feature69','feature70','feature71','feature72','feature73','feature74','feature75','feature76','feature77','feature78','feature79','feature80','feature81','feature82','feature83','feature84','feature85','feature86','feature87','feature88','feature89','feature90','feature91','feature92','feature93','feature94','feature95','feature96','feature97','feature98','feature99','feature100','feature101','feature102','feature103','feature104','feature105','feature106']] # 等价于iris_dataset.data
# X = Fault_diagnosis_data.iloc[:, "feature0":Fault_diagnosis_data.columns != "label"]
y = Fault_diagnosis_data['label']
print(X.shape)
print(y.shape)
#%%
# 对标签集进行编码
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# y = encoder.fit_transform(y1)
# print(y)
#y现在为012


#%%
#决策树递归特征消除(rfe)。--- 旨在找到性能最好的特征子集，反复创建模型，
# 通过考虑越来越小的特征集合来递归的选择特征（消除差的留下好的）
# features_linear == X; labels == y
dtc = st.DecisionTreeClassifier()
rfe = sf.RFE(estimator=dtc, n_features_to_select=107,step=1)
#estimator选择回归器,n_features_to_select为选择的特征个数,step为每迭代1次去除多少个特征
X = rfe.fit_transform(X, y)
print(X.shape)
# 2.包裹思想,estimator选择回归器,n_features_to_select为选择的特征个数,step为每迭代1次去除多少个特征,拟合转换
# rfe = RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)
# rfe_ft = rfe.fit_transform(X,Y)
# print("rfe:",rfe_ft)

#%%
#训练集测试集划分
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train shape: {}".format(X_train.shape))        # X_train shape: (4722, 107)
print("y_train shape: {}".format(y_train.shape))        # y_train shape: (4722,)
print("X_test shape: {}".format(X_test.shape))          # X_test shape: (1574, 107)
print("y_test shape: {}".format(y_test.shape))          # y_test shape: (1574,)

#%%
# import importlib
# importlib.reload(plt)
# print(Fault_diagnosis_data.describe())
# fig = Fault_diagnosis_data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
# fig=plt.gcf()
# fig.set_size_inches(10,6)
# plt.show()

#%%
# 生成预处理之后的数据EDA报告
# import ydata_profiling as yp
# report = yp.ProfileReport(Fault_diagnosis_data)
# report.to_file('report2.html')

#%%
#模型搭建

#KNN分类器
#k 近邻算法中 k 的含义是，我们可以考虑训练集中与新数据点最近的任意 k 个邻居（比如说，距离最近的3 个或5 个邻居），
# 而不是只考虑最近的那一个。然后，我们可以用这些邻居中数量最多的类别做出预测。
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

#决策树分类器
#%%
classifier = st.DecisionTreeClassifier(criterion='entropy',splitter="best",max_depth=9,min_impurity_decrease = 0, min_samples_leaf=6)
classifier.fit(X_train, y_train)
score_c = classifier.score(X_test,y_test)
print(score_c)
y_pred = classifier.predict(X_test)
# print(y_pred)
# print(y_test.values)
print("---Classifier is READY---")
# rfc = RandomForestClassifier(n_estimators=25,oob_score=True)
# rfc = rfc.fit(X,y)
# print(rfc.oob_score_)
# rfc = RandomForestClassifier(n_estimators=25)
# rfc = rfc.fit(X_train,y_train)
# score_r = rfc.score(X_test,y_test)
# print(score_r)

from sklearn.model_selection import cross_val_score
# rfc = RandomForestClassifier(n_estimators=100,random_state=100)#实例化
# score_pre = cross_val_score(rfc,X,y,cv=10,scoring= 'accuracy').mean()
# print(score_pre)

#%%
#随机森林和决策树在十组交叉验证下的效果对比
rfc_l = []
clf_l = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc,X,y,cv=10).mean()
    rfc_l.append(rfc_s)
    clf = st.DecisionTreeClassifier()
    clf_s = cross_val_score(clf,X,y,cv=10).mean()
    clf_l.append(clf_s)

plt.plot(range(1,11),rfc_l,label="Random Forest")
plt.plot(range(1,11),clf_l,label="Decision Tree")
plt.legend()
plt.show()

#%%
#学习曲线，找出得分最高的n_estimators  281 maxscore
scorel = []
for i in range(0,400,10):
    rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1,random_state=100)
    score = cross_val_score(rfc,X,y,cv=10).mean()
    scorel.append(score)
    print("当前n_estimators:",i,max(scorel),(scorel.index(max(scorel))*10)+1)#打印最大score及对应n
plt.figure(figsize=[20,5])
plt.plot(range(1,401,10),scorel)
plt.show()

# print(rfc.feature_importances_)
# print(rfc.predict_proba(X_test))
# print(rfc.predict(X_test))

#%%
# #在确定好的范围内，进一步细化学习曲线
# scorel1 = []
# for i in range(275,290):
#     rfc = RandomForestClassifier(n_estimators=i,
#     n_jobs=-1,
#     random_state=100)
#     score = cross_val_score(rfc,X,y,cv=10).mean()
#     scorel1.append(score)
#     print(max(scorel),([*range(275,290)][scorel1.index(max(scorel1))]))
#
# plt.figure(figsize=[20,5])
# plt.plot(range(275,290),scorel1)
# plt.show()

#%%
# param_grid = {'n_estimator':np.arange(0, 200, 10)}
param_grid = {'max_depth':np.arange(1, 20, 1)}
# param_grid = {'max_leaf_nodes':np.arange(25,50,1)}
rfc = RandomForestClassifier(n_estimators=281,random_state=100)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(X,y)
print(GS.best_params_)#显示调整出来的最佳参数              #{'max_depth': 18}
print(GS.best_score_)#返回调整好的最佳参数对应的准确率     #0.8600666212430917

#%%
# param_grid = {'max_features':np.arange(10,107,1)}
# rfc = RandomForestClassifier(n_estimators=281,random_state=100)
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(X,y)
# print(GS.best_params_)
# print(GS.best_score_)

#%%
rfc = RandomForestClassifier(n_estimators=281,criterion='gini',max_depth=18,max_features=107,random_state=100)
score = cross_val_score(rfc,X,y,cv=
10).mean()
print(score)
# 0.8616577081282963max_feature10
# 0.8651490145607792 max_feature107
#%%
rfc.fit(X,y)
y_rf_pred = rfc.predict(X)#使用随机森林（281个决策树分类器）多数投票产生的预测label列表
print(y_rf_pred)
#%%
print(y_pred)

#%%
#用网格搜索调整参数
parameters = {'splitter':('best','random')
                ,'criterion':("gini","entropy")
                ,"max_depth":[*range(1,10)]
                ,'min_samples_leaf':[*range(1,50,5)]
                ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
}
classifier = st.DecisionTreeClassifier(random_state=30)
GS = GridSearchCV(classifier, parameters, cv=10)# cv=10,做10次交叉验证
GS.fit(X_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
# {'criterion': 'entropy', 'max_depth': 9, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 6, 'splitter': 'best'}
# 0.8377741487582442

#%%
#超参数的学习曲线，是一条以超参数的取值为横坐标，模型的度量指标为纵坐标的曲线，它是用来衡量不同超参数取值下模型的表现的线
# import matplotlib.pyplot as plt
# test = []
# for i in range(10):
# 	classifier = st.DecisionTreeClassifier(max_depth=i+1
# 										,criterion="entropy"
# 							 			,random_state=30
# 										,splitter="random"
# 	)
# 	classifier = classifier.fit(X_train, y_train)
# 	score = classifier.score(X_test, y_test)
# 	test.append(score)
# plt.plot(range(1,11),test,color="red",label="max_depth")
# plt.legend()
# plt.show()


#%%
#定义一个numpy数组 species_name
# species_name = np.array(["setosa","versicolor","virginica"])

#对新输入数据做出预测
# X_new = pd.DataFrame([[5, 2.9, 1, 0.2]] , columns=list(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
# 注意：要用pandas中DataFrame类型，这种类型里面包含feature_name，不会有warning；
# 而 X_new = np.array([[5, 2.9, 1, 0.2]])是numpy里的ndarry类型；只有数据,没有feature_name；会有警告，

# print("X_new.shape: {}".format(X_new.shape))
# prediction = knn.predict(X_new)
# print("Prediction: {}".format(prediction))  # Prediction: [0]

# print(y1)   print(y)
# y1 = Fault_diagnosis_data['species']     y = 0,1,2

# print("Predicted species name: {}".format(species_name[prediction]))      # Predicted target name: ['setosa']




#%%

# 多分类问题 混淆矩阵
# 在混淆矩阵中，每一行之和表示该类别的真实样本数量，每一列之和表示被预测为该类别的样本数量。
#
# TP 预测为正样本且真实值为正样本 ； FN 预测为负样本但真实值实际为正样本
# FP 预测为正样本但真实值为负样本 ； TN 预测为负样本且真实值为负样本
# Accuracy = 预测对的样本数占样本总数的比例            Accuracy=TP+TN/(TP+TN+FP+FN)
# Precision = 预测为正的样本中有多少是真正的正样本     Precision=TP/(TP+FP)
# Recall = 真实值为正例的有多少被预测正确了           Recall = TP/(TP+FN)
# F1score=2∗Precision∗Recall/(Precision+Recall)

#评价指标：精确率、召回率
# precision查准率。即正确预测为正的占全部预测为正的比例。个人理解：真正正确的占所有预测为正的比例，就是你以为的正样本,到底猜对了多少.。
# recall查全率。即正确预测为正的占全部实际为正的比例。个人理解：真正正确的占所有实际为正的比例，就是真正的正样本,到底找出了多少.。
"""
将多分类转换成了二分类问题，且需计算每一个类别的预测准确率，召回率，
随后计算平均预测准确率macro_P，平均召回率macro_R，最后的评价指标为：macro_F1 = (2macro_P*macro_R)/(macro_P + macro_R)
将Label为0作正类；label为1,2,3,4,5作负类。
将Label为1作正类；label为0,2,3,4,5作负类。
将Label为2作正类；label为0,1,3,4,5作负类。
将Label为3作正类；label为0,1,2,4,5作负类。
将Label为4作正类；label为0,1,2,3,5作负类。
将Label为5作正类；label为0,1,2,3,4作负类。
"""
#%%
length_train = len(X_train)
length_test = len(X_test)

TP = 0
FP = 0
TN = 0
FN = 0
item = 0

a = 0
#创建列表存放每一个类别的预测准确率和召回率
# accuracy_i = []
precision_i = []
recall_i = []
#循环向列表中写入数据
while a < 6:
    while item < length_test:
        if a == y_pred[item]:
            if a == y_test.values[item]:
                TP += 1
            else:
                FP += 1
        else:
            if a == y_test.values[item]:
                FN += 1
            else:
                TN += 1
        item += 1
    print("Label为%d的指标：TP:%d,TN:%d,FP:%d,FN:%d" %(a,TP,TN,FP,FN))
    # macro_accuracy.append((TP + TN) / (TP + TN + FP + FN))    # 准确率
    precision_i.append(TP / (TP + FP))  # 精确率
    recall_i.append(TP / (TP + FN))  # 召回率
    print(precision_i)
    print(recall_i)
    a += 1
    TP, FP, TN, FN, item = 0, 0, 0, 0, 0

#%%
#模型评价指标
sum = 0
sun = 0
for p in precision_i:
    sum += p
macro_P = sum/len(precision_i)
print("macro_P:",macro_P)

for r in recall_i:
    sun += r
macro_R = sun/len(recall_i)
print("macro_R:",macro_R)

print("macro_F1:",(2*macro_P*macro_R)/(macro_P+macro_R))
print("排行得分：",100*(2*macro_P*macro_R)/(macro_P+macro_R))

#%%
# print('训练集数量:',length_train)
# print('测试集数量:',length_test)
# print("Label1准确率：", accuracy_L1)
# print("Label1精确率：", precision_L1)
# print("Label1召回率：", recall_L1)
# print("Label1-F1-score：",(2*precision_L1*recall_L1)/(precision_L1+recall_L1))

#%%
#将y_pred和y_test写入到csv文件
import pandas as pd
#use pandas
#write date by using the form of dict
df= pd.DataFrame({'y_prediction':y_pred,'y_test':y_test})
df.to_csv("./data/y_pred_test.csv",index=False)


#%%模型评估
print("Test set predictions: \n {}".format(y_pred))

#将测试集预测label输出到csv文件中

print("Test_Set Score: {:.10f}".format(np.mean(y_pred == y_test)))   # Test set score: 0.97   #np.mean函数输出两个矩阵/数组的相似程度

#%%
import importlib
importlib.reload(plt)

#%%
target_names = ["0","1","2","3","4","5"]

plt.rcParams["font.sans-serif"] = ["SimHei", "楷体"]
plt.figure(figsize=(30,30))
plot_tree(classifier,
          feature_names=X,
          class_names=target_names,
          filled=True,
          rounded=True)
plt.show()
plt.savefig('Tree_Visualization')