# -*- coding: utf-8 -*-
"""
@Author  : gpf
@License : (C) Copyright 2023
@Time    : 2023/04/18 13:06

"""
# %% 导入相关包
import numpy as np
import pandas as pd
# import pybaobabdt as pybaobabdt
# import sklearn.tree as st
# import sklearn.feature_selection as sf
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree, DecisionTreeClassifier  # 树图
# from sklearn import datasets
# from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
# %%
# 导入数据集
from sklearn.tree import plot_tree

data = "C:\\Users\\Rui\\Desktop\\A10\\preprocess_train.csv"

Fault_diagnosis_data = pd.read_csv(data,usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                            59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                                            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108])
# Fault_diagnosis_data = Fault_diagnosis_data.dropna()    # 丢弃含空值的行、列
print(Fault_diagnosis_data.head())

# %%
# 查看数据集信息
Fault_diagnosis_data.info()
print("---------------")
print(Fault_diagnosis_data.keys())
print("---------------")

# %%
# 数据预处理
# 筛选出有nan和空值的属性
# 统计每列有多少个缺失值(既包括nan也包括空单元格)
# 指定列填充数据

# 创建3个列表，分别存放:缺失属性列分别适合用平均值mean、中位数median、众数mode来替换的feature、
feature_mean = ['feature0', 'feature2', 'feature4', 'feature7', 'feature8', 'feature11', 'feature15', 'feature22',
                'feature23',
                'feature27', 'feature28', 'feature30', 'feature38', 'feature41', 'feature42', 'feature43', 'feature48',
                'feature49',
                'feature51', 'feature56', 'feature58', 'feature63', 'feature70', 'feature71', 'feature75', 'feature79',
                'feature81',
                'feature82', 'feature84', 'feature85', 'feature86', 'feature95', 'feature96', 'feature99', 'feature102',
                'feature106']
feature_median = ['feature3', 'feature10', 'feature12', 'feature17', 'feature18', 'feature21', 'feature24', 'feature25',
                  'feature26',
                  'feature29', 'feature34', 'feature37', 'feature40', 'feature45', 'feature47', 'feature50',
                  'feature52', 'feature53',
                  'feature55', 'feature62', 'feature68', 'feature69', 'feature73', 'feature74', 'feature83',
                  'feature90', 'feature93',
                  'feature98', 'feature103', 'feature104']
feature_mode = ['feature1', 'feature20', 'feature32', 'feature54', 'feature60', 'feature64', 'feature65', 'feature78',
                'feature80',
                'feature88', 'feature92']
print("feature_mean数组长度：", len(feature_mean))  # 36 30 11
print("feature_median数组长度：", len(feature_median))
print("feature_mode数组长度：", len(feature_mode))
# 1.
print("初始feature0列空值数量：", Fault_diagnosis_data['feature0'].isna().sum())  # 查看未处理的feature0列空值数量
length1 = len(feature_mean)
i = 0
while i < length1:
    Fault_diagnosis_data[feature_mean[i]].fillna(Fault_diagnosis_data[feature_mean[i]].mean(),
                                                 inplace=True)  # 使用该列平均值进行空缺值替换
    i += 1
print(Fault_diagnosis_data['feature0'].isna().sum())  # 查看使用平均值替换处理后的feature0列空值数量

# 2.
print("初始feature3列空值数量：", Fault_diagnosis_data['feature3'].isna().sum())  # 查看未处理的feature0列空值数量
length2 = len(feature_median)
j = 0
while j < length2:
    Fault_diagnosis_data[feature_median[j]].fillna(Fault_diagnosis_data[feature_median[j]].median(), inplace=True)
    j += 1
print(Fault_diagnosis_data['feature3'].isna().sum())  # 查看使用平均值替换处理后的feature0列空值数量

# 3.
print("初始feature1列空值数量：", Fault_diagnosis_data['feature1'].isna().sum())  # 查看未处理的feature0列空值数量
length3 = len(feature_mode)
k = 0
while k < length3:
    Fault_diagnosis_data[feature_mode[k]].fillna(Fault_diagnosis_data[feature_mode[k]].mode().iloc[0], inplace=True)
    k += 1
print(Fault_diagnosis_data['feature1'].isna().sum())  # 查看使用平均值替换处理后的feature0列空值数量

# %%
# 查看经填补空缺值操作之后的数据：
print(Fault_diagnosis_data.isna().sum())  # 查看使用平均值替换处理后的feature0列空值数量
# print("所有特征列含空值的数量：",Fault_diagnosis_data.isnull().sum())

# %%
# 载入特征和标签集
X = Fault_diagnosis_data[
    ['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
     'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17',
     'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25',
     'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33',
     'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41',
     'feature42', 'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49',
     'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57',
     'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65',
     'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73',
     'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature81',
     'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89',
     'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97',
     'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105',
     'feature106']]  # 等价于iris_dataset.data
# X = Fault_diagnosis_data.iloc[:, "feature0":Fault_diagnosis_data.columns != "label"]
y = Fault_diagnosis_data['label']
print(X.shape)
print(y.shape)
# %%
# 随机森林和决策树在十组交叉验证下的效果对比
# from sklearn.model_selection import cross_val_score
# rfc_l = []
# clf_l = []
# for i in range(10):
#     rfc = RandomForestClassifier(n_estimators=25)
#     rfc_s = cross_val_score(rfc,X,y,cv=10).mean()
#     rfc_l.append(rfc_s)
#     clf = st.DecisionTreeClassifier()
#     clf_s = cross_val_score(clf,X,y,cv=10).mean()
#     clf_l.append(clf_s)
#
# plt.plot(range(1,11),rfc_l,label="Random Forest")
# plt.plot(range(1,11),clf_l,label="Decision Tree")
# plt.legend()
# plt.show()

# %%
# 学习曲线，找出得分最高的n_estimators  281 maxscore
# scorel = []
# for i in range(0,400,10):
#     rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1,random_state=100)
#     score = cross_val_score(rfc,X,y,cv=10).mean()
#     scorel.append(score)
#     print("当前n_estimators:",i,max(scorel),(scorel.index(max(scorel))*10)+1)#打印最大score及对应n
# plt.figure(figsize=[20,5])
# plt.plot(range(1,401,10),scorel)
# plt.show()

# print(rfc.feature_importances_)
# print(rfc.predict_proba(X_test))
# print(rfc.predict(X_test))


# %%
# param_grid = {'n_estimator':np.arange(0, 200, 10)}
# param_grid = {'max_depth':np.arange(1, 20, 1)}
# # param_grid = {'max_leaf_nodes':np.arange(25,50,1)}
# rfc = RandomForestClassifier(n_estimators=281,random_state=100)
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(X,y)
# print(GS.best_params_)#显示调整出来的最佳参数              #{'max_depth': 18}
# print(GS.best_score_)#返回调整好的最佳参数对应的准确率     #0.8600666212430917

#%%
#定义训练集
X_rf_train = X[:5600]
y_rf_train = y[:5600]

#取测试集
X_rf_test = X[5600:6296]
y_rf_test = y[5600:6296]
print(X_rf_test)
print(y_rf_test)
# %%
# 建立随机森林模型
rfc = RandomForestClassifier(n_estimators=281, criterion='gini', max_depth=18, max_features=107, random_state=100)
rfc.fit(X_rf_train, y_rf_train)#模型训练
#%%
y_rf_pred = rfc.predict(X_rf_test)  # 使用随机森林（281个决策树分类器）多数投票产生的预测label列表(测试集改为1000条样本)
print(y_rf_pred)
# score = cross_val_score(rfc,X,y,cv=10).mean()#10次交叉验证取score平均值
# print(score)
# 0.8616577081282963 max_feature10
# 0.8651490145607792 max_feature107
# %%

# %%
"""
将多分类转换成了二分类问题，且需计算每一个类别的预测准确率，召回率，
随后计算平均预测准确率macro_P，平均召回率macro_R，最后的评价指标为：macro_F1 = (2macro_P*macro_R)/(macro_P + macro_R)

"""
# %%
# length_train = len(X)
length_test = len(y_rf_test) #1000

TP = 0
FP = 0
TN = 0
FN = 0
item = 0

a = 0
# 创建列表存放每一个类别的预测准确率和召回率
# accuracy_i = []
precision_i = []
recall_i = []
# 循环向列表中写入数据
while a < 6:
    while item < length_test:  #1000
        if a == y_rf_pred[item]:
            if a == y_rf_test.values[item]:
                TP += 1
            else:
                FP += 1
        else:
            if a == y_rf_test.values[item]:
                FN += 1
            else:
                TN += 1
        item += 1
    print("Label为%d的指标：TP:%d,TN:%d,FP:%d,FN:%d" % (a, TP, TN, FP, FN))
    # macro_accuracy.append((TP + TN) / (TP + TN + FP + FN))    # 准确率
    precision_i.append(TP / (TP + FP))  # 精确率
    recall_i.append(TP / (TP + FN))  # 召回率
    print("precision_i:",precision_i)
    print("recall_i:",recall_i)
    a += 1
    TP, FP, TN, FN, item = 0, 0, 0, 0, 0


# %%
# 模型评价指标
sum = 0
sun = 0
for p in precision_i:
    sum += p
macro_P = sum / len(precision_i)
print("macro_P:", macro_P)

for r in recall_i:
    sun += r
macro_R = sun / len(recall_i)
print("macro_R:", macro_R)

print("macro_F1:", (2 * macro_P * macro_R) / (macro_P + macro_R))
print("排行得分：", 100 * (2 * macro_P * macro_R) / (macro_P + macro_R))

# %%
# print('训练集数量:',length_train)
# print('测试集数量:',length_test)
# print("Label1准确率：", accuracy_L1)
# print("Label1精确率：", precision_L1)
# print("Label1召回率：", recall_L1)
# print("Label1-F1-score：",(2*precision_L1*recall_L1)/(precision_L1+recall_L1))

# %%
# 将y_rf_pred和y_rf_test写入到csv文件
import pandas as pd
# use pandas
# write date by using the form of dict
df = pd.DataFrame({'y_rf_prediction': y_rf_pred, 'y_rf_test': y_rf_test})
df.to_csv("./data/y_rf_pred_test.csv", index=False)
# print(1)
# %%模型评估
print("Test set predictions: \n {}".format(y_rf_pred))
print("Test_Set Score: {:.15f}".format(np.mean(y_rf_pred == y_rf_test)))  #np.mean函数输出两个矩阵/数组的相似程度

#%%
#随机森林可视化
# import importlib
# importlib.reload(plt)
from sklearn import tree
# import pydotplus
# target_names = ["0", "1", "2", "3", "4", "5"]
# feature_name = ['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42', 'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49', 'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature81', 'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89', 'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105', 'feature106']
# Estimators = rfc.estimators_
# for index, model in enumerate(Estimators):
#     filename = 'tree_estimators_' + str(index) + '.pdf'
#     dot_data = tree.export_graphviz(model , out_file=None,
#                          feature_names=feature_name,
#                          class_names=target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_pdf(filename)

