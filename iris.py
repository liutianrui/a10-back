# 导入相关包
import numpy as np
import pandas as pd

#%%
# 导入数据集
data = "./data/iris.csv"

iris_local = pd.read_csv(data, usecols=[0, 1, 2, 3, 4, 5])
iris_local = iris_local.dropna()    # 丢弃含空值的行、列
print(iris_local.head())

#%%
# 查看数据集信息
iris_local.info()
print(123)
print(iris_local.keys())
#%%
# 载入特征和标签集
X = iris_local[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] # 等价于iris_dataset.data
y1 = iris_local['species']     # 等价于iris_dataset.target

#%%
# 对标签集进行编码
#0 代表setosa，1 代表versicolor，2 代表virginica
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y1)
# print(y)
#y现在为012

#%%
#训练集测试集划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#%%
print("X_train shape: {}".format(X_train.shape))        # X_train shape: (112, 4)
print("y_train shape: {}".format(y_train.shape))        # y_train shape: (112,)

print("X_test shape: {}".format(X_test.shape))          # X_test shape: (38, 4)
print("y_test shape: {}".format(y_test.shape))          # y_test shape: (38,)

#%%
# import importlib
# importlib.reload(plt)
# print(iris_local.describe())
# fig = iris_local.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
# fig=plt.gcf()
# fig.set_size_inches(10,6)
# plt.show()

#%%
# 生成数据EDA报告
# import ydata_profiling as yp
# report = yp.ProfileReport(iris_local)
# report.to_file('report.html')

#%%
#使用KNN分类器
#k 近邻算法中 k 的含义是，我们可以考虑训练集中与新数据点最近的任意 k 个邻居（比如说，距离最近的3 个或5 个邻居），
# 而不是只考虑最近的那一个。然后，我们可以用这些邻居中数量最多的类别做出预测。
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("---KNN is READY---")
#%%
#定义一个numpy数组 species_name
species_name = np.array(["setosa","versicolor","virginica"])

#对新输入数据做出预测
X_new = pd.DataFrame([[5, 2.9, 1, 0.2]] , columns=list(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
# 注意：要用pandas中DataFrame类型，这种类型里面包含feature_name，不会有warning；
# 而 X_new = np.array([[5, 2.9, 1, 0.2]])是numpy里的ndarry类型；只有数据,没有feature_name；会有警告，

print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))  # Prediction: [0]

# print(y1)   print(y)
# y1 = iris_local['species']     y = 0,1,2

print("Predicted species name: {}".format(species_name[prediction]))      # Predicted target name: ['setosa']

#%%模型评估
y_pred = knn.predict(X_test)
print("Test set predictions: \n {}".format(y_pred))
print("Test set score: {:.10f}".format(np.mean(y_pred == y_test)))   # Test set score: 0.97

