my_list =list(range(0,108))
print(my_list)

for num in range(107):
    print("'"+'feature'+ str(num)+"'",end=",")

# t.feature0, t.feature1
#%%
for i in range(107):
    print("t." + "feature" +str(i),end=", ")

#%%
#{0}', '{1}', '{2}
for j in range(107):
    print("'{"+str(j)+"}'",end=", ")

#%%
for k in range(107):
    print("feature"+str(k) +"= db.Column(db.Text)")

#%%
for x in range(107):
    print("'feature"+str(x)+"'",end=", ")

#%%
for m in range(107):
    print("self.feature"+str(m) +"= feature"+str(m))

#%%
for s in range(107):
    print("i."+"feature"+str(s),end=", ")

#%%
for p in range(107):
    print("'feature"+str(p)+"'",end=", ")

#%%
length_test = 200

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
