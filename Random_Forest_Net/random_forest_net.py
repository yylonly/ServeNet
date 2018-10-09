import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import read_hdf
from sklearn.utils import Bunch
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error
from time import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier


def type2idx(Data_c,Type_c):
    n_samples=len(Data_c)
    target = np.empty((n_samples,), dtype=np.int)
    for idx in range(n_samples):
        if Data_c[idx] in Type_c:
            target[idx]=Type_c.index(Data_c[idx])
        else:
            target[idx] = -1
    return target


# Load data
TrainServices = read_hdf('../RandomSplittedByCatagories.h5', key='Train')
TestServices = read_hdf('../RandomSplittedByCatagories.h5', key='Test')

data_train=list(TrainServices['Service Desciption'])
target_train=list(TrainServices['Service Classification'])
data_test=list(TestServices['Service Desciption'])
target_test=list(TestServices['Service Classification'])

Train_data=Bunch(data=data_train,target=target_train)
Test_data=Bunch(data=data_test,target=target_test)
# X, Y = shuffle(All_data.data, All_data.target, random_state=13)

X_train=data_train
Y_train=target_train
X_test=data_test
Y_test=target_test

n_top_words = 20

Type_c = (list(np.unique(target_train)))

print(Type_c)
print("Service Description: \n" ,X_train[0])
print("Service Classification:",Y_train[0][0])
print(len(X_train))
print(len(X_test))

Y_train=type2idx(Y_train,Type_c)
Y_test=type2idx(Y_test,Type_c)

max_features = 2000
n_topics = 275
max_iter = 100


tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=max_features)
# tf_train = tfidf_vectorizer.fit_transform(data_train.data)
X_train = tfidf_vectorizer.fit_transform(X_train)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
X_test = tfidf_vectorizer.transform(X_test)

# Train processing
clf = RandomForestClassifier(n_estimators=1000, max_depth=40, random_state=0)

t0 = time()
clf.fit(X_train, Y_train)
t1 = time()
print("Train time: ", t1 - t0)


train_top5 = clf.predict_proba(X_train)
train_top1 = clf.predict(X_train)

test_pre_top5 = clf.predict_proba(X_test)
test_pre_top1 = clf.predict(X_test)

ret = np.empty((len(Y_test),), dtype=np.int)
train_ret = np.empty((len(Y_train),), dtype=np.int)
for i in range(len(Y_test)):
    Top5 = sorted(zip(clf.classes_, test_pre_top5[i]), key=lambda x: x[1])[-5:]
    Top5=list(map(lambda x: x[0], Top5))

    if Y_test[i] in Top5:
        ret[i] = Y_test[i]
    else:
        ret[i] = Top5[-1]

for i in range(len(Y_train)):
    Top5_train = sorted(zip(clf.classes_, train_top5[i]), key=lambda x: x[1])[-5:]
    Top5_train = list(map(lambda x: x[0], Top5_train))

    if Y_train[i] in Top5_train:
        train_ret[i] = Y_train[i]
    else:
        train_ret[i] = Top5_train[-1]

f1_s = f1_score(Y_test, ret, average='micro')

print("Test top5 acc:%f,train top5  acc:%f" % (accuracy_score(Y_test, ret), accuracy_score(Y_train, train_ret)))
print("Test top1 acc:%f,train top1 acc:%f" % (
accuracy_score(Y_test, test_pre_top1), accuracy_score(Y_train, train_top1)))
print("F1_score:%f" % float(f1_s))

####################################################################
# calculate accuracy of each category.
type_c_index = type2idx(Type_c, Type_c)

result_dict = {}
total_dict = {}
for idx in type_c_index:
    category = Type_c[idx]
    total_count = 0
    account = 0
    for i in range(len(Y_test)):
        if Y_test[i] == idx:
            total_count += 1
            if Y_test[i] == ret[i]:
                account += 1

    result_dict[category] = account / total_count * 1.
    total_dict[category] = total_count

for cate in result_dict.keys():
    total_account = total_dict[cate]
    acc = result_dict[cate]
    print("%s (%d): %.3f" % (cate, total_account, acc))