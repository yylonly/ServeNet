import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pandas import read_hdf, concat, DataFrame
from sklearn.utils import Bunch
from sklearn.metrics import f1_score,accuracy_score, mean_squared_error
from time import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.naive_bayes import MultinomialNB

"""
    Naive bayes network for services classification using random dataset selecting method.
"""


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
TrainServices = read_hdf('D:\python_projects\ServeNet\RandomSplittedByCatagories.h5', key='Train')
TestServices = read_hdf('D:\python_projects\ServeNet\RandomSplittedByCatagories.h5', key='Test')

train_len = len(TrainServices)
test_len = len(TestServices)

# Merge training and testing data
allData = concat([TrainServices, TestServices])

# kf
kf = KFold(n_splits=10, shuffle=False)

TrainServices = DataFrame()
TestServices = DataFrame()

top_5_train_scores = []
top_5_test_scores = []

top_1_train_scores = []
top_1_test_scores = []

avg_top_5_train_acc_socre = avg_top_5_test_acc_socre = 0.0
avg_top_1_train_acc_score = avg_top_1_test_acc_score = 0.0
avg_f_1_score = 0.0
index = 0
for train_index, test_index in kf.split(allData):
    # create new training and testing services
    TrainServices, TestServices = allData.iloc[train_index], allData.iloc[test_index]

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

    Y_train=type2idx(Y_train,Type_c)
    Y_test=type2idx(Y_test,Type_c)

    max_features = 2000
    n_topics = 275
    max_iter = 100

    tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=max_features)
    X_train = tfidf_vectorizer.fit_transform(X_train)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    X_test = tfidf_vectorizer.transform(X_test)

    # Train processing
    bayes_net = MultinomialNB()

    t0 = time()
    bayes_net.fit(X_train, Y_train)
    t1 = time()
    print("Train time: ", t1 - t0)

    train_top5 = bayes_net.predict_proba(X_train)
    train_top1 = bayes_net.predict(X_train)

    test_pre_top5 = bayes_net.predict_proba(X_test)
    test_pre_top1 = bayes_net.predict(X_test)

    ret = np.empty((len(Y_test),), dtype=np.int)
    train_ret = np.empty((len(Y_train),), dtype=np.int)
    for i in range(len(Y_test)):
        Top5 = sorted(zip(bayes_net.classes_, test_pre_top5[i]), key=lambda x: x[1])[-5:]
        Top5=list(map(lambda x: x[0], Top5))

        if Y_test[i] in Top5:
            ret[i] = Y_test[i]
        else:
            ret[i] = Top5[-1]

    for i in range(len(Y_train)):
        Top5_train = sorted(zip(bayes_net.classes_, train_top5[i]), key=lambda x: x[1])[-5:]
        Top5_train = list(map(lambda x: x[0], Top5_train))

        if Y_train[i] in Top5_train:
            train_ret[i] = Y_train[i]
        else:
            train_ret[i] = Top5_train[-1]

    f1_s = f1_score(Y_test, ret, average='micro')

    top_5_train_scores.append(accuracy_score(Y_train, train_ret))
    top_5_test_scores.append(accuracy_score(Y_test, ret))
    top_1_train_scores.append(accuracy_score(Y_train, train_top1))
    top_1_test_scores.append(accuracy_score(Y_test, test_pre_top1))

    avg_top_5_train_acc_socre += accuracy_score(Y_train, train_ret)
    avg_top_5_test_acc_socre += accuracy_score(Y_test, ret)
    avg_top_1_train_acc_score += accuracy_score(Y_train, train_top1)
    avg_top_1_test_acc_score += accuracy_score(Y_test, test_pre_top1)
    avg_f_1_score += float(f1_s)
    # print("Test top5 acc:%f,train top5  acc:%f" % (accuracy_score(Y_test, ret), accuracy_score(Y_train, train_ret)))
    # print("Test top1 acc:%f,train top1 acc:%f" % (accuracy_score(Y_test, test_pre_top1), accuracy_score(Y_train, train_top1)))
    # print("F1_score:%f" % float(f1_s))
    index += 1


avg_top_5_train_acc_socre /= index
avg_top_5_test_acc_socre /= index
avg_top_1_train_acc_score /= index
avg_top_1_test_acc_score /=index
avg_f_1_score /= index

print("Test top5 acc:%f,train top5  acc:%f" % (avg_top_5_test_acc_socre, avg_top_5_train_acc_socre))
print("Test top1 acc:%f,train top1 acc:%f" % (avg_top_1_test_acc_score, avg_top_1_train_acc_score))
print("F1_score:%f" % avg_f_1_score)

# mean
top_5_train_mean = np.mean(top_5_train_scores)
top_5_test_mean = np.mean(top_5_test_scores)
top_1_train_mean = np.mean(top_1_train_scores)
top_1_test_mean = np.mean(top_1_test_scores)


print("Mean top-5 train: %0.3f top-5 test: %0.3f top-1 train: %.3f top-1 test: %.3f" % (top_5_train_mean, \
    top_5_test_mean, top_1_train_mean, top_1_test_mean))

# variance
top_5_train_var = np.var(top_5_train_scores)
top_5_test_var = np.var(top_5_test_scores)
top_1_train_var = np.var(top_1_train_scores)
top_1_test_var = np.var(top_1_test_scores)

print("Variance top-5 train: %0.6f top-5 test: %0.6f top-1 train: %.6f top-1 test: %.6f" % (top_5_train_var, \
    top_5_test_var, top_1_train_var, top_1_test_var))
