import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import read_hdf, concat
from sklearn.utils import Bunch
from sklearn.metrics import f1_score, accuracy_score
from time import time
from sklearn.utils import shuffle

from sklearn.naive_bayes import MultinomialNB

"""
    Naive bayes network for services classification using random dataset selecting method.
"""

# Load data
TrainServices = read_hdf('D:\python_projects\ServeNet\RandomSplittedByCatagories.h5', key='Train')
TestServices = read_hdf('D:\python_projects\ServeNet\RandomSplittedByCatagories.h5', key='Test')

train_len = len(TrainServices)

# Merge training and testing data
allData = concat([TrainServices, TestServices])

top_5_train_scores = []
top_5_test_scores = []
top_1_train_scores = []
top_1_test_scores = []
f_1_scores = []

avg_top_5_train_acc_socre = avg_top_5_test_acc_socre = 0.0
avg_top_1_train_acc_score = avg_top_1_test_acc_score = 0.0
avg_f_1_score = 0.0

# random iteration number
iter_num = 10
for idx in range(iter_num):
    print(" %d iteration:" % idx)

    AllData = shuffle(allData, random_state=42)

    # Split training and testing dataset
    TrainServices = allData[: train_len]
    TestServices = allData[train_len:]

    data_train = list(TrainServices['Service Desciption'])
    target_train = list(TrainServices['Service Classification'])
    data_test = list(TestServices['Service Desciption'])
    target_test = list(TestServices['Service Classification'])

    X_train=data_train
    Y_train=target_train
    X_test=data_test
    Y_test=target_test

    encoder = preprocessing.LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.fit_transform(Y_test)

    max_features = 2000

    tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=max_features)
    tfidf_vectorizer.fit(list(AllData['Service Desciption']))

    X_train = tfidf_vectorizer.transform(X_train)
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
        Top5_test = sorted(zip(bayes_net.classes_, test_pre_top5[i]), key=lambda x: x[1])[-5:]
        Top5_test=list(map(lambda x: x[0], Top5_test))

        if Y_test[i] in Top5_test:
            ret[i] = Y_test[i]
        else:
            ret[i] = Top5_test[-1]

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
    f_1_scores.append(float(f1_s))

    avg_top_5_train_acc_socre += accuracy_score(Y_train, train_ret)
    avg_top_5_test_acc_socre += accuracy_score(Y_test, ret)
    avg_top_1_train_acc_score += accuracy_score(Y_train, train_top1)
    avg_top_1_test_acc_score += accuracy_score(Y_test, test_pre_top1)
    avg_f_1_score += float(f1_s)

avg_top_5_train_acc_socre /= iter_num
avg_top_5_test_acc_socre /= iter_num
avg_top_1_train_acc_score /= iter_num
avg_top_1_test_acc_score /= iter_num
avg_f_1_score /= iter_num

print("Test top5 acc:%f,train top5  acc:%f" % (avg_top_5_test_acc_socre, avg_top_5_train_acc_socre))
print("Test top1 acc:%f,train top1 acc:%f" % (avg_top_1_test_acc_score, avg_top_1_train_acc_score))
print("F1_score:%f" % avg_f_1_score)

# mean
top_5_train_mean = np.mean(top_5_train_scores)
top_5_test_mean = np.mean(top_5_test_scores)
top_1_train_mean = np.mean(top_1_train_scores)
top_1_test_mean = np.mean(top_1_test_scores)

print(top_5_train_scores)
print(top_5_test_scores)

print("Mean top-5 train: %0.3f top-5 test: %0.3f top-1 train: %.3f top-1 test: %.3f" % (top_5_train_mean, \
    top_5_test_mean, top_1_train_mean, top_1_test_mean))

# variance
top_5_train_std = np.var(top_5_train_scores)
top_5_test_std = np.var(top_5_test_scores)
top_1_train_std = np.var(top_1_train_scores)
top_1_test_std = np.var(top_1_test_scores)

print("Std top-5 train: %0.6f top-5 test: %0.6f top-1 train: %.6f top-1 test: %.6f" % (top_5_train_std, \
                                                                                       top_5_test_std, top_1_train_std, top_1_test_std))


