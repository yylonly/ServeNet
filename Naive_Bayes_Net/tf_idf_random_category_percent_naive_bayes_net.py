import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import read_hdf, concat
import os
from sklearn.metrics import f1_score, accuracy_score
from time import time

from Utils.utils import type2idx

from sklearn.naive_bayes import MultinomialNB

"""
    Naive bayes network for services classification.
"""
# Load data
basic_path = 'D:\python_projects\ServeNet_others\data\\ramdom_categorg_percent'
data_files = None
if os.path.exists(basic_path):
    data_files = os.listdir(basic_path)

if data_files is None:
    print('None data files in ', basic_path)
    exit()

top_5_train_scores = []
top_5_test_scores = []

top_1_train_scores = []
top_1_test_scores = []

avg_top_5_train_acc = avg_top_5_test_acc = 0.0
avg_top_1_train_acc = avg_top_1_test_acc = 0.0

for fl in data_files:
    url = os.path.join(basic_path, fl)
    if not os.path.exists(url):
        print(url, ' not found!')
        continue
    print('Processing file: ', fl)
    TrainServices = read_hdf(url, key='Train')
    TestServices = read_hdf(url, key='Test')
    AllData = concat([TrainServices, TestServices])

    data_train = list(TrainServices['Service Desciption'])
    target_train = list(TrainServices['Service Classification'])
    data_test = list(TestServices['Service Desciption'])
    target_test = list(TestServices['Service Classification'])

    X_train = data_train
    Y_train = target_train
    X_test = data_test
    Y_test = target_test

    Type_c = (list(np.unique(target_train)))

    encoder = preprocessing.LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.fit_transform(Y_test)

    max_features = 1500

    tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=max_features)
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

    # test top-5 accuracy
    for i in range(len(Y_test)):
        Top5_test = sorted(zip(bayes_net.classes_, test_pre_top5[i]), key=lambda x: x[1])[-5:]
        Top5_test = list(map(lambda x: x[0], Top5_test))

        if Y_test[i] in Top5_test:
            ret[i] = Y_test[i]
        else:
            ret[i] = Top5_test[-1]
    # train top-5 accuracy
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

    avg_top_5_train_acc += accuracy_score(Y_train, train_ret)
    avg_top_5_test_acc += accuracy_score(Y_test, ret)

    avg_top_1_train_acc += accuracy_score(Y_train, train_top1)
    avg_top_1_test_acc += accuracy_score(Y_test, test_pre_top1)

    print("=" * 60)
    print("Test top5 acc:%f,  Train top5 acc:%f" % (accuracy_score(Y_test, ret), accuracy_score(Y_train, train_ret)))
    print("Test top1 acc:%f,  Train top1 acc:%f" % (accuracy_score(Y_test, test_pre_top1),
                                                   accuracy_score(Y_train, train_top1)))
    print("F1_score:%f" % float(f1_s))
    print("=" * 60)
    ####################################################################
    # calculate accuracy of each category.
    # type_c_index = type2idx(Type_c, Type_c)
    #
    # result_dict = {}
    # total_dict = {}
    # for idx in type_c_index:
    #     category = Type_c[idx]
    #     total_count = 0
    #     account = 0
    #     for i in range(len(Y_test)):
    #         if Y_test[i] == idx:
    #             total_count += 1
    #             if Y_test[i] == ret[i]:
    #                 account += 1
    #
    #     result_dict[category] = account / total_count * 1.
    #     total_dict[category] = total_count
    #
    # for cate in result_dict.keys():
    #     total_account = total_dict[cate]
    #     acc = result_dict[cate]
    #     print("%s (%d): %.3f" % (cate, total_account, acc))

avg_top_5_train_acc /= 10.
avg_top_5_test_acc /= 10.
avg_top_1_train_acc /= 10.
avg_top_1_test_acc /= 10.

print("Test top5 acc:%f,  Train top5 acc:%f" % (avg_top_5_test_acc, avg_top_5_train_acc))
print("Test top1 acc:%f,  Train top1 acc:%f" % (avg_top_1_test_acc, avg_top_1_train_acc))

# mean
top_5_train_mean = np.mean(top_5_train_scores)
top_5_test_mean = np.mean(top_5_test_scores)
top_1_train_mean = np.mean(top_1_train_scores)
top_1_test_mean = np.mean(top_1_test_scores)


print("Mean top-5 train: %0.3f top-5 test: %0.3f top-1 train: %.3f top-1 test: %.3f" % (top_5_train_mean, \
    top_5_test_mean, top_1_train_mean, top_1_test_mean))

# variance
top_5_train_std = np.var(top_5_train_scores)
top_5_test_std = np.var(top_5_test_scores)
top_1_train_std = np.var(top_1_train_scores)
top_1_test_std = np.var(top_1_test_scores)

print("Std top-5 train: %0.6f top-5 test: %0.6f top-1 train: %.6f top-1 test: %.6f" % (top_5_train_std, \
                                                                                       top_5_test_std, top_1_train_std, top_1_test_std))



