import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import read_hdf, concat
from sklearn.utils import Bunch
from sklearn.metrics import f1_score, accuracy_score
from time import time

from Utils.utils import type2idx

from sklearn.naive_bayes import MultinomialNB

"""
    Naive bayes network for services classification.
"""
# Load data
TrainServices = read_hdf('D:\python_projects\ServeNet_others\data\\ramdom_categorg_percent\RandomSplittedByCatagories9.h5', key='Train')
TestServices = read_hdf('D:\python_projects\ServeNet_others\data\\ramdom_categorg_percent\RandomSplittedByCatagories9.h5', key='Test')
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

count = 0
for t in target_test:
    print(t)
    if t == 'Tools':
        count += 1
print(count)

encoder = preprocessing.LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.fit_transform(Y_test)

max_features = 660

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

print("=" * 60)
print("Test top5 acc:%.4f,  Train top5 acc:%.4f" % (accuracy_score(Y_test, ret), accuracy_score(Y_train, train_ret)))
print("Test top1 acc:%.4f,  Train top1 acc:%.4f" % (accuracy_score(Y_test, test_pre_top1),
                                               accuracy_score(Y_train, train_top1)))
print("F1_score:%.4f" % float(f1_s))
print("=" * 60)
####################################################################
# calculate accuracy of each category.
type_c_index = type2idx(Type_c, Type_c)

result_dict = {}
total_dict = {}
avg = 0.0
correct_num = 0
print(Y_test.shape)
print(ret.shape)
for idx in type_c_index:
    category = Type_c[idx]
    total_count = 0
    account = 0
    for i in range(len(Y_test)):
        if Y_test[i] == idx:
            total_count += 1
            if Y_test[i] == ret[i]:
                account += 1
                correct_num += 1

    result_dict[category] = account / total_count * 1.
    total_dict[category] = total_count

total_num = 0
for cate in result_dict.keys():
    total_account = total_dict[cate]
    total_num += total_account
    acc = result_dict[cate]
    print("%s (%d): %.4f" % (cate, total_account, acc))

print(total_num)
print(correct_num / total_num)
print(correct_num)




