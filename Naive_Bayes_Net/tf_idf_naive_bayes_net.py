import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pandas import read_hdf
from sklearn.utils import Bunch
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error
from time import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.naive_bayes import MultinomialNB

"""
    Naive bayes network for services classification.
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

print("Model LDA...")
# lda = LatentDirichletAllocation(n_topics=n_topics,
#                                         max_iter=max_iter,
#                                         learning_method='batch', random_state=0, n_jobs=1, evaluate_every=10,#doc_topic_prior=0.01,topic_word_prior=0.01,
#                                         verbose=1)

# X_train = lda.fit_transform(X_train)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
X_test = tfidf_vectorizer.transform(X_test)
# X_test = lda.transform(X_test)

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

print("Test top5 acc:%f,train top5  acc:%f" % (accuracy_score(Y_test, ret), accuracy_score(Y_train, train_ret)))
print("Test top1 acc:%f,train top1 acc:%f" % (
accuracy_score(Y_test, test_pre_top1), accuracy_score(Y_train, train_top1)))
print("F1_score:%f" % float(f1_s))