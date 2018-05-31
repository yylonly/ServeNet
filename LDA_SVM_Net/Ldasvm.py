import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from pandas import  read_hdf
from sklearn.utils import Bunch
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error
from time import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def type2idx(Data_c,Type_c):
    n_samples=len(Data_c)
    target = np.empty((n_samples,), dtype=np.int)
    for idx in range(n_samples):
        if Data_c[idx] in Type_c:
            target[idx]=Type_c.index(Data_c[idx])
        else:
            target[idx] = -1
    return target
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

TrainServices = read_hdf('./data/RandomSplittedByCatagories.h5', key='Train')
TestServices = read_hdf('./data/RandomSplittedByCatagories.h5', key='Test')
# All_data=read_hdf('./data/RandomSplittedByCatagories.h5',key='AllData')

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
# n_components = 10
n_top_words = 20
# data_samples = data_train.data[:n_samples]
Type_c=(list(np.unique(target_train)))
#
# offset = int(len(X) * 0.8)
# X_train, Y_train = X[:offset], Y[:offset]
# X_test, Y_test = X[offset:], Y[offset:]


print("Service Description: \n" ,X_train[0])
print("Service Classification:",Y_train[0][0])
print(len(X_train))
print(len(X_test))
Y_train=type2idx(Y_train,Type_c)
Y_test=type2idx(Y_test,Type_c)
print("Extracting tf-idf features for LDA...")

max_features=2000
n_topics=275
max_iter=60
kernel='rbf'
save_partName=kernel+'_'+str(max_features)+'_'+str(n_topics)+'_'+str(max_iter)

tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=max_features)
# tf_train = tfidf_vectorizer.fit_transform(data_train.data)
X_train = tfidf_vectorizer.fit_transform(X_train)
# print(X_train[0])

print("Model LDA...")
lda = LatentDirichletAllocation(n_topics=n_topics,
                                        max_iter=max_iter,
                                        learning_method='batch', random_state=0, n_jobs=1, evaluate_every=10,#doc_topic_prior=0.01,topic_word_prior=0.01,
                                        verbose=1)

X_train=lda.fit_transform(X_train)

#
# joblib.dump(lda, 'lda275_e.model')
# print(X_train[0])
# 打印最佳模型
joblib.dump(lda, 'lda_'+save_partName+'.model')
# print("\nTopics in LDA model:")
print("same tfidf_vectorizer")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

X_test = tfidf_vectorizer.transform(X_test)
X_test=lda.transform(X_test)

# train_target=y_train
# test_target=y_test
max_iter=range(1,1000,10)
F1_score=[]
train_errors1 = list()
test_errors1 = list()
train_errorstop5 = list()
test_errorstop5 = list()

train_acctop1=list()
test_acctop1=list()
train_acctop5=list()
test_acctop5=list()
# model=svm_cross_validation(ldax,train_target)
# print("Model Done")
# joblib.dump(model, 'svm.md')
# model = joblib.load('svm.md')
n_samples=X_train.shape[0]

for idx, iter in enumerate(max_iter):
    clf=SVC(gamma=1,probability=True,decision_function_shape='ovr',kernel=kernel,max_iter=iter,tol=1e-5)
    t0 = time()
    clf.fit(X_train,Y_train)
    t1 = time()
    print("Time:",t1-t0)
    train_top5=clf.predict_proba(X_train)
    train_top1=clf.predict(X_train)

    train_acctop1.append((iter,accuracy_score(Y_train, train_top1)))
    train_errors1.append((iter, mean_squared_error(Y_train, train_top1)))

    # test_target=type2idx(Test_Y,Type_c)
    test_pre_top5 = clf.predict_proba(X_test)
    test_pre_top1 = clf.predict(X_test)
    test_acctop1.append((iter,accuracy_score(Y_test, test_pre_top1)))
    test_errors1.append((iter, mean_squared_error(Y_test, test_pre_top1)))


    ret=np.empty((len(Y_test),), dtype=np.int)
    train_ret=np.empty((len(Y_train),), dtype=np.int)
    for  i in range(len(Y_test)):
        Top5 = sorted(zip(clf.classes_, test_pre_top5[i]), key=lambda x: x[1])[-5:]
        Top5=list(map(lambda x: x[0], Top5))

        if Y_test[i] in Top5:
            ret[i]=Y_test[i]
        else:
            ret[i]=Top5[-1]
        # print("True-Type=%d and pre=%d..."
        #       % (test_target[i], ret[i]))
    for i in range(len(Y_train)):
        Top5_train = sorted(zip(clf.classes_, train_top5[i]), key=lambda x: x[1])[-5:]
        Top5_train = list(map(lambda x: x[0], Top5_train))
        if Y_train[i] in Top5_train:
            train_ret[i] = Y_train[i]
        else:
            train_ret[i] = Top5_train[-1]

    f1_s=f1_score(Y_test, ret, average='micro')
    train_acctop5.append((iter, accuracy_score(Y_train, train_ret)))
    test_acctop5.append((iter, accuracy_score(Y_test, ret)))
    train_errorstop5.append((iter, mean_squared_error(Y_train, train_ret)))
    test_errorstop5.append((iter, mean_squared_error(Y_test, ret)))
    print("-"*80)
    print("Test top5 acc:%f,train top5  acc:%f"%(accuracy_score(Y_test, ret),accuracy_score(Y_train, train_ret)))
    print("Test top1 acc:%f,train top1 acc:%f"%(accuracy_score(Y_test, test_pre_top1),accuracy_score(Y_train, train_top1)))
    print("Epoch:%d,F1_score:%f"%(iter,float(f1_s)))
    F1_score.append((iter,f1_s))

joblib.dump(F1_score, 'F1_score'+save_partName+'.dat')
joblib.dump(train_errors1, 'train_errors1'+save_partName+'.dat')
joblib.dump(test_errors1, 'test_errors1'+save_partName+'.dat')
joblib.dump(train_acctop1, 'train_acctop1'+save_partName+'.dat')
joblib.dump(test_acctop1, 'test_acctop1'+save_partName+'.dat')
joblib.dump(train_acctop5, 'train_acctop5'+save_partName+'.dat')
joblib.dump(test_acctop5, 'test_acctop5'+save_partName+'.dat')
joblib.dump(test_errorstop5, 'test_errorstop5'+save_partName+'.dat')
joblib.dump(train_errorstop5, 'train_errorstop5'+save_partName+'.dat')
# plt.plot(*zip(*F1_score))
