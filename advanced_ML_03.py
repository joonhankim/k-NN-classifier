# -*- coding: utf-8 -*-

# DO NOT CHANGE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def create_bootstrap(X,y,ratio):
    # X: input data matrix
    # ratio: sampling ratio
    # return bootstraped dataset (newX,newy)
    ind=np.random.choice(range(len(X)),replace=True,size=int(len(X)*ratio))
    newX=X[ind]
    newy=y[ind]
    return newX,newy

def voting(y):
    # y: 2D matrix with n samples by n_estimators
    # return voting results by majority voting (1D array)
    y=y.astype('int64')
    k=np.argmax(np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 1, y),axis=1)
    
    return k 
    
# bagging
def bagging_cls(X,y,n_estimators,k,ratio):
    # X: input data matrix
    # y: output target
    # n_estimators: the number of classifiers
    # k: the number of nearest neighbors
    # ratio: sampling ratio
    # return list of n k-nn models trained by different boostraped sets
    bagg_list=[]
    for i in range(n_estimators):
        kn_clf=KNeighborsClassifier(n_neighbors=k)
        newX,newy=create_bootstrap(X,y,ratio)
        result=kn_clf.fit(newX,newy)
        bagg_list.append(result)
    return bagg_list
    

data=load_iris()
X=data.data[:,:2]
y=data.target    

n_estimators=3
k=3
ratio=0.8

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = np.c_[xx.ravel(), yy.ravel()]

models = bagging_cls(X,y,n_estimators,k,ratio)
y_models = np.zeros((len(xx.ravel()),n_estimators))
for i in range(n_estimators):
    y_models[:,i]=models[i].predict(Z)

y_pred=voting(y_models)



# Draw decision boundary
plt.contourf(xx,yy,y_pred.reshape(xx.shape),cmap=plt.cm.RdYlBu)
plt.scatter(X[:,0],X[:,1],c='k',s=10)

