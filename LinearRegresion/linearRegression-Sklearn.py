import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Kind of init for 3D-graphs below lines
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Taking data of andrew ng excersise
names = ["X1", "X2", "Y"]
dataset = pd.read_csv("data/ex1data2.csv", names=names)
X = dataset.drop('Y', axis=1)
Y = dataset.Y


# print(Y.max())
# print(Y.mean())
myLinearModel = LinearRegression()
myLinearModel_normalized = LinearRegression()
myLinearModel.fit(dataset[["X1", "X2"]], dataset["Y"])
plt.figure("NON-normalized graph")
plt.scatter(X.X1, Y, marker='^', c='g')
plt.scatter(X.X2, Y, marker='^', c='r')
plt.plot(X.X1, ((myLinearModel.coef_[1]*X.X2) + (myLinearModel.coef_[0]*X.X1) + myLinearModel.intercept_))
plt.xlabel('red : no of bedrooms, Green : area')
plt.ylabel('price of house')

def nomalizecolumn(col):
    col = ((col - col.mean())/(col.max()))*100
    return col


X.X1 = nomalizecolumn(X.X1)
X.X2 = nomalizecolumn(X.X2)
norm_data_set = pd.concat([X, Y], axis=1)
# print(norm_data_set)
myLinearModel_normalized.fit(norm_data_set[["X1", "X2"]], dataset["Y"])
# A = pd.concat([Y, N], axis=1)
# First we normalize the features to get good graph
plt.figure("normalized graph")
plt.scatter(X.X1, Y, marker='^', c='g')
plt.scatter(X.X2, Y, marker='^', c='r')
plt.xlabel('red : no of bedrooms, Green : area')
plt.ylabel('price of house')

# Linear-Regression starts

# First FIT and then PREDICT !!!

# we need to reshape series (in this case Y since its size gives (47,) instead of (47,1) we need both args
# reshape(-1, 1) makes it into proper vector

data = [[2014, 3], [2814, 4]]
a = pd.DataFrame(data, columns=['X1', 'X2'])
# print(myLinearModel.predict(dataset[["X1", "X2"]])-dataset["Y"])

# plt.plot(X.X1, ((myLinearModel.coef_[1]*X.X1) + myLinearModel.intercept_))

plt.show()

