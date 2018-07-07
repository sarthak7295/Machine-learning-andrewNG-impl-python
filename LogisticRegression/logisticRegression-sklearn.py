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
dataset = pd.read_csv("data/ex2data2.csv", names=names)
X = dataset.drop('Y', axis=1)
Y = dataset.Y
# print(dataset)
category_one = dataset[dataset["Y"] == 1]
category_zero = dataset[dataset["Y"] == 0]

plt.scatter(category_one.X1, category_one.X2, marker='x', c='g')
plt.scatter(category_zero.X1, category_zero.X2, marker='x', c='r')
mylogisticModel = LogisticRegression()

# adding more features since it was not working fine
X2 = X**2
X2.columns = ['XX1', 'XX2']
feature_rich_x = pd.concat([X, X2], axis=1)
print(feature_rich_x)
mylogisticModel.fit(feature_rich_x[["X1", "X2", "XX1", "XX2"]], dataset["Y"])
# print(mylogisticModel.coef_)
# mylogisticModel.intercept_
data = [[0, 0, 0, 0]]
a = pd.DataFrame(data, columns=['X1', 'X2','XX1','XX2'])
print(mylogisticModel.predict(feature_rich_x))
plt.plot()

plt.show()