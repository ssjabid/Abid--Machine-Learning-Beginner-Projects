# Start by importing packages - LET YOUR NUTS HANG

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pylab as pl

# GET THE DATA

fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()

print(fruits.head())

# we have 59 pieces of fruit in this data set and 7 features

print(fruits.shape)
print(fruits['fruit_name'].unique()) # there are 4 unique fruits

print(fruits.groupby('fruit_name').size()) # there are not many mandarins, other than that its quite balanced

# lets create some charts

sns.countplot(fruits['fruit_name'], label = 'count')
plt.show()
# pretty simple

# lets do some serious VISUALIZATION
# lets make some box plots for all 4 numerical variables

fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey = False,
            figsize=(9, 9), title = 'Box Plot for each numerical variable')

plt.savefig('fruits_box')
plt.show();

#it looks like color has a gaussian distribution why?
#lets make some histograms to address each of these box plots further

fruits.drop('fruit_label', axis=1).hist(bins=30, figsize=(9,9))
plt.suptitle('Histogram for each numerical variable')
plt.savefig('fruits_hist')
plt.show()

# Some pairs of attributes are correlated (mass and width). This suggests a high correlation
# and a predictable relationship.


# lets plot some scatter graphs!

feature_names = ['mass', 'width', 'height', 'color_score']
x = fruits[feature_names]
y = fruits['fruit_label']

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(x, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix')
plt.show()

# lets print a statistical summary

print(fruits.describe())

# we can see that the numerical values do not have the same scale, we will need to apply scaling to the test
# set that we computed for the training set

# CREATE TRAINING AND TEST SETS

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# BUILD MODELS - LOGISTIC REGRESSION

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print('Accuracy of Logistic Regression classifier on training set:{:.2f}'.format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic Regression classifier on training set:{:.2f}'.format(logreg.score(x_test, y_test)))

# training set = 0.7, test set = 0.4

clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(x_test, y_test)))

# training set = 1, test set = 0.73


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test)))

# training set = 1, test set = 0.95

# plot the decision boundary of the kNN classifier

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def plot_fruit_knn(X, y, n_neighbors, weights):
    X_mat = X[['height', 'width']].as_matrix()
    y_mat = y.as_matrix()

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X_mat, y_mat)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])

    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

    plt.show()

plot_fruit_knn(X_train, y_train, 5, 'uniform')

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])

# for this particular set we obtain the highest accuracy when k = 5