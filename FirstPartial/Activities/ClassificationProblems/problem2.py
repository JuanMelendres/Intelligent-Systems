from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Import WINE data set
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)
print(wine.feature_names)

# Plot pairs of variables

plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Set1, edgecolor='k')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()

# 5-fold cross-validation
print("\nSVM - lineal\n")
kf = KFold(n_splits = 5, shuffle = True)
clf = svm.SVC(kernel = 'linear')

acc = 0
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)
    # Predict one new sample
    print("Prediction for a new observation", clf.predict(x_test))

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    print('acc = ', acc_i)

    acc += acc_i 

acc = acc/5
print('ACC = ', acc)

# 5-fold cross-validation
print("\nSVM de base radial\n")
kf = KFold(n_splits = 5, shuffle = True)
clf = svm.SVC(kernel = 'rbf')

acc = 0
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)
    # Predict one new sample
    print("Prediction for a new observation", clf.predict(x_test))

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    print('acc = ', acc_i)

    acc += acc_i 

acc = acc/5
print('ACC = ', acc)

# 5-fold cross-validation
print("\nk-NN (para k = 3)\n")
kf = KFold(n_splits = 5, shuffle = True)
clf = KNeighborsClassifier(n_neighbors=3)

acc = 0
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)
    # Predict one new sample
    print("Prediction for a new observation", clf.predict(x_test))

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    print('acc = ', acc_i)

    acc += acc_i 

acc = acc/5
print('ACC = ', acc)

# 5-fold cross-validation
print("\nÁrbol de decisión\n")
kf = KFold(n_splits = 5, shuffle = True)
clf = DecisionTreeClassifier(random_state=0)

acc = 0
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)
    # Predict one new sample
    print("Prediction for a new observation", clf.predict(x_test))

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    print('acc = ', acc_i)

    acc += acc_i 

acc = acc/5
print('ACC = ', acc)