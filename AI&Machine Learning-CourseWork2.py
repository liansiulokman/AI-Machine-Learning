import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# Read in data: df


# Answer to Question [7], Part [2a]
######### Data Preprocessing #########

#Checked if there are NA values in the data
df.info()
df.head()

df.shape
print('\nFeatures:', df.columns)
df_na = (df.isna().sum()/df.shape[0])*100
print('\nNumber of NAs:\n')
print(df_na)
#Result: no NA values


#Turn target variables into 1/0 for classification prediction, 1 = CH, 0 = MM
df["Purchase_CH"] = df["Purchase"].apply(lambda target: (target=='CH') * 1)

#Turn Store7 into 1/0(numeric) instead of string(Yes/No), 1 = Yes, 0 = No
df["Store7"] = df["Store7"].apply(lambda store7: (store7=='Yes') * 1)

#Turn StoreID into separate dummies as the numeric value in StoreID variable has no numerical meaning to the model
df['StoreID_1'] = df.StoreID.apply(lambda x: (x==1) * 1)
df['StoreID_2'] = df.StoreID.apply(lambda x: (x==2) * 1)
df['StoreID_3'] = df.StoreID.apply(lambda x: (x==3) * 1)
df['StoreID_4'] = df.StoreID.apply(lambda x: (x==4) * 1)
df['StoreID_7'] = df.StoreID.apply(lambda x: (x==7) * 1)

#Turn STORE into separate dummies as the numeric value in STORE variable has no numerical meaning to the model
df['STORE_0'] = df.STORE.apply(lambda x: (x==0) * 1)
df['STORE_1'] = df.STORE.apply(lambda x: (x==1) * 1)
df['STORE_2'] = df.STORE.apply(lambda x: (x==2) * 1)
df['STORE_3'] = df.STORE.apply(lambda x: (x==3) * 1)
df['STORE_4'] = df.STORE.apply(lambda x: (x==4) * 1)

#Drop the unique variable ID and the columns that already used to derive another new variables
df.drop(columns=['ID','Purchase','StoreID','STORE'], inplace = True)

print('\nFeatures in dataset:', df.shape[1])
print('Number of instances:', df.shape[0])
print('\nFeatures:', df.columns)


#Split data into training and testing datasets
#Split the data into X = all the featurse, and y = the target variable
X = df.loc[:, df.columns != 'Purchase_CH'].to_numpy()
y = df['Purchase_CH'].to_numpy()

#Check if dataset is imbalanced
total = df.shape[0]
ones = df['Purchase_CH'].sum()
zeroes = total - ones
print('\nTotal number of instances: ', total)
print('Total number of 1 (CH juice): ', ones, '({:.2f}%)'.format(ones/total*100))
print('Total number of 0 (MM juice): ', zeroes, '({:.2f}%)'.format(zeroes/total*100))
#61% vs 39%, not imbalanced dataset

#normalize the data using min max scaler
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# Answer to Question [7], Part [2b]
#split training and testing using 80%/20% ratio and set the random_state to reproduce same result
#test data is used for the model evaluation for each model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

#Check splited datasets' sizes
X_train.shape
X_test.shape
y_train.shape
y_test.shape
#training data: 856 instances
#testing data: 214 instances


# Answer to Question [7], Part [2c]

######### Model 1 - Decision Tree #########
print("\n########### Model 1 - Decision Tree ###########")
clf = DecisionTreeClassifier(random_state=100, criterion="entropy",
                             min_samples_split=2, min_samples_leaf=2, 
                             max_depth=30, max_leaf_nodes=50, min_weight_fraction_leaf=.1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#print confusion matrix stats
print("\nModel 1 - Decision Tree (Before Tuning):")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tp: ", tp, " tn: ", tn, " fp: ", fp, " fn: ", fn)

#print performance
print("AUC Score = {:.2f}".format(roc_auc_score(y_test, y_pred)))


#Model 1 - Tuning (Use Grid Search to find the best hyparameters for the decision tree model, 
# use auc score as the scoring parameter)
base_clf = DecisionTreeClassifier(splitter="best", class_weight=None, random_state=100)
parameters_grid = {"criterion": ("gini", "entropy"),
                  "max_depth": [2, 10, 30, 50, 100, 200],
                  "max_leaf_nodes": [None, 5, 10, 50, 100],
                  "min_impurity_decrease": [0, 0.1, 0.2],
                  "min_samples_leaf": [1, 2, 10, 50],
                  "min_samples_split": [2, 10, 50],
                  "min_weight_fraction_leaf": [.1 , .2]}
clf_gs = GridSearchCV(base_clf, param_grid=parameters_grid, scoring="roc_auc", n_jobs = 3, cv = 5, return_train_score = True)
clf_gs.fit(X_train, y_train)


#Model 1 - Print the best hyperparameter values for decision tree model
print('\nBest hyperparameter values: ', clf_gs.best_params_)
#{'criterion': 'entropy', 'max_depth': 10, 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_samples_leaf': 1,
#  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.1}

#Model 1 - Print performance for the fine-tuned model
clf_best = DecisionTreeClassifier(random_state=100, criterion="entropy",
                             min_samples_split=2, min_samples_leaf=1, 
                             max_depth=10, max_leaf_nodes=None, min_impurity_decrease=0, min_weight_fraction_leaf=0.1)

clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)

print("\nModel 1 - Decision Tree (Tuned):")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
print("tp: ", tp, " tn: ", tn, " fp: ", fp, " fn: ", fn)
print("AUC Score = {:.2f}".format(roc_auc_score(y_test, y_pred_best)))
#0.83




######### Model 2 - KNN #########
print("\n########### Model 2 - KNN ###########")
knn_clf = KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(X_train, y_train)
knn_y_pred = knn_clf.predict(X_test)

#print confusion matrix stats
print("\nModel 2 - KNN (Before Tuning):")
tn, fp, fn, tp = confusion_matrix(y_test, knn_y_pred).ravel()
print("tp: ", tp, " tn: ", tn, " fp: ", fp, " fn: ", fn)

#print performance
print("AUC Score = {:.2f}".format(roc_auc_score(y_test, knn_y_pred)))

#Model 2 - Tuning (Use Grid Search to find the best hyparameters for the KNN model, 
# use auc score as the scoring parameter)
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

knn_gs = KNeighborsClassifier()
knn_clf_gs = GridSearchCV(knn_gs, hyperparameters, cv=10, scoring="roc_auc")
knn_clf_gs.fit(X_train,y_train)

#Model 2 - Print the best hyperparameter values for KNN model
print('\nBest hyperparameter values: ', knn_clf_gs.best_params_)
#auc: n_neighbors: 22, leaf_size: 1, p: 2

#Model 2 - Print performance for the fine-tuned model
knn_clf_best = KNeighborsClassifier(n_neighbors=22, leaf_size=1, p=2)

knn_clf_best.fit(X_train, y_train)
knn_y_pred_best = knn_clf_best.predict(X_test)
print("\nModel 2 - KNN (Tuned):")
tn, fp, fn, tp = confusion_matrix(y_test, knn_y_pred_best).ravel()
print("tp: ", tp, " tn: ", tn, " fp: ", fp, " fn: ", fn)
print("AUC Score = {:.2f}".format(roc_auc_score(y_test, knn_y_pred_best)))
#0.78



######### Model 3 - SVM #########
print("\n########### Model 3 - SVM ###########")
svm_clf = SVC(random_state=100)
svm_clf.fit(X_train, y_train)

svm_y_pred = svm_clf.predict(X_test)

#print confusion matrix stats
print("\nModel 3 - SVM (Before Tuning):")
tn, fp, fn, tp = confusion_matrix(y_test, svm_y_pred).ravel()
print("tp: ", tp, " tn: ", tn, " fp: ", fp, " fn: ", fn)

#print performance
print("AUC Score = {:.2f}".format(roc_auc_score(y_test, svm_y_pred)))


#Model 3 - Tuning (Use Grid Search to find the best hyparameters for the SVM model, 
# use auc score as the scoring parameter)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm_clf_gs = GridSearchCV(SVC(random_state=100), tuned_parameters, scoring='roc_auc')
svm_clf_gs.fit(X_train,y_train)

#Model 3 - Print the best hyperparameter values for SVM model
print('\nBest hyperparameter values: ', svm_clf_gs.best_params_)
#C:1000, gamma: 0.001, kernel: 'rbf' 

#Model 3 - Print performance for the fine-tuned model
svm_clf_best = SVC(random_state=100, kernel='rbf', gamma=0.001, C=1000)

svm_clf_best.fit(X_train, y_train)
svm_y_pred_best = svm_clf_best.predict(X_test)
print("\nModel 3 - SVM (Tuned):")
tn, fp, fn, tp = confusion_matrix(y_test, svm_y_pred_best).ravel()
print("tp: ", tp, " tn: ", tn, " fp: ", fp, " fn: ", fn)
print("AUC Score = {:.2f}".format(roc_auc_score(y_test, svm_y_pred_best)))
#0.85
