#####################################################

## Support Vector Machines Notes ##

    # supervised learning models used for classification and regression analysis
    # in terms of classification, SVM will assign each data point to one of two categories
        # making it a non-probabilistic binary linear classifier
    # separate categories are defined by a clear gap that is as wide as possible

#####################################################

# Directory imports
import os

# Data manipulation imports 
import pandas as pd
import numpy as np

# Data Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_breast_cancer


os.chdir('C:/Users/Sam/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/16-Support-Vector-Machines')

cancer = load_breast_cancer()
cancer.keys()

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.head()

cancer['target_names']

X = df_feat
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# SVM import
from sklearn.svm import SVC

model = SVC()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,.01,.001,.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(X_train,y_train)

grid.best_params_
grid.best_estimator_

pred_2 = grid.predict(X_test)

print(classification_report(y_test,pred_2))
print(confusion_matrix(y_test,pred_2))


