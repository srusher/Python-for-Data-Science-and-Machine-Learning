##################################################################

# Logistic regression attempts to predict a classification of data in categories as opposed to predicted a value 
# the Sigmoid model is used for Logistic Regression

##################################################################

## PART 1: Exploratory Data Analysis

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/Sam/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression')

train = pd.read_csv('titanic_train.csv')
train.head()

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train, hue='Sex',palette='RdBu_r')
plt.show()

sns.displot(train['Age'].dropna(),kde=False,bins=30)
plt.show()

sns.countplot(x='SibSp',data=train)
plt.show()

train['Fare'].hist(bins=40,figsize=(10,4))
plt.show()

##################################################################

## PART 2: Data Cleaning

sns.boxplot(x='Pclass',y='Age',data=train)
plt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

train.drop('Cabin',axis=1,inplace=True)
train.head()

train.dropna(inplace=True)

# ML algorithms don't know how to deal with strings, so you need to convert strings in numerical categories

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()
train = pd.concat([train,sex,embark],axis=1)
train.head(2)

# Drop columns that your dummy variables originated from
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
train.head()

##################################################################

## PART 3

X = train.drop('Survived',axis=1)
y = train['Survived']

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101,)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)