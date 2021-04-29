# UPDATE to Scikit-Learn code:

#   from sklearn.cross_validation import train_test_split

#       has been changed to :

#   from sklearn.model_selection import train_test_split

###################################

# Part 1: Training Data

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/Sam/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression')

df = pd.read_csv('USA_Housing.csv')

df.head()
df.info()
df.describe()
df.columns

sns.pairplot(df)
plt.show()

sns.distplot(df['Price'])
plt.show()


sns.heatmap(df.corr(),annot=True)
plt.show()


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)

lm.coef_

X_train.columns

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf

###################################

# Part 2: Predictions and Test Data

predictions = lm.predict(X_test)
predictions
# ^^ predicted prices for each house

plt.scatter(y_test,predictions)
# ^^ can compare actual test data with the predictions to determine if the predicitons were acurate

sns.distplot((y_test-predictions))
# ^^ a normal distribution in your plot is good!

from sklearn import metrics

# The metrics below are what we're concerned about

metrics.mean_absolute_error(y_test,predictions)
# ^^ mean absolute error

metrics.mean_squared_error(y_test,predictions)
# ^^ mean squared error - punishes variation more that absolute error

np.sqrt(metrics.mean_squared_error(y_test,predictions))
# ^^ root mean square error - just take the square root of the mean square error

plt.show()
