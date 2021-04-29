# Directory imports
import os

# Data manipulation imports 
import pandas as pd
import numpy as np

# Data Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/Sam/Desktop\Machine Learning/11. Deep Learning/TensorFlow_FILES/DATA')

# housing data assoicated with King County, Washington
df = pd.read_csv('kc_house_data.csv')

# checking to see of there are null values
df.isnull().sum()

df.describe()

#######################

# Graphs and Exploratory Data Analysis

plt.figure(figsize=(10,6))
sns.distplot(df['price'])
plt.show()

sns.countplot(df['bedrooms'])
plt.show()

df.corr()['price'].sort_values()

plt.figure(figsize=(10,6))
sns.scatterplot(x='price',y='sqft_living',data=df)
plt.show()

sns.boxplot(x='bedrooms',y='price',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='price',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='lat',y='price',data=df)
plt.show()

# plotting out latitude and longitude of houses with shading based on price - the resulting 
#   graph will produce a figure similar to the geography of western Washington
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')
plt.show()


df.sort_values('price',ascending=False).head(20)

# getting last index of top 1%
len(df)*.01

# excluding top 1% from new df
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=non_top_1_perc,edgecolor=None,alpha=0.2,palette='RdYlGn',hue='price')
plt.show()

sns.boxplot(x='waterfront',y='price',data=df)
plt.show()

####################

# Feature Engineering

# dropping id column
df = df.drop('id',axis=1)

#converting date column values to datetime objects
df['date'] = pd.to_datetime(df['date'])

# extracting month and year values and making them into their own column
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

df.head()

sns.boxplot(x='month',y='price',data=df)
plt.show()

# there doesn't seem to be any correlation between price and date so we will drop the column
df = df.drop('date',axis=1)

df.head()

df['zipcode'].value_counts()

df = df.drop('zipcode',axis=1)

df['yr_renovated'].value_counts()

df['sqft_basement'].value_counts()

#########################

# Creating the Deep Learning Model

X = df.drop('price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test), batch_size=128,epochs=400)

# Exploring the losses from each epoch
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score

predictions = model.predict(X_test)
np.sqrt(mean_squared_error(y_test,predictions))

mean_absolute_error(y_test,predictions)

# the mean absolute error is around 100,000; the mean housing price is about 500,000 - so our model
#   was not very accurate at predicting housing prices


single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
model.predict(single_house)

df.head(1)