# Directory imports
import os

# Data manipulation imports 
import pandas as pd
import numpy as np

# Data Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/Sam/Desktop\Machine Learning/11. Deep Learning/TensorFlow_FILES/DATA')

# gem stone dataset - price and two arbitrary features
df = pd.read_csv('../DATA/fake_reg.csv')

df.head()

sns.pairplot(df)
plt.show()

from sklearn.model_selection import train_test_split

# cannot make X a dataframe - it must be converted to a numpy array with .values
X = df[['feature1','feature2']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape
X_test.shape

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# help(Sequential)

# In Dense(), the first integer parameter represents the number of nodes or 'neurons' in that 
#   particular layer; the activation parameter accepts a string value that identifies the 
#   activation equation 
model = Sequential([Dense(4,activation='relu'),Dense(2,activation='relu'),Dense(1)])

# this same model can also be written like this:

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

# the syntax above is more convenient for editing or dropping different layers

# here is our actual model:

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')

# arbitrarily assigning epochs value - not the correct method
model.fit(x=X_train,y=y_train,epochs=250)

loss_df = pd.DataFrame(model.history.history)

loss_df.plot()
plt.show()


########################################################

## PART 3 ##

########################################################

model.evaluate(X_test,y_test,verbose=0)
model.evaluate(X_train,y_train,verbose=0)

test_predictions = model.predict(X_test)

test_predictions = pd.Series(test_predictions.reshape(300,))

pred_df = pd.DataFrame(y_test,columns=['Test True Y'])
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns = ['Test True Y','Model Predictions']

pred_df

sns.scatterplot(x='Test True Y',y='Model Predictions',data=pred_df)
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(pred_df['Test True Y'],pred_df['Model Predictions'])

new_gem = [[998,1000]]

new_gem = scaler.transform(new_gem)

# predicting price of new gem
model.predict(new_gem)

# you can save you model and load it into other products
from tensorflow.keras.models import load_model

# saving model
model.save('my_gem_model.h5')

# loading model
later_model = load_model('my_gem_model.h5')
