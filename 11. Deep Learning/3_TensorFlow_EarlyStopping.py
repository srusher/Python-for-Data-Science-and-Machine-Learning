# Directory imports
import os

# Data manipulation imports 
import pandas as pd
import numpy as np

# Data Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/Sam/Desktop\Machine Learning/11. Deep Learning/TensorFlow_FILES/DATA')

df = pd.read_csv('cancer_classification.csv')

df.info()
df.describe().transpose()

sns.countplot(x='benign_0__mal_1',data=df)
plt.show()

df.corr()['benign_0__mal_1'].sort_values()

sns.heatmap(df.corr())
plt.show()

X = df.drop('benign_0__mal_1',axis=1)
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#################################################

# PART 2: Preventing Overfitting

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
X_train.shape

model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')

# this model will intentionally use too many epochs
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test))

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

# the plot above demonstrates a good example of overfitting


model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

# refitting the model with the same amount of epochs but with an early stop parameter
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

##########################################

# DROP-OUT LAYERS

from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dropout(rate=0.5)) # adding drop-out layer

model.add(Dense(15,activation='relu'))
model.add(Dropout(rate=0.5)) # adding drop-out layer

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

model_loss_2 = pd.DataFrame(model.history.history)
model_loss_2.plot()
plt.show()

predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))