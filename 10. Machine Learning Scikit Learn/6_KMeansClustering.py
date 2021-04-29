#####################################################

## K Means Clustering Notes ##

    # unsupervised learning model that attempts to group unlabeled data
    # randomly assigns a number of data points that will act as centroids for each cluster (number is desginated ny k-value)

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

os.chdir('C:/Users/Sam/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering')

# generating data with scikit-learn
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)

plt.scatter(data[0][:,0],data[0][:,1])
plt.show()

# import KMeans fucntion
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])
kmeans.cluster_centers_

fig , (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_)

ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()