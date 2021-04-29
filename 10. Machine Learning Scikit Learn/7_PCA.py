#####################################################

## PCA Notes ##

    # Unsupervised statistical technique used to examine the interrelations among a set of variables
        # in order to identify the underlying structure of those variables
    # Also known as a general factor analysis
    # Determines several orthogonal lines (at right angles) of best fit to the data set

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

# Data-set import
from sklearn.datasets import load_breast_cancer

os.chdir('C:/Users/Sam/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/18-Principal-Component-Analysis')

cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()

cancer['target']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

# PCA import
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.show()

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
sns.heatmap(df_comp,cmap='plasma')
plt.show()