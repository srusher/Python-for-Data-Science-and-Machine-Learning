#####################################################

## Recommender Systems Notes ##

    # 2 Types: Content-Based and Collaborative Filtering (CF)

        # CB focus on attributes of a particular item and recommend similar items
        # CF focus on knowledge of user's preference for an item (more commonly used)
            # Memory-Based CF and Model-Based CF

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

os.chdir('C:/Users/Sam/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems')

column_names = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('u.data',sep='\t',names=column_names)

df.head()

movie_titles = pd.read_csv('Movie_Id_Titles')

# merging movie title on item ID (adds a movie title column associated with item ID)
df = pd.merge(df,movie_titles,on='item_id')

df.head()

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings.head()

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

ratings.head()

sns.histplot(data=ratings,x='num of ratings',bins=70)
plt.show()

sns.histplot(data=ratings,x='rating',bins=70)
plt.show()

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=.5)
plt.show()


###################################################

## Recommender Systems: Part 2

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()

ratings.sort_values('num of ratings',ascending=False).head(10)

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

starwars_user_ratings.head()

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_starwars.head()

# this will show that there are completely unrelated movies that are perfectly correlated with
## Star Wars
corr_starwars.sort_values('Correlation',ascending=False).head(10)

# this is because some user's may have only rated one other movie

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()