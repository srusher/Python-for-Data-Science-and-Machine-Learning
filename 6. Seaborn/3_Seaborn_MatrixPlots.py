import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
tips.head()
flights.head()

tc = tips.corr()
# ^^ corr() will produce a matrix form of the data: the index will now be labeled with a variable

sns.heatmap(tc,annot=True,cmap='coolwarm')
# ^^ heatmaps are good for visualizing a correlation between 2 variables
# ^^ annot arg will add numbers into the heatmap squares
# ^^ cmap arg stand for color map
plt.show()

fp = flights.pivot_table(index='month',columns='year',values='passengers')
# ^^ index arg will set that variable as the index for each row in the matrix
# ^^ columns arg will set that variable as the column values
# ^^ values arg will set that variable as the data for this matrix

sns.heatmap(fp,cmap='magma',linecolor='white',linewidths=1)
# ^^ linecolor will change color of grid lines in heatmap
# ^^ linewidths will expand grid line and separate heatmap squares/grids
plt.show()

## CLUSTERING METHODS ##
sns.clustermap(fp,cmap='coolwarm',standard_scale=1)
# ^^ this will dispay hierarchical clustering
# ^^ standard_scale arg will normalize the data
plt.show()