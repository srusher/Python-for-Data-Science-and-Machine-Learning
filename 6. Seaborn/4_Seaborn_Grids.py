import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

iris = sns.load_dataset('iris')
iris.head()
iris.info()

iris['species'].unique()
sns.pairplot(iris)
# ^^ pretty much a simplified version of PairGrid
plt.show()

g = sns.PairGrid(iris)
# ^^ this allows you to have more customization and controls over pairplots
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()

f = sns.FacetGrid(data=tips,col='time',row='smoker')
f.map(sns.distplot,'total_bill')
plt.show()



