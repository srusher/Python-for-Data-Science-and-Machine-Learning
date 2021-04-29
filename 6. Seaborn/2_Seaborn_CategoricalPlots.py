import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)
# ^^ barplots used for categorical data
# ^^ estimator is an aggregate function
plt.show()

sns.countplot(x='sex',data=tips)
# ^^ the estimator is counting the number of occurences in a countplot
plt.show()

sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')
# ^^ typical box and whiskers plot
plt.show()

sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',split=True)
# ^^ shows the distr of data across a category (similar to box plot)
plt.show()

sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex',split=True)
# ^^ draws a scatter plot where 1 variable is categorical
# ^^ jitter arg will add random noise to better separate points
plt.show()

sns.violinplot(x='day',y='total_bill',data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips,color='black')
# ^^ similar to stripplot but points will not overlap
# ^^ this plot does not work well with large datasets
# ^^ can be combined with a violin plot with
plt.show()

sns.factorplot(x='day',y='total_bill',data=tips,kind='bar')
# ^^ accepts a 'kind' argument, which lets you call a specific plot
plt.show()