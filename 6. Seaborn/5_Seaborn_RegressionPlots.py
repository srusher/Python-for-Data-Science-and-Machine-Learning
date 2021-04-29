import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','v'],scatter_kws=({'s':100}))
# ^^ produces a linear regression model
# ^^ hue will create 2 linear models based on the sex variable
# ^^ scatter_kws can pass in a dict to make data points bigger
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',aspect=.6,size=8)
# ^^ col arg can generate two separate graphs (side-by-side) based on sex 
plt.show()