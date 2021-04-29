import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

plt.figure(figsize=(12,3))
# ^^ calling matplotlib figure will override sns sizing
sns.set_context('poster')
sns.set_style('ticks')
sns.countplot(x='sex',data=tips)

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm')

plt.show()