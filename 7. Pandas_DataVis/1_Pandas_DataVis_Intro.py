import os
import numpy as np
import pandas as pd
import seaborn as sns
# ^^ importing sns will apply certain styles to pandas plots without actually having to call sns methods
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\sjrus\Desktop\Machine Learning\7. Pandas_DataVis')

df1 = pd.read_csv('df1',index_col=0)
df1.index
df1.head()

df2 = pd.read_csv('df2')
df2.head()

df1['A'].hist()
# ^^ can generate plots in pandas by referencing the dataframe
plt.show()

df1['A'].plot(kind='hist')
# ^^ use .plot followed by kind arg to make plot with pandas

df1['A'].plot.hist()
# ^^ can call hist() directly too
# ^^ this will be our primary method of generating plots

df2.plot.area(alpha=.4)

df2.plot.bar(stacked=True)

df1['A'].plot.hist(bins=50)

df1.plot.line(y='B')
# ^^ if x value is not provided the index of the df is set to x by default

df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm')

df2.plot.box()

df = pd.DataFrame(np.random.randn(1000,2),columns=['a','b'])
df.plot.hexbin(x='a',y='b',gridsize=25, cmap='coolwarm')

df2['a'].plot.kde()

df2.plot.kde()

plt.show()