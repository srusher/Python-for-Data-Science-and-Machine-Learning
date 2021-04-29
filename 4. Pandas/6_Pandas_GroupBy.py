import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

# think of the GroupBy function in Pandas as the GroupBy clause in SQL
## In SQL: typically used for aggregate functions and returns values for each distinct row


# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)

print(df)

byComp = df.groupby('Company')
print(byComp.mean())

print(byComp.sum())

print(df.groupby('Company').sum().loc['MSFT'])

# .describe() gives you a bunch of useful info about the dataframe
print(df.groupby('Company').describe())

# you can also use .transpose() to alter the way the resulting table looks
print(df.groupby('Company').describe().transpose())



