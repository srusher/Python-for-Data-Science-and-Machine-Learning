import numpy as np
import pandas as pd
from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])

# A dataframe is basically a bunch of series that share an index

print(df)

print(df['W'])

print(df[['W','X']]) # need two sets of brackets to access multiple columns

# adding columns to a dataframe:

df['new'] = df['W'] + df['Y']

print(df)

# dropping columns from a dataframe
    # have to set axis = 1 to specify columns - by default axis is set to 0 which indicates rows
    # this will just create a view - this will not actually drop the column from 'df'

df.drop('new',axis=1)

    # dropping columns from the reference df:
    # 'inplace' needs to be set to True to affect the reference df

df.drop('new',axis=1,inplace=True)

print(df)

# dropping rows

df.drop('E') # axis set to '0' by default

# selecting rows

print(df.loc['A'])

# selecting rows by index

print(df.iloc[0]) # returns row 'A'

# selecting specific values

print(df.loc['B','Y']) # single value
print(df.loc[['A','B'],['W','Y']]) # selecting multiple rows and columns