import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])

# This section will focus on Conditional Selection

print(df > 0) # prints a boolean dataframe

print(df['W'] > 0) # prints boolean values from column 'W'
print(df[df['W'] > 0]) # print the entire dataframe with the condition that 'W' is greater than 0
print(df['Z']<0)
print(df[df['W']>0][['Y','X']]) # show the rows in columns X and Y where W is > 0

# the variables below will accomplish the same thing as the line of code above

boolser = df['W']>0
result = df[boolser]
mycols = ['Y','X']
print(result[mycols])

print(df[df['W']>0][df['Y']>1]) # show rows where W is greater than 0 and where Y is greater than 1

# ^^ this can also be done with & operator below:

print(df[(df['W']>0) & (df['Y']>1)])

## Indexing concepts

df.reset_index() # this will reset the index of the df to a numerical value

    # setting the index

newIDX = 'CA NY WY OR CO'.split()

df['State'] = newIDX # adding column States and the row with each state value
print(df)



