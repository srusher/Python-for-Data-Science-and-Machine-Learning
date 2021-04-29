import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

# creating a df from the variables above
# this will produce a multi-index table (index within an index) G1 - 1, 2, 3 ; G2 - 1, 2, 3
df = pd.DataFrame(np.random.randn(6,2),hier_index,['A','B'])

print(df)

print(df.loc['G1'].loc[1])

# This will label the indices:
df.index.names = ['Groups','Num']

print(df)

# indexing a single value
print(df.loc['G2'].loc[2]['B'])
print(df.loc['G1'].loc[3]['A'])

# cross-section
print(df.xs('G1'))

# can grab two sub-indices from different indices
print(df.xs(1,level='Num'))