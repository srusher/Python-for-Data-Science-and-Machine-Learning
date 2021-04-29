import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}

df = pd.DataFrame(d)

print(df)

# dropping missing or NA data
## this will create a VIEW, it will not alter the original df - inplace must be set to true to alter df
print(df.dropna()) # drops rows with NA

print(df.dropna(axis=1)) # drops columns with NA

## thresh will set the threshold for the number of rows that are kept
print(df.dropna(thresh=2))


# filling in missing data
## this will also just create a view
print(df.fillna(value='FILL VALUE')) # replaces NA with the specified value

print(df['A'].fillna(value=df['A'].mean()))
