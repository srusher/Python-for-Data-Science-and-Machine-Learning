import numpy as np
import pandas as pd

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()

print(df)
df['col2'].count()

# finding unique/distinct values
print(df['col2'].unique())

## finding number of unique values
print(df['col2'].nunique())

## displaying unique values and counts
print(df['col2'].value_counts())

# conditional selection
print(df[df['col1']>2])

# apply() method

def times2(x):
    return x*2

print(df['col1'].apply(times2))

print(df['col3'].apply(len))

print(df['col2'].apply(lambda x: x*2))


# return information about df
print(df.columns)

print(df.index)


# sorting a df
print(df.sort_values('col2'))


# finding null values
print(df.isnull())


# pivot table
## a pivot table is creates a different multi-index view from the original dataset
## ^^ it basically reorganizes a table into a potentially more complex view

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df2 = pd.DataFrame(data)

print(df2)

print(df2.pivot_table(values='D',index=['A','B'],columns=['C']))