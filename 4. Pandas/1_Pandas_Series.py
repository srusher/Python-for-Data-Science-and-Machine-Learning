import numpy as np
import pandas as pd

labels = ['a','b','c']
my_data = [10,20,30]
arr = np.array(my_data)

d = {'a':10,'b':20,'c':30}

# know the difference between arrays in NumPy and series in Pandas

print(pd.Series(data = my_data))

print(pd.Series(data=my_data,index=labels)) # gives you labels for each index position

pd.Series(arr,labels) # pass in numpy array instead

pd.Series(d) # pass in the dictionary

# panda series can hold a variety of data types

pd.Series(data=[sum,print,len]) # this isn't common but panda series can store functions as a data type

ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])

ser2 = pd.Series([1,2,5,4],['USA','Germany','Italy','Japan'])

print(ser1['Germany']) # access values like you would in a dictionary

print(ser2['Italy'])

print(ser1+ser2) # operations will convert integer types to float types

