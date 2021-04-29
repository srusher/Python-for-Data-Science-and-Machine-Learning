import numpy as np

arr = np.arange(0,11)

print(arr)

print(arr[8])

print(arr[0:5])

print(arr[0:])

# NumPy arrays can broadcast values to arrays like this:

arr[0:5] = 100 # this replaces the first 5 elements with 100

print(arr)

# if you want to make a copy of an array instead creating a reference to the original array then use the copy() method:

arr_copy = arr.copy() # this will prevent the original arr from being changed when arr_copy is changed

arr_copy[0:4] = 69

print(arr_copy)
print(arr)

mat1 = np.array([[5,10,15],[20,25,30],[35,40,45]])

print(mat1)

arr1 = np.arange(0,11)

print(arr1[arr1>5]) # comparison operators used in the indexing format will return the True values for the comparison

