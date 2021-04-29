list1 = [1,2,3]

import numpy as np

arr = np.array(list1)
print(arr)

my_mat = [[1,2,3],[4,5,6],[7,8,9]]
print(np.array(my_mat))

print(np.arange(0,10,2)) # arange will generate an array of a specified size

np.zeros(3) # generate an array of 3 zeros

print(np.zeros((5,5))) # generate a matrix of zeros by passing in a tuple

print(np.linspace(0,5,10))

print(np.eye(4)) #generates an identity matrix

# Generating Random Numbers

print(np.random.rand(5))

print(np.random.rand(5,5))

print(np.random.randn(2))

print(np.random.randn(2,2))

print(np.random.randint(1,100))

print(np.random.randint(1,100,10))

arr = np.arange(25)

ranarr = np.random.randint(0,50,10)

arr.reshape(5,5) # reshapes the 'arr' array variable into the specified matrix type

ranarr.max() # produces max value from matirx

ranarr.argmax() # returns the index of the max value

arr.dtype # returns the data type of an array




