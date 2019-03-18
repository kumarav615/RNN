# Python Program illustrating
# working of argmax()

import numpy as geek

from numpy import array
# Working on 2D array
array = array([
	[0.1, 100.0],
	[0.2, 200.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])

print("INPUT ARRAY : \n", array)

# No axis mentioned, so works on entire array
print("\nMax element : ", geek.argmax(array))

# returning Indices of the max element
# as per the indices
print("\nIndices of Max element : ", geek.argmax(array, axis=0)) #<----First Column, Second COlumn. Each column max value
print("\nIndices of Max element : ", geek.argmax(array, axis=1)) #Each Row max value

print([geek.argmax(a) for a in array])
