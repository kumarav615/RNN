# Python Program illustrating
# working of argmax()

import numpy as geek

# Working on 2D array
array = geek.arange(12).reshape(3, 4)
print("INPUT ARRAY : \n", array)

# No axis mentioned, so works on entire array
print("\nMax element : ", geek.argmax(array))

# returning Indices of the max element
# as per the indices
print("\nIndices of Max element : ", geek.argmax(array, axis=0))
print("\nIndices of Max element : ", geek.argmax(array, axis=1))

print([geek.argmax(a) for a in array])