# 100 numpy exercises - my solution
# Keni Mou

#%% [markdown]
# #### 1. Import the numpy package under the name `np` (★☆☆)

#%%


#%% [markdown]
# #### 2. Print the numpy version and the configuration (★☆☆)

#%%


#%% [markdown]
# #### 3. Create a null vector of size 10 (★☆☆)

#%%


#%% [markdown]
# #### 4.  How to find the memory size of any array (★☆☆)

#%%


#%% [markdown]
# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

#%%


#%% [markdown]
# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

#%%


#%% [markdown]
# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

#%%


#%% [markdown]
# #### 8.  Reverse a vector (first element becomes last) (★☆☆)

#%%


#%% [markdown]
# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

#%%


#%% [markdown]
# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

#%%


#%% [markdown]
# #### 11. Create a 3x3 identity matrix (★☆☆)

#%%


#%% [markdown]
# #### 12. Create a 3x3x3 array with random values (★☆☆)

#%%


#%% [markdown]
# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

#%%


#%% [markdown]
# #### 14. Create a random vector of size 30 and find the mean value (★☆☆)

#%%


#%% [markdown]
# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

#%%


#%% [markdown]
# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

#%%


#%% [markdown]
# #### 17. What is the result of the following expression? (★☆☆)
#%% [markdown]
# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# np.nan in set([np.nan])
# 0.3 == 3 * 0.1
# ```

#%%


#%% [markdown]
# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

#%%


#%% [markdown]
# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

#%%


#%% [markdown]
# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

#%%


#%% [markdown]
# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

#%%


#%% [markdown]
# #### 22. Normalize a 5x5 random matrix (★☆☆)

#%%


#%% [markdown]
# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

#%%


#%% [markdown]
# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

#%%


#%% [markdown]
# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

#%%


#%% [markdown]
# #### 26. What is the output of the following script? (★☆☆)
#%% [markdown]
# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

#%%


#%% [markdown]
# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
#%% [markdown]
# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

#%%


#%% [markdown]
# #### 28. What are the result of the following expressions?
#%% [markdown]
# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

#%%


#%% [markdown]
# #### 29. How to round away from zero a float array ? (★☆☆)

#%%


#%% [markdown]
# #### 30. How to find common values between two arrays? (★☆☆)

#%%


#%% [markdown]
# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

#%%


#%% [markdown]
# #### 32. Is the following expressions true? (★☆☆)
#%% [markdown]
# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

#%%


#%% [markdown]
# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

#%%


#%% [markdown]
# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

#%%


#%% [markdown]
# #### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

#%%


#%% [markdown]
# #### 36. Extract the integer part of a random array using 5 different methods (★★☆)

#%%


#%% [markdown]
# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

#%%


#%% [markdown]
# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

#%%


#%% [markdown]
# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

#%%


#%% [markdown]
# #### 40. Create a random vector of size 10 and sort it (★★☆)

#%%


#%% [markdown]
# #### 41. How to sum a small array faster than np.sum? (★★☆)

#%%


#%% [markdown]
# #### 42. Consider two random array A and B, check if they are equal (★★☆)

#%%


#%% [markdown]
# #### 43. Make an array immutable (read-only) (★★☆)

#%%


#%% [markdown]
# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

#%%


#%% [markdown]
# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

#%%


#%% [markdown]
# #### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

#%%


#%% [markdown]
# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

#%%


#%% [markdown]
# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

#%%


#%% [markdown]
# #### 49. How to print all the values of an array? (★★☆)

#%%


#%% [markdown]
# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

#%%


#%% [markdown]
# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

#%%


#%% [markdown]
# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

#%%


#%% [markdown]
# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

#%%


#%% [markdown]
# #### 54. How to read the following file? (★★☆)
#%% [markdown]
# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

#%%


#%% [markdown]
# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

#%%


#%% [markdown]
# #### 56. Generate a generic 2D Gaussian-like array (★★☆)

#%%


#%% [markdown]
# #### 57. How to randomly place p elements in a 2D array? (★★☆)

#%%


#%% [markdown]
# #### 58. Subtract the mean of each row of a matrix (★★☆)

#%%


#%% [markdown]
# #### 59. How to sort an array by the nth column? (★★☆)

#%%


#%% [markdown]
# #### 60. How to tell if a given 2D array has null columns? (★★☆)

#%%


#%% [markdown]
# #### 61. Find the nearest value from a given value in an array (★★☆)

#%%


#%% [markdown]
# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

#%%


#%% [markdown]
# #### 63. Create an array class that has a name attribute (★★☆)

#%%


#%% [markdown]
# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

#%%


#%% [markdown]
# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

#%%


#%% [markdown]
# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

#%%


#%% [markdown]
# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

#%%


#%% [markdown]
# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

#%%


#%% [markdown]
# #### 69. How to get the diagonal of a dot product? (★★★)

#%%


#%% [markdown]
# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

#%%


#%% [markdown]
# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

#%%


#%% [markdown]
# #### 72. How to swap two rows of an array? (★★★)

#%%


#%% [markdown]
# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

#%%


#%% [markdown]
# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

#%%


#%% [markdown]
# #### 75. How to compute averages using a sliding window over an array? (★★★)

#%%


#%% [markdown]
# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)

#%%


#%% [markdown]
# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

#%%


#%% [markdown]
# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

#%%


#%% [markdown]
# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

#%%


#%% [markdown]
# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

#%%


#%% [markdown]
# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)

#%%


#%% [markdown]
# #### 82. Compute a matrix rank (★★★)

#%%


#%% [markdown]
# #### 83. How to find the most frequent value in an array?

#%%


#%% [markdown]
# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

#%%


#%% [markdown]
# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)

#%%


#%% [markdown]
# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

#%%


#%% [markdown]
# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

#%%


#%% [markdown]
# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

#%%


#%% [markdown]
# #### 89. How to get the n largest values of an array (★★★)

#%%


#%% [markdown]
# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

#%%


#%% [markdown]
# #### 91. How to create a record array from a regular array? (★★★)

#%%


#%% [markdown]
# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

#%%


#%% [markdown]
# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

#%%


#%% [markdown]
# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

#%%


#%% [markdown]
# #### 95. Convert a vector of ints into a matrix binary representation (★★★)

#%%


#%% [markdown]
# #### 96. Given a two dimensional array, how to extract unique rows? (★★★)

#%%


#%% [markdown]
# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

#%%


#%% [markdown]
# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

#%%


#%% [markdown]
# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

#%%


#%% [markdown]
# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

#%%



