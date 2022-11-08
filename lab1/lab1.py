import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

arr_2d = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],])
print(arr_2d)

arr_int = np.array([1, 2, 3, 4, 5], dtype='i')
print(arr_int.dtype)

arr_int_4b = np.array([1, 2, 3, 4, 5], dtype='i4')
print(arr_int_4b.dtype)

arr_zeros = np.zeros(10, dtype='i')
print(arr_zeros)

arr_zeros_2d = np.zeros((5, 5))
print(arr_zeros_2d)

arr_ones = np.ones((3, 3))
print(arr_ones)

arr_lin = np.linspace(0, 10, 6)
print(arr_lin)

arr_eye = np.eye(5)
print(arr_eye)

arr_rand1 = np.random.rand(5)
arr_rand2 = np.random.rand(3, 2)
print(arr_rand1)
print(arr_rand2)

arr_randn = np.random.randn(5, 5)
print(arr_randn)

arr_randint = np.random.randint(1, 49, (3, 6))
print(arr_randint)

arr_2d2 = np.array(([5, 10, 15], [20, 25, 30], [35, 40, 45]))
print(arr_2d2)
print(arr_2d2[0])
print(arr_2d2[1, 2])
print(arr_2d2[:, 1])
print(arr_2d2[-1, -1])

matrix = np.random.randint(1, 100, (5, 5))
print(matrix)
print(matrix[1:4, 1:4])

arr = np.ones((5, 4))
print(arr)
print(arr.shape)

arr = np.ones(25)
print(arr)
arr = arr.reshape((5, 5))
print(arr)

arr = np.random.randint(1, 100, 10)
print(arr)
print(arr.min())
print(arr.max())
print(f'max value: {arr.max()}, index: {arr.argmax()}')
print(f'min value: {arr.min()}, index: {arr.argmin()}')

arr = np.arange(11)
print(arr)
print(np.sqrt(arr))
print(np.exp(arr))

arr0 = np.random.randint(1, 10, (3, 3))
arr1 = np.random.randint(1, 10, (3, 3))

print(arr0)
print(arr1)
print(arr0 + arr1)
print(arr0 - arr1)
print(arr0 * arr1)
print(arr0 ** 2)

print(arr0.dot(arr1))
print(arr0[0].dot(arr1[1]))

# TASKS
# 1
arr1 = np.ones((1, 50), dtype="i") * 5
print("\nTASK 1:")
print(arr1)

# 2
arr2 = np.linspace(1, 25, 25, dtype="i")
arr2 = arr2.reshape(5, 5)
print("\nTASK 2:")
print(arr2)

# 3
arr3 = np.linspace(10, 50, 21, dtype="i")
print("\nTASK 3:")
print(arr3)

# 4
arr4 = np.eye(6, dtype="i") * 8
print("\nTASK 4:")
print(arr4)

# 5
arr5 = np.linspace(0, 0.99, 100)
arr5 = arr5.reshape(10, 10)
print("\nTASK 5:")
print(arr5)

# 6
arr6 = np.linspace(0, 1, 50)
print("\nTASK 6:")
print(arr6)

# 7
arr7 = arr2[2:5, 1:5]
print("\nTASK 7:")
print(arr7)

# 8
arr8 = arr2[0:3, -1]
arr8 = arr8.reshape(3, 1)
print("\nTASK 8:")
print(arr8)

# 9
sum9 = np.sum(arr2[3:5, :])
print("\nTASK 9:")
print(sum9)

# 10
arr10 = np.random.randint(0, 1000, (np.random.randint(0, 10), np.random.randint(0, 10)))
print("\nTASK 10:")
print(arr10)
