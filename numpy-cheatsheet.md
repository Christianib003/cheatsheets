## **NumPy Cheatsheet for Machine Learning**

Below is an elaborated **NumPy Cheatsheet for Machine Learning**, designed to help you quickly reference the essential NumPy functionalities used in ML. It includes brief, understandable explanations, return values, usage examples, and edge cases where applicable. NumPy is a core Python library for numerical computing, widely used in ML for handling data in the form of arrays (vectors, matrices, etc.) efficiently. Let’s dive in!

---

### **1. Getting Started**
- **Importing NumPy**: Always start by importing NumPy, typically aliased as `np`.
  ```python
  import numpy as np
  ```

---

### **2. Array Creation**
NumPy arrays (`ndarray`) are the backbone of ML data manipulation. Here’s how to create them:

- **`np.array(object)`**
  - **Explanation**: Converts a list, tuple, or other iterable into a NumPy array.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    array = np.array([1, 2, 3])  # 1D array
    array_2d = np.array([[1, 2], [3, 4]])  # 2D array
    ```
  - **Edge Case**: Ensure consistent dimensions in nested lists, or it raises a `ValueError`.

- **`np.zeros(shape)`**
  - **Explanation**: Creates an array filled with zeros.
  - **Return**: `ndarray` of specified shape
  - **Usage**:
    ```python
    zeros = np.zeros((2, 3))  # 2x3 array of zeros
    ```

- **`np.ones(shape)`**
  - **Explanation**: Creates an array filled with ones.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    ones = np.ones((2, 3))  # 2x3 array of ones
    ```

- **`np.full(shape, value)`**
  - **Explanation**: Creates an array filled with a specified value.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    full = np.full((2, 3), 7)  # 2x3 array of 7s
    ```

- **`np.arange(start, stop, step)`**
  - **Explanation**: Creates a 1D array with evenly spaced values.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    range_array = np.arange(0, 10, 2)  # array([0, 2, 4, 6, 8])
    ```

- **`np.linspace(start, stop, num)`**
  - **Explanation**: Creates a 1D array with `num` evenly spaced values between `start` and `stop`.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    linspace = np.linspace(0, 1, 5)  # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    ```

- **`np.random.random(shape)`**
  - **Explanation**: Creates an array of random floats between 0 and 1.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    random = np.random.random((2, 2))  # 2x2 array of random floats
    ```

---

### **3. Array Attributes**
Understand your array’s properties:

- **`array.shape`**
  - **Explanation**: Returns the dimensions (rows, columns) of the array.
  - **Return**: Tuple
  - **Usage**:
    ```python
    array = np.array([[1, 2], [3, 4]])
    print(array.shape)  # (2, 2)
    ```

- **`array.ndim`**
  - **Explanation**: Number of dimensions (axes).
  - **Return**: Integer
  - **Usage**:
    ```python
    print(array.ndim)  # 2
    ```

- **`array.size`**
  - **Explanation**: Total number of elements.
  - **Return**: Integer
  - **Usage**:
    ```python
    print(array.size)  # 4
    ```

- **`array.dtype`**
  - **Explanation**: Data type of the array’s elements.
  - **Return**: `dtype` object (e.g., `int64`, `float32`)
  - **Usage**:
    ```python
    print(array.dtype)  # int64
    ```

---

### **4. Indexing and Slicing**
Access and modify array elements or subsets:

- **Basic Indexing**: `array[i]` or `array[i, j]`
  - **Return**: Single element
  - **Usage**:
    ```python
    array = np.array([1, 2, 3])
    print(array[0])  # 1
    array_2d = np.array([[1, 2], [3, 4]])
    print(array_2d[1, 0])  # 3
    ```

- **Slicing**: `array[start:stop:step]`
  - **Return**: `ndarray` (view of original)
  - **Usage**:
    ```python
    array = np.array([1, 2, 3, 4, 5])
    print(array[1:4])  # [2, 3, 4]
    print(array_2d[0:2, 1])  # [2, 4]
    ```
  - **Edge Case**: Modifying a slice modifies the original array (it’s a view).

- **Boolean Indexing**: `array[condition]`
  - **Return**: `ndarray` with elements where condition is `True`
  - **Usage**:
    ```python
    array = np.array([1, 2, 3, 4])
    print(array[array > 2])  # [3, 4]
    ```

- **Fancy Indexing**: `array[[indices]]`
  - **Return**: `ndarray` (copy of elements)
  - **Usage**:
    ```python
    array = np.array([10, 20, 30, 40])
    print(array[[1, 3]])  # [20, 40]
    ```
  - **Edge Case**: Returns a copy, not a view—changes don’t affect the original.

---

### **5. Array Operations**
Perform efficient computations:

- **Element-wise Arithmetic**: `+`, `-`, `*`, `/`, `**`
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    a = np.array([1, 2])
    b = np.array([3, 4])
    print(a + b)  # [4, 6]
    print(a * 2)  # [2, 4]
    ```

- **Universal Functions (ufuncs)**: `np.sin`, `np.exp`, `np.log`, etc.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    array = np.array([0, 1, 2])
    print(np.exp(array))  # [1. , 2.718..., 7.389...]
    ```

- **Aggregate Functions**: `np.sum`, `np.mean`, `np.std`, `np.min`, `np.max`
  - **Return**: Scalar or `ndarray` (if `axis` specified)
  - **Usage**:
    ```python
    array = np.array([[1, 2], [3, 4]])
    print(np.sum(array))  # 10
    print(np.mean(array, axis=0))  # [2., 3.]
    ```

- **Broadcasting**: Operate on arrays of different shapes.
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    array = np.array([[1, 2], [3, 4]])
    print(array + 5)  # [[6, 7], [8, 9]]
    print(array + np.array([1, 2]))  # [[2, 4], [4, 6]]
    ```
  - **Edge Case**: Shapes must be compatible, or it raises a `ValueError`.

---

### **6. Reshaping and Manipulating Arrays**
Prepare data for ML models:

- **`array.reshape(shape)`**
  - **Return**: `ndarray` with new shape
  - **Usage**:
    ```python
    array = np.arange(6)  # [0, 1, 2, 3, 4, 5]
    reshaped = array.reshape((2, 3))  # [[0, 1, 2], [3, 4, 5]]
    ```
  - **Edge Case**: Total size must match, or it raises a `ValueError`.

- **`array.flatten()`**
  - **Return**: 1D `ndarray` (copy)
  - **Usage**:
    ```python
    array = np.array([[1, 2], [3, 4]])
    flat = array.flatten()  # [1, 2, 3, 4]
    ```

- **`np.concatenate((arrays), axis)`**
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    a = np.array([[1, 2]])
    b = np.array([[3, 4]])
    concat = np.concatenate((a, b), axis=0)  # [[1, 2], [3, 4]]
    ```

- **`np.vstack((arrays))`** / **`np.hstack((arrays))`**
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    a = np.array([1, 2])
    b = np.array([3, 4])
    vstack = np.vstack((a, b))  # [[1, 2], [3, 4]]
    hstack = np.hstack((a, b))  # [1, 2, 3, 4]
    ```

- **`np.split(array, indices_or_sections)`**
  - **Return**: List of `ndarray`s
  - **Usage**:
    ```python
    array = np.array([1, 2, 3, 4])
    split = np.split(array, 2)  # [array([1, 2]), array([3, 4])]
    ```

---

### **7. Linear Algebra**
Key for ML algorithms like regression and neural networks:

- **`np.dot(a, b)`** or **`a @ b`**
  - **Return**: Scalar (dot product) or `ndarray` (matrix multiplication)
  - **Usage**:
    ```python
    a = np.array([1, 2])
    b = np.array([3, 4])
    print(np.dot(a, b))  # 11
    matrix = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
    print(matrix)  # [[19, 22], [43, 50]]
    ```

- **`array.T`** or **`np.transpose(array)`**
  - **Return**: `ndarray`
  - **Usage**:
    ```python
    array = np.array([[1, 2], [3, 4]])
    print(array.T)  # [[1, 3], [2, 4]]
    ```

- **`np.linalg.inv(array)`**
  - **Return**: `ndarray` (inverse matrix)
  - **Usage**:
    ```python
    array = np.array([[1, 2], [3, 4]])
    print(np.linalg.inv(array))  # [[-2. ,  1. ], [ 1.5, -0.5]]
    ```
  - **Edge Case**: Raises `LinAlgError` if matrix is singular (non-invertible).

- **`np.linalg.det(array)`**
  - **Return**: Float (determinant)
  - **Usage**:
    ```python
    print(np.linalg.det(array))  # -2.0
    ```

- **`np.linalg.eig(array)`**
  - **Return**: Tuple of eigenvalues (`ndarray`) and eigenvectors (`ndarray`)
  - **Usage**:
    ```python
    eigenvalues, eigenvectors = np.linalg.eig(array)
    ```

---

### **8. Random Number Generation**
Useful for initializing weights, shuffling data, etc.:

- **`np.random.rand(shape)`**
  - **Return**: `ndarray` (uniform [0, 1))
  - **Usage**:
    ```python
    rand = np.random.rand(2, 2)
    ```

- **`np.random.randn(shape)`**
  - **Return**: `ndarray` (standard normal distribution)
  - **Usage**:
    ```python
    normal = np.random.randn(2, 2)
    ```

- **`np.random.randint(low, high, shape)`**
  - **Return**: `ndarray` (random integers)
  - **Usage**:
    ```python
    ints = np.random.randint(0, 10, (2, 2))
    ```

- **`np.random.choice(array, size, replace=True/False)`**
  - **Return**: `ndarray` (random samples)
  - **Usage**:
    ```python
    choices = np.random.choice([1, 2, 3], size=2, replace=False)  # e.g., [2, 1]
    ```

---

### **9. Handling Missing Values**
Deal with incomplete data:

- **`np.nan`**
  - **Explanation**: Represents a missing or undefined value.
  - **Usage**:
    ```python
    array = np.array([1, 2, np.nan])
    ```

- **`np.isnan(array)`**
  - **Return**: Boolean `ndarray`
  - **Usage**:
    ```python
    print(np.isnan(array))  # [False, False, True]
    ```
  - **Edge Case**: NaN propagates in calculations (e.g., `np.sum(array)` returns `nan`).

---

### **10. ML-Specific Techniques**
Common operations in ML workflows:

- **Normalization**:
  - **Return**: `ndarray` (values scaled to [0, 1])
  - **Usage**:
    ```python
    array = np.array([1, 2, 3, 4])
    normalized = (array - np.min(array)) / (np.max(array) - np.min(array))
    print(normalized)  # [0. , 0.333..., 0.666..., 1. ]
    ```

- **Standardization**:
  - **Return**: `ndarray` (zero mean, unit variance)
  - **Usage**:
    ```python
    standardized = (array - np.mean(array)) / np.std(array)
    ```

- **One-Hot Encoding**:
  - **Return**: `ndarray` (binary matrix)
  - **Usage**:
    ```python
    labels = np.array([0, 1, 2])
    one_hot = np.zeros((labels.size, labels.max() + 1))
    one_hot[np.arange(labels.size), labels] = 1
    print(one_hot)  # [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    ```

- **Euclidean Distance**:
  - **Return**: Float
  - **Usage**:
    ```python
    a = np.array([1, 2])
    b = np.array([3, 4])
    distance = np.sqrt(np.sum((a - b) ** 2))  # 2.828...
    ```

---

### **11. Performance Tips and Edge Cases**
- **Vectorization**: Avoid loops; use array operations for speed.
  ```python
  array = np.array([1, 2, 3])
  result = array * 2  # Faster than a for loop
  ```

- **Views vs. Copies**:
  - Slicing returns a view (modifies original).
  - Fancy indexing returns a copy (doesn’t modify original).

- **Data Types**: Arrays have fixed types; operations may cast (e.g., `int` to `float`).
  ```python
  array = np.array([1, 2], dtype=int)
  print(array / 2)  # [0.5, 1. ] (float64)
  ```

- **In-place Operations**: Save memory with `+=`, `*=`, etc.
  ```python
  array += 1  # Modifies array directly
  ```

- **Broadcasting Errors**: Ensure compatible shapes to avoid `ValueError`.

---

### Resources:
- Youtube: [NumPy Full Python Course - NeuralNine](https://youtu.be/4c_mwnYdbhQ?si=4uxl7w7v5FVhYCrm)

*~Happy Learning!!~*