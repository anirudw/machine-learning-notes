## NumPy array creation

```python

import numpy as np

import time

  

np.zeros(4)

np.zeros((4, 2))

np.random.random_sample((4, 2))

np.arange(4.)

np.random.rand(4)

np.array([5, 4, 3, 2])

np.array([5., 4, 3, 2])

```

  

## Shapes and dimensions

- `x.shape` gives the dimensions of an array

- `x.ndim` gives the number of dimensions

- `reshape` changes the shape without changing the data

  

```python

x.shape

x.ndim

x.reshape(m, 1)

x.reshape(1, m)

```

  

## 1-D indexing

- `a[i]` returns a scalar

- negative indices count from the end

- index must be in range

  

```python

a = np.arange(10)

a[2]

a[-1]

a[2:7:1]

a[2:7:2]

a[3:]

a[:3]

a[:]

```

  

## 2-D indexing

- `A[i, j]` selects one element

- `A[i]` selects row `i`

- `A[:, j]` selects column `j`

- `A[i, :]` selects row `i`

  

```python

A = np.arange(6).reshape(-1, 2)

A[2, 0]

A[2]

B = np.arange(20).reshape(-1, 10)

B[0, 2:7:1]

B[:, 2:7:1]

B[:,:]

B[1, :]

B[1]

```

  

## Element-wise operations

- NumPy arithmetic is element-wise by default

  

```python

a = np.array([1, 2, 3, 4])

b = np.array([-1, -2, 3, 4])

  

a + b

a - b

a * b

a / b

a ** 2

-a

```

  

## Broadcasting

- Scalars automatically expand across arrays

- Shapes must be compatible

  

```python

5 * a

a + 5

```

  

Example of invalid shapes:

```python

np.array([1, 2, 3, 4]) + np.array([1, 2])

```

  

## Dot product

For vectors:

$$

a \cdot b = \sum_{i=1}^{n} a_i b_i

$$

  

Loop version:

```python

def my_dot(a, b):

    x = 0

    for i in range(a.shape[0]):

        x = x + a[i] * b[i]

    return x

```

  

NumPy version:

```python

np.dot(a, b)

```

  

## Vectorization

- Vectorized code replaces explicit loops with array operations

- Usually much faster than Python loops

  

Example:

```python

f = w * x + b

```

  

instead of:

```python

f = np.zeros(m)

for i in range(m):

    f[i] = w * x[i] + b

```

  

## Matrix and vector shapes

Common course notation:

- `X` has shape `(m, n)`

- `w` has shape `(n,)`

- output of `X @ w` has shape `(m,)` in the vectorized form used here

  

Example:

```python

X = np.array([[1], [2], [3], [4]])

w = np.array([2])

np.dot(X[1], w)

```

  

## Key NumPy tools

- `np.array`

- `np.zeros`

- `np.arange`

- `np.random.rand`

- `np.random.random_sample`

- `reshape`

- slicing with `:`

- element-wise arithmetic

- `np.sum`

- `np.mean`

- `np.dot`

- `@` for matrix multiplication
  

## Quick reference

```python

# create

np.array(...)

np.zeros(...)

np.ones(...)

np.arange(...)

np.random.rand(...)

np.random.random_sample(...)

  

# inspect

x.shape

x.ndim

x.reshape(...)

  

# access

a[i]

a[i:j:k]

A[i, j]

A[i, :]

A[:, j]

  

# math

a + b

a * b

np.sum(a)

np.mean(a)

np.dot(a, b)

```

---
## Tags

#python #numpy