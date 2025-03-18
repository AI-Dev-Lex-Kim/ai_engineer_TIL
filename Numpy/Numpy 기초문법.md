- [numpy.ndarray 속성값](#numpyndarray-속성값)
- [배열 생성하기](#배열-생성하기)
  - [np.array()](#nparray)
  - [np.zeros(), np.ones(), np.empty()](#npzerosnponesnpempty)
  - [zero\_like](#zero_like)
  - [np.arange(),np.linspace()](#nparangenplinspace)
- [배열 변경](#배열-변경)
  - [reshape()](#reshape)
  - [np.vstack(), np.hstack() 데이터 합치기](#npvstack-nphstack-데이터-합치기)
  - [np.hsplit() 데이터 나누기](#nphsplit-데이터-나누기)
- [데이터 복사](#데이터-복사)
  - [view](#view)
  - [copy](#copy)
- [기본 연산](#기본-연산)
  - [sum(), min(), max(), argmax(), cumsum()](#sum-min-max-argmax-cumsum)
  - [np.fromfunction()](#npfromfunction)
  - [.flat](#flat)

## numpy.ndarray 속성값

```python
a = np.arange(15).reshape(3, 5)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]

a.shape
# (3, 5) 배열의 각 축(axis)의 크기

a.ndim
# 2 축의 개수(Dimension)

a.dtype
# int64 각 요소(Element)의 타입

a.itemsize
# 8 각 요소(Element)의 타입의 bytes 크기

a.size
# 15 전체 요소(Element)의 개수

type(a)
# <class 'numpy.ndarray'>

```

- ndarray.shape : 배열의 각 축(axis)의 크기
- ndarray.ndim : 축의 개수(Dimension)
- ndarray.dtype : 각 요소(Element)의 타입
- ndarray.itemsize : 각 요소(Element)의 타입의 bytes 크기
- ndarray.size : 전체 요소(Element)의 개수

## 배열 생성하기

### np.array()

```python
a = np.array([2,3,4])
print(a)
# [2 3 4]
print(a.dtype)
# int64

b = np.array([1.2, 3.5, 5.1])
print(b.dtype)
# float64
```

반드시 파라미터로 배열을 넣어주어야함.

2D 배열이나 3D 배열등도 마찬가지 방법으로 입력으로 주면 생성할 수 있다.

```python
b **=** np.array([(1.5,2,3), (4,5,6)])
**print**(b)
*# [[1.5 2.  3. ]
#  [4.  5.  6. ]]*
```

### np.zeros(), np.ones(), np.empty()

`np.zeros()`, `np.ones()`, `np.empty()`를 이용하여 다양한 차원의 데이터를 쉽게 생성할 수 있다.

- `np.zeros(shape)` : 0으로 구성된 N차원 배열 생성
- `np.ones(shape)` : 1로 구성된 N차원 배열 생성
- `np.empty(shape)` : 초기화되지 않은 N차원 배열 생성

```python
#[3,4] 크기의 배열을 생성하여 0으로 채움
print(np.zeros((3,4)))

# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# [2,3,4] 크기의 배열을 생성하여 1로 채움
print(np.ones((2,3,4), dtype=np.int16))
# [[[1 1 1 1]
#   [1 1 1 1]
#   [1 1 1 1]]

#  [[1 1 1 1]
#   [1 1 1 1]
#   [1 1 1 1]]]

# 초기화 되지 않은 [2,3] 크기의 배열을 생성
print(np.empty((2,3)))
# [[1.39069238e-309 1.39069238e-309 1.39069238e-309]
#  [1.39069238e-309 1.39069238e-309 1.39069238e-309]]
```

### zero_like

```python
a = np.array([1,2,3])
np.zeros_like(a)
# array([0, 0, 0])

A = np.array([[1,2,3],[4,5,6]])
np.zeros_like(A)
# array([[0, 0, 0],
#        [0, 0, 0]])
```

다른 배열의 행과 열에 요소만 0으로 가득채운다.

### np.arange(),np.linspace()

```python
np.arange(): N 만큼 차이나는 숫자 생성
np.linspace(): N 등분한 숫자 생성
# 10이상 30미만 까지 5씩 차이나게 생성
print(np.arange(10, 30, 5))
# [10 15 20 25]

# 0이상 2미만 까지 0.3씩 차이나게 생성
print(np.arange(0, 2, 0.3))
# [0.  0.3 0.6 0.9 1.2 1.5 1.8]

# 0~99까지 100등분
x = np.linspace(0, 99, 100)
print(x)
# [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
#  18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.
#  36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53.
#  54. 55. 56. 57. 58. 59. 60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70. 71.
#  72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82. 83. 84. 85. 86. 87. 88. 89.
#  90. 91. 92. 93. 94. 95. 96. 97. 98. 99.]
```

## 배열 변경

### reshape()

`reshape()`을 통해 데이터는 그대로 유지한 채 차원을 쉽게 변경해준다.

```python
b = np.arange(12).reshape(4,3)
print(b)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

```

요소값은 그대로 하면서 4행, 3열로 만들어줌.

`.T` 는 전치가 가능하게한다.

원본 복사x, 복사하여 결과 return 함

```python
a = np.floor(10*npr.random((3,4)))
print(a)
# [[8. 0. 0. 6.]
#  [1. 4. 3. 0.]
#  [0. 3. 1. 9.]]

print(a.shape)
# (3, 4)

# 모든 원소를 1차원으로 변경
print(a.ravel())
# [8. 0. 0. 6. 1. 4. 3. 0. 0. 3. 1. 9.]

# [3,4] => [2,6]로 변경
print(a.reshape(2,6))
# [[8. 0. 0. 6. 1. 4.]
#  [3. 0. 0. 3. 1. 9.]]

# [3,4]의 전치(transpose)변환으로 [4,3]
print(a.T)
# [[8. 1. 0.]
#  [0. 4. 3.]
#  [0. 3. 1.]
#  [6. 0. 9.]]

print(a.T.shape)
# (4, 3)

print(a.shape)
# (3, 4)
```

### np.vstack(), np.hstack() 데이터 합치기

두 데이터를 합칠 수 있다.

- `np.vstack()`: axis=0 기준으로 쌓음
- `np.hstack()`: axis=1 기준으로 쌓음

```python
a = np.floor(10*npr.random((2,2)))
print(a)
# [[1. 4.]
#  [2. 4.]]

b = np.floor(10*npr.random((2,2)))
print(b)
# [[3. 7.]
#  [3. 7.]]

# [2,2] => [4,2]
print(np.vstack((a,b)))
# [[1. 4.]
#  [2. 4.]
#  [3. 7.]
#  [3. 7.]]

# [2,2] => [2,4]
print(np.hstack((a,b)))
# [[1. 4. 3. 7.]
#  [2. 4. 3. 7.]]
```

### np.hsplit() 데이터 나누기

axis=1 기준 인덱스로 데이터를 분할할 수 있습니다.

```python
a = np.floor(10*npr.random((2,12)))
print(a)
# [[4. 4. 1. 7. 7. 8. 8. 8. 4. 3. 5. 3.]
#  [9. 8. 7. 5. 6. 8. 9. 6. 9. 5. 4. 7.]]

# [2,12] => [2,4] 데이터 3개로 등분
print(np.hsplit(a, 3))
# [array([[4., 4., 1., 7.],
#        [9., 8., 7., 5.]]), array([[7., 8., 8., 8.],
#        [6., 8., 9., 6.]]), array([[4., 3., 5., 3.],
#        [9., 5., 4., 7.]])]

# [2,12] => [:, :3], [:, 3:4], [:, 4:]로 분할
print(np.hsplit(a, (3,4)))
# [array([[4., 4., 1.],
#        [9., 8., 7.]]), array([[7.],
#        [5.]]), array([[7., 8., 8., 8., 4., 3., 5., 3.],
#        [6., 8., 9., 6., 9., 5., 4., 7.]])]
```

## 데이터 복사

### view

얕은 복사로 요소의 객체 참조값이 같아, 원본이 변경될수있다.

```python
c = a.view()
# c와 a의 참조값은 다름
print(c is a)
# False

c = c.reshape((2, 6))
print(a.shape)
# (3, 4)

# c의 데이터와 a의 데이터의 참조값은 같음
c[0, 4] = 1234
print(a)
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]

# a를 슬라이싱해도 데이터의 참조값은 같음
s = a[ : , 1:3]
s[:] = 10
print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]
```

### copy

깊은 복사를 통해 요소의 객체 참조값이 중복되지 않는다.

```python
d = a.copy()
# a와 d의 참조값은 다름
print(d is a)
# False

# a와 d의 데이터의 참조값도 다름
d[0,0] = 9999
print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]
```

## 기본 연산

숫자가 각각의 요소에 연산이 적용된다.

```python
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print(b)
# [0 1 2 3]

# a에서 b에 각각의 원소를 -연산
c = a-b
print(c)
# [20 29 38 47]

# b 각각의 원소에 제곱 연산
print(b**2)
# [0 1 4 9]

print(a<35)
# [ True  True False False]
```

- : 각각의 원소끼리 곱셈 (Elementwise product, Hadamard product)
- `@` : 행렬 곱셈 (Matrix product)
- `.dot()` : 행렬 내적 (dot product)

```python
A = np.array( [[1,1],
               [0,1]] )
B = np.array( [[2,0],
               [3,4]] )
print(A * B)
# [[2 0]
#  [0 4]]

print(A @ B)
# [[5 4]
#  [3 4]]

print(A.dot(B))
# [[5 4]
#  [3 4]]
```

### sum(), min(), max(), argmax(), cumsum()

- `.sum()`: 모든 요소의 합
- `.min()`: 모든 요소 중 최소값
- `.max()`: 모든 요소 중 최대값
- `.argmax()`: 모든 요소 중 최대값의 인덱스
- `.cumsum()`: 모든 요소의 누적합

```python
a = np.arange(8).reshape(2, 4)**2
print(a)
# [[ 0  1  4  9]
#  [16 25 36 49]]

# 모든 요소의 합
print(a.sum())
# 140

# 모든 요소 중 최소값
print(a.min())
# 0

# 모든 요소 중 최대값
print(a.max())
# 49

# 모든 요소 중 최대값의 인덱스
print(a.argmax())
# 7

# 모든 요소의 누적합
print(a.cumsum())
# [  0   1   5  14  30  55  91 140]
```

axis 값을 입력하면 행과 열을 기준으로도 연산할 수 있다.

`axis=0`은 `shape`에서 첫번째부터 순서대로 해당된다.

```python
b **=** np.arange(12).reshape(3,4)
**print**(b)
*# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]*
**print**(b.sum(axis**=**0))
*# [12 15 18 21]
# [*[0 + 4 + 8], [1 + 5 + 9]...*]*
**print**(b.sum(axis**=**1))
*# [ 6 22 38]*
```

0은 열 방향으로 계산

### np.fromfunction()

각 요소에 행열 인덱스를 바탕으로 함수 계산이 가능하다.

```python
def f(x,y):
    return 10*x+y

b = np.fromfunction(f, (5,4), dtype=int)
print(b)
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]
#  [30 31 32 33]
#  [40 41 42 43]]

print(b[2,3])
# 23

print(b[0:5, 1])
# [ 1 11 21 31 41]

print(b[ : ,1])
# [ 1 11 21 31 41]

print(b[1:3, : ])
# [[10 11 12 13]
#  [20 21 22 23]]

print(b[-1])
# [40 41 42 43]
```

### .flat

flat을 통해 for 문 계산이 가능하다.

```python
for row in b:
    print(row)
# [0 1 2 3]
# [10 11 12 13]
# [20 21 22 23]
# [30 31 32 33]
# [40 41 42 43]

for element in b.flat:
    print(element)
# 0
# 1
# 2
# 3
# 10
# 11
# 12
# 13
# 20
# 21
# 22
# 23
# 30
# 31
# 32
# 33
# 40
# 41
# 42
# 43
```

참고

- [고려대학교 Numpy 정리 사이트](https://compmath.korea.ac.kr/appmath/NumpyBasics.html#)
- [Numpy 공식홈페이지](https://numpy.org/devdocs/user/quickstart.html)
- [Numpy Quiz](https://www.w3schools.com/quiztest/quiztest.asp?qtest=NUMPY)
- [Numpy geeksforgeeks](https://www.geeksforgeeks.org/introduction-to-numpy/?ref=gcse_outind#key-features-of-numpy)
- [Numpy 공홈 번역 블로그](https://laboputer.github.io/machine-learning/2020/04/25/numpy-quickstart/)
