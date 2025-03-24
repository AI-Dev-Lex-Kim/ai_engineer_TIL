- [**Pandas에서 날짜 데이터 타입으로 변경하는 방법**](#--pandas------------------------)

- [**1. `pd.to_datetime()` 사용법**](#1-pdto_datetime-사용법)
  - [**기본 사용법**](#기본-사용법)
  - [**DataFrame의 컬럼을 `datetime`으로 변환**](#dataframe의-컬럼을-datetime으로-변환)
- [**2. 날짜 포맷 지정하기**](#2-날짜-포맷-지정하기)
- [**3. `errors` 인자 사용**](#3-errors-인자-사용)
- [**4. `datetime` 타입으로 변환된 후 활용 예시**](#4-datetime-타입으로-변환된-후-활용-예시)
- [**5. `astype()`을 이용한 데이터 타입 변경**](#5-astype을-이용한-데이터-타입-변경)
  - [**요약**](#요약)

### **Pandas에서 날짜 데이터 타입으로 변경하는 방법**

Pandas에서 날짜형 데이터를 다루기 위해서는 `datetime` 데이터 타입으로 변환해야 한다. `datetime` 데이터 타입으로 변환하면 날짜와 시간 관련 연산을 쉽게 할 수 있다. 이를 위해 주로 `pd.to_datetime()` 함수를 사용한다.

## **1. `pd.to_datetime()` 사용법**

### **기본 사용법**

`pd.to_datetime()` 함수는 문자열 형식으로 된 날짜를 `datetime` 객체로 변환한다.

```python
import pandas as pd

# 문자열로 된 날짜
date_str = '2022-10-17'

# 문자열을 datetime으로 변환
date = pd.to_datetime(date_str)

print(date)

```

출력:

```
2022-10-17 00:00:00

```

### **DataFrame의 컬럼을 `datetime`으로 변환**

`DataFrame`에서 날짜가 문자열로 되어 있는 컬럼을 `datetime` 데이터 타입으로 변환할 때 사용한다.

```python
# DataFrame 생성
df = pd.DataFrame({
    'order_time': ['2022-10-17', '2022-10-18', '2022-10-19']
})

# order_time 컬럼을 datetime으로 변환
df['order_time'] = pd.to_datetime(df['order_time'])

print(df)

```

출력:

```
   order_time
0  2022-10-17
1  2022-10-18
2  2022-10-19

```

- `order_time` 컬럼이 `datetime64` 데이터 타입으로 변환된다.

---

## **2. 날짜 포맷 지정하기**

문자열에서 날짜를 변환할 때, 날짜 형식이 명확한 경우에는 `format` 인자를 사용하여 더 빠르게 변환할 수 있다. `format` 인자는 날짜 형식을 지정하는 데 사용된다.

```python
# 'YYYY-MM-DD' 형식으로 되어 있는 문자열을 변환
date_str = '2022-10-17'
date = pd.to_datetime(date_str, format='%Y-%m-%d')

print(date)

```

출력:

```
2022-10-17 00:00:00

```

- `%Y`는 4자리 연도, `%m`은 2자리 월, `%d`는 2자리 일을 나타낸다.

---

## **3. `errors` 인자 사용**

`errors` 인자는 날짜 변환 시 오류가 발생할 때 어떻게 처리할지 지정하는 데 사용된다.

- `'raise'` (기본값): 오류가 발생하면 예외를 발생시킨다.
- `'coerce'`: 변환할 수 없는 값은 `NaT`(Not a Time)으로 처리한다.
- `'ignore'`: 오류가 발생해도 원본 값을 그대로 반환한다.

```python
# 잘못된 날짜 형식이 있을 경우 'coerce'를 사용하여 NaT로 처리
df = pd.DataFrame({
    'order_time': ['2022-10-17', 'invalid_date', '2022-10-19']
})

df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')

print(df)

```

출력:

```
   order_time
0  2022-10-17
1         NaT
2  2022-10-19

```

- `invalid_date`는 `NaT`로 처리되었다.

---

## **4. `datetime` 타입으로 변환된 후 활용 예시**

`datetime`으로 변환된 컬럼은 날짜와 시간 관련 기능들을 활용할 수 있다. 예를 들어, 날짜에서 년도, 월, 일, 요일 등을 추출할 수 있다.

```python
df = pd.DataFrame({
    'order_time': ['2022-10-17', '2022-10-18', '2022-10-19']
})

df['order_time'] = pd.to_datetime(df['order_time'])

# 년도, 월, 일, 요일 추출
df['year'] = df['order_time'].dt.year
df['month'] = df['order_time'].dt.month
df['day'] = df['order_time'].dt.day
df['weekday'] = df['order_time'].dt.weekday

print(df)

```

출력:

```
   order_time  year  month  day  weekday
0  2022-10-17  2022     10   17        6
1  2022-10-18  2022     10   18        0
2  2022-10-19  2022     10   19        1

```

- `.dt`를 사용하여 날짜와 관련된 다양한 정보를 추출할 수 있다.
- `.weekday`는 월요일부터 시작하여 0에서 6까지 값을 가진다.

---

## **5. `astype()`을 이용한 데이터 타입 변경**

날짜형으로 변환된 데이터를 다시 문자열로 변경할 수 있다. `astype()`을 사용하여 `datetime`을 다른 형식으로 변환할 수 있다.

```python
# datetime을 문자열로 변환
df['order_time_str'] = df['order_time'].astype(str)

print(df)

```

출력:

```
   order_time order_time_str
0  2022-10-17      2022-10-17
1  2022-10-18      2022-10-18
2  2022-10-19      2022-10-19

```

---

### **요약**

- `pd.to_datetime()`을 사용하면 날짜 데이터를 `datetime` 타입으로 변환할 수 있다.
- `format` 인자를 사용하여 날짜 형식을 명확하게 지정할 수 있다.
- `errors` 인자를 사용하여 오류 처리 방식을 설정할 수 있다.
- 변환된 날짜에서 년, 월, 일 등의 정보를 `.dt` 속성을 통해 추출할 수 있다.
