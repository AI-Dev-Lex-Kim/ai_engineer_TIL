- [DataFrame 생성, 불러오기, 저장, 데이터 일부 처리](#dataframe-생성-불러오기-저장-데이터-일부-처리)
  - [컬럼명 설정](#컬럼명-설정)
- [데이터 가져오기 read\_csv()](#데이터-가져오기-read_csv)
  - [header(옵션)](#header옵션)
  - [Index\_col(옵션)](#index_col옵션)
  - [df.head()](#dfhead)
- [DataFrame 저장 to\_csv()](#dataframe-저장-to_csv)
- [DataFrmae 일부만 가져오기 pd.loc vs pd](#datafrmae-일부만-가져오기-pdloc-vs-pd)
  - [pd.loc](#pdloc)
  - [pd](#pd)
  - [DataFrame 할당](#dataframe-할당)

<br>

## DataFrame 생성, 불러오기, 저장, 데이터 일부 처리

```python
two_dimensional_list = [
    ['skirt', 10, 30000],
    ['sweater', 15, 60000],
    ['coat', 6, 95000],
    ['jeans', 11, 35000]
]
two_dimensional_array = np.array(two_dimensional_list)

list_df = pd.DataFrame(two_dimensional_list)
array_df = pd.DataFrame(two_dimensional_array)
```

`pd.DataFrame(parameter)` parameter로 array, list 둘다 와도된다.

DataFrame은 객체로 이루어짐

따라서 `pd.DataFrame[’category’]`가 가능함

<br>

항상 value 값은 배열로 만들어주어야한다.

```python
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
```

<br>

### 컬럼명 설정

```python
list_df = pd.DataFrame(two_dimensional_list, columns=['category', 'quantity', 'price'])
array_df = pd.DataFrame(two_dimensional_array, columns=['category', 'quantity', 'price'])

```

`pd.DataFrame(array, columns:컬렁명)`

columns은 배열에 원하는 순으로 써주면된다.

`index=['Product A', 'Product B']`

columns 대신 index도 가능하다

<br>

## 데이터 가져오기 read_csv()

### header(옵션)

데이터가 header가 없을 수 있다.

이럴때 names을 써주면 된다.

`names=[’mathces’, ‘golas’]`

```python
import pandas as pd
pd.read_csv('tottenham_2021.csv', index_col='product_name', header='None'. names=['matches', 'minutes', 'goals', 'assists'])
```

<br>

### Index_col(옵션)

index_col로 첫번째 **col의 네이밍을 할수있다.**

만약 **숫자 값을 넣으면 헤더에있는 인덱스 위치**가 제일 앞의 col에 온다.

ex) `index_col=0`

아래의 product_name

![image.png](../images/Pandas/DataFrame%20생성,불러오기/1.png)

<br>

`header=”None”` 을 하면 위의 header의 값이 0,1,2,3,4 이런식으로 나온다.

![image.png](../images/Pandas/DataFrame%20생성,불러오기/2.png)

`names=['matches', 'minutes', 'goals', 'assists']`

header의 이름을 넣어줄 수 있다.

<br>

![image.png](../images/Pandas/DataFrame%20생성,불러오기/3.png)

<br>

### df.head()

제일 위의 5개만 데이터로 나온다.

```python
df.head()
```

<br>

## DataFrame 저장 to_csv()

```python
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv('cows_and_goats.csv')
```

`cows_and_goats.csv`이라는 이름으로 저장하게 된다.

<br>

## DataFrmae 일부만 가져오기 pd.loc vs pd

### pd.loc

`pd.loc[’buerger’]` 을하면 row에 `buerger` 값을 가져온다.

<br>

주의

- loc을 쓸때는 **배열**로 불러와야한다.
- loc의 0번째 요소는 **row의 값**만을 가져온다.

<br>

`pd.loc[’buerger’, ‘goals’]` 을하면 row의 buger와 col의 goals를 가져온다.

<br>

### pd

`pd[’category’]` 을 하면 col의 값을 가져온다

`pd[[’category’, 'fat']]` 을하면 col의 두가지 값을 다가져온다.

`pd[’category’ : ‘fat’]`은 col값을 가져오지 않는다.

- row의 값만을 가져온다.
- `pd[’Whopper’ : ‘hambuerger’]`

<br>

주의

- pd[key] 는 **col 값**만 가져온다.

<br>

### DataFrame 할당

```python
pd.loc[pd['calories'] > 500 , 'high_calorie'] = True
```

위처럼 하면 기존에 없던 컬럼의 high_calories 가 True로 추가된다.

`pd['calories'] > 500` 인조건들만.
