- [데이터 입출력 및 생성](#데이터-입출력-및-생성)
  - [pd.read\_csv()](#pdread_csv)
  - [pd.DataFrame()](#pddataframe)
  - [**df.head(행\_개수)**](#dfhead행_개수)
  - [**df.tail(행\_개수)**](#dftail행_개수)
  - [**df.dtypes**](#dfdtypes)
- [인덱싱 및 슬라이싱](#인덱싱-및-슬라이싱)
  - [df.iloc\[\]](#dfiloc)
  - [df.loc\[\]](#dfloc)
  - [df\[\] 인덱싱 및 불린 인덱싱](#df-인덱싱-및-불린-인덱싱)
- [데이터 수정 및 추가](#데이터-수정-및-추가)
  - [행 값 변경](#행-값-변경)
  - [컬럼 값 변경](#컬럼-값-변경)
  - [행/컬럼 추가](#행컬럼-추가)
  - [df.quantile()](#dfquantile)
  - [df.var()](#dfvar)
  - [df.std()](#dfstd)
  - [누적 연산: cumsum(), cumprod()](#누적-연산-cumsum-cumprod)
  - [**df.corr()**](#dfcorr)
- [데이터 시각화](#데이터-시각화)
  - [df.plot()](#dfplot)
  - [seaborn 시각화 함수 (sns.barplot, sns.lineplot 등)](#seaborn-시각화-함수-snsbarplot-snslineplot-등)
- [데이터 타입 및 문자열 처리](#데이터-타입-및-문자열-처리)
  - [df\[컬럼\].astype()](#df컬럼astype)
  - [문자열 관련 함수 (str.lower, str.upper, str.capitalize, str.split, str.strip, str.replace)](#문자열-관련-함수-strlower-strupper-strcapitalize-strsplit-strstrip-strreplace)
  - [round()](#round)
- [결측치 및 중복 데이터 처리](#결측치-및-중복-데이터-처리)
  - [df.isna(), df.isna().sum(), df.isna().any()](#dfisna-dfisnasum-dfisnaany)
  - [df.dropna(), df\[컬럼\].fillna()](#dfdropna-df컬럼fillna)
  - [df.duplicated()](#dfduplicated)
  - [df.drop\_duplicates()](#dfdrop_duplicates)
- [날짜/시간 처리](#날짜시간-처리)
  - [pd.to\_datetime()](#pdto_datetime)
  - [series.dt.\*](#seriesdt)
  - [pd.to\_timedelta(series, unit=날짜)](#pdto_timedeltaseries-unit날짜)
  - [df.loc\[날짜\]](#dfloc날짜)
- [데이터 결합 및 그룹화](#데이터-결합-및-그룹화)
  - [pd.concat(\[left\_df1, right\_df2\] ,ignore\_index=True, axis=1)](#pdconcatleft_df1-right_df2-ignore_indextrue-axis1)
  - [pd.merge(left\_df, right\_df, on, how, suffix, left\_on, right\_on, left\_index, right\_index)](#pdmergeleft_df-right_df-on-how-suffix-left_on-right_on-left_index-right_index)
  - [left\_df.join(right\_df, on, lsuffix, rsuffix, how,)](#left_dfjoinright_df-on-lsuffix-rsuffix-how)
  - [df.groupby(그룹화할 열)](#dfgroupby그룹화할-열)
  - [df.groupby(그룹화할 열).size()](#dfgroupby그룹화할-열size)
  - [df.groupby(그룹화할 열).count()](#dfgroupby그룹화할-열count)
  - [df.groupby(그룹화할 열).min(numeric\_only=True)](#dfgroupby그룹화할-열minnumeric_onlytrue)
  - [df.groupby(그룹화할 열).max(),mean(),sum()](#dfgroupby그룹화할-열maxmeansum)
  - [df.groupby(그룹화할 열)\[집계할 열\]](#dfgroupby그룹화할-열집계할-열)
  - [df.groupby(\[그룹화할 열,그룹화할 열\])](#dfgroupby그룹화할-열그룹화할-열)
  - [df.groupby(그룹화할 열)\[집계할 열\].agg(min | max | mean | …)](#dfgroupby그룹화할-열집계할-열aggmin--max--mean--)
  - [df.groupby(그룹화할 열)\[집계할 열\].agg(\[{ ‘score’ : ‘mean’, ‘runtime’: \[’min’, ‘max’\]})](#dfgroupby그룹화할-열집계할-열agg-score--mean-runtime-min-max)
  - [pd.pivot\_table(df, values, index, columns)](#pdpivot_tabledf-values-index-columns)
- [시리즈 고유값 및 개수 확인](#시리즈-고유값-및-개수-확인)
  - [series.unique()](#seriesunique)
  - [series.value\_counts(dropna, normalize)](#seriesvalue_countsdropna-normalize)
- [데이터 인덱싱 및 쿼리](#데이터-인덱싱-및-쿼리)
  - [df\[\] 인덱싱](#df-인덱싱)
  - [df.query(비교연산자)](#dfquery비교연산자)
- [행 및 컬럼 삭제, 이름 변경](#행-및-컬럼-삭제-이름-변경)
  - [df.drop(value)](#dfdropvalue)
  - [df.rename()](#dfrename)
- [범주형 데이터 처리 및 함수 적용](#범주형-데이터-처리-및-함수-적용)
  - [pd.cut(series, bins, right, labels)](#pdcutseries-bins-right-labels)
  - [series.apply()](#seriesapply)
  - [series.select()](#seriesselect)
- [기타 DataFrame 기본 속성 및 함수](#기타-dataframe-기본-속성-및-함수)
  - [df.sort\_values(by='컬럼명', ascending=True)](#dfsort_valuesby컬럼명-ascendingtrue)
  - [df.columns](#dfcolumns)
  - [df.info()](#dfinfo)
  - [df.shape](#dfshape)

## 데이터 입출력 및 생성

### pd.read_csv()

CSV 파일을 읽어 DataFrame으로 반환한다. (예: header, index_col, parse_dates 옵션 사용한다.)

- header=None
- index_col
  - 숫자, 문자
- parse_dates=[원하는 컬럼들 넣어줌]
  - 해당 컬럼들은 datetime 데이터타입으로 변경된체 불러옴

```python
df = pd.read_csv('data.csv', header=0, index_col='id', parse_dates=['date'])
```

### pd.DataFrame()

딕셔너리 등 다양한 데이터를 DataFrame으로 생성하여 반환한다.

```python
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(df)

```

### **df.head(행\_개수)**

DataFrame의 상위 일부 행 반환

- 기본값: `행_개수=5`

### **df.tail(행\_개수)**

DataFrame의 하위 일부 행 반환

- 기본값: `행_개수=5`

### **df.dtypes**

각 컬럼의 데이터타입 정보를 반환

## 인덱싱 및 슬라이싱

### df.iloc[]

- 정수 기반 인덱싱으로 요소를 선택하며, 단일 요소는 Series, 다중 선택은 DataFrame을 반환한다.
  예제:

결과: 첫 예제는 단일 값, 두 번째 예제는 DataFrame 출력이다.

    ```python
    element = df.iloc[0, 1]
    subset = df.iloc[[0, 1], [0, 1]]
    print(element)
    print(subset)

    ```

### df.loc[]

- 라벨 기반 인덱싱으로 요소를 선택하며, 단일 행은 Series, 다중 선택은 DataFrame을 반환한다.
  예제:

결과: 지정한 라벨과 조건에 해당하는 Series 또는 DataFrame 출력이다.

    ```python
    row_a = df.loc['a']
    subset = df.loc[df['A'] < 3, 'B']
    print(row_a)
    print(subset)

    ```

### df[] 인덱싱 및 불린 인덱싱

- 단일 컬럼명을 문자열로 입력하면 Series, 리스트로 입력하면 DataFrame을 반환하며, 불린 조건 선택 시 조건에 맞는 행을 포함한 DataFrame을 반환한다.
  예제:

결과: 첫 번째는 Series, 두 번째는 DataFrame, 세 번째는 조건에 맞는 DataFrame 출력이다.

    ```python
    col_series = df['A']
    col_df = df[['A']]
    subset = df[df['A'] < 2]

    ```

## 데이터 수정 및 추가

### 행 값 변경

- df.loc['row_label'] = [...]를 통해 특정 행의 값을 새로운 값으로 수정한다.
  예제:

결과: 인덱스 'a'의 값이 수정된 Series 출력이다.

    ```python
    df.loc['a'] = [10, 20]
    print(df.loc['a'])

    ```

### 컬럼 값 변경

- df['컬럼'] = [...] 또는 단일 값 할당(df['컬럼'] = 1)을 통해 컬럼의 값을 수정 또는 전체 값을 변경한다.
  예제:

결과: 'A' 컬럼의 값이 변경된 Series 출력이다.

    ```python
    df['A'] = [100, 200, 300]
    print(df['A'])

    ```

### 행/컬럼 추가

- df.loc['new_row'] = [...]로 행,
- df['새컬럼'] = [...]로 컬럼을 추가한다.
  예제
  ```python
  df.loc['new_row'] = [7, 8]
  df['C'] = [9, 10, 11]

      ```

### 마스킹을 통한 조건부 할당

- df.loc[조건, '컬럼'] = 값으로 조건에 따른 특정 컬럼 값을 변경한다.
  예제:

결과: 조건에 해당하는 행의 'B' 컬럼 값이 0으로 변경된 DataFrame 출력이다.

    ```python
    df.loc[df['A'] < 200, 'B'] = 0
    ```

## 인덱스 설정 및 정렬

### **df.set_index('컬럼명')**

지정한 컬럼을 인덱스로 설정

### **df.reset_index()**

기존 인덱스를 열로 이동시키고, 기본 정수 인덱스로 리셋한 DataFrame 반환

### df.sort_index()

인덱스를 오름차순으로 정렬해줌

## 통계 및 요약 함수

### df.describe()

수치형 데이터의 기본 통계정보(평균, 표준편차 등)를 DataFrame으로 반환한다.

- include=’all’ → 숫자 값 뿐만 아니라 다른 값들도 통계정보가 나온다.

```python
summary = df.describe()
```

### df.quantile()

- 지정한 분위수의 값을 Series 또는 DataFrame으로 반환한다.
  ```python
  q1 = df.quantile(0.25)
  q3 = df.quantile(0.75)
  ```

### df.var()

- 각 컬럼의 분산 값을 Series로 반환한다.
  ```python
  variance = df.var()
  ```

### df.std()

- 각 컬럼의 표준편차 값을 Series로 반환한다.
  ```python
  std_dev = df.std()
  ```

### 누적 연산: cumsum(), cumprod()

- cumsum()은 누적 합을, cumprod()는 누적 곱을 Series로 반환한다.
  예제:

결과: 'A' 컬럼의 누적합과 누적곱을 포함한 DataFrame 출력이다.

    ```python
    df['누적합'] = df['A'].cumsum()
    df['누적곱'] = df['A'].cumprod()
    print(df[['A', '누적합', '누적곱']])

    ```

### **df.corr()**

각 컬럼 간의 상관관계 계산 (numeric한 데이터만 고려)

- **옵션:**
  - `numeric_only=True` 또는 `numeric_only=False` → 숫자형 데이터만 상관계산에 포함할지 여부 설정

## 데이터 시각화

### df.plot()

DataFrame 데이터를 다양한 종류의 그래프로 시각화하며, matplotlib Axes 객체를 반환한다.

- 파라미터
- x,
- y,
- kind,
- labels
- kind=bar, scatter, pie
- bins
- bm_method

```python
ax = df.plot(x='A', y='B', kind='scatter')
```

### seaborn 시각화 함수 (sns.barplot, sns.lineplot 등)

DataFrame 데이터를 그래프로 출력한다.

파라미터

- data=데이터프레임
- x
- y
- errorbar
- hue
- multiplestack

```python
import seaborn as sns
ax = sns.barplot(data=df, x='A', y='B', hue='C')
```

## 데이터 타입 및 문자열 처리

### df[컬럼].astype()

특정 컬럼의 데이터타입을 변경한 Series를 반환한다.
'A' 컬럼의 데이터타입이 float로 변경된 정보 출력이다.

```python
df['A'] = df['A'].astype(float)
```

### 문자열 관련 함수 (str.lower, str.upper, str.capitalize, str.split, str.strip, str.replace)

문자열 처리 관련 함수로, 각각 문자열을 소문자, 대문자, 첫글자 대문자로 변환하거나, 분할, 공백 제거, 문자열 치환을 수행한 Series를 반환한다.
예제:

결과: 'A' 컬럼과 소문자로 변환된 'A_lower' 컬럼 출력이다.

```python
df['A_lower'] = df['A'].astype(str).str.lower()
print(df[['A', 'A_lower']])

```

### round()

수치 데이터를 지정 소숫점 자리까지 반올림한 Series 또는 DataFrame을 반환한다.
round(series, 2) → 소숫점 둘째자리까지 반올림함

```python
df['A_round'] = round(df['A'], 2)
```

## 결측치 및 중복 데이터 처리

### df.isna(), df.isna().sum(), df.isna().any()

df.isna() → 데이터에 결측값이 있으면 True, 없으면 False

df.isna().sum()

→ 각 컬럼별 결측값 개수 알려줌

- True는 1, False는 0으로 계산함

df.isna().any(axis=1)

- row을 각각 확인하면서 True가 하나라도 있으면 True, 없으면 False가 나옴

```python
print(df.isna())
print(df.isna().sum())
print(df.isna().any(axis=1))

```

### df.dropna(), df[컬럼].fillna()

df.dropna()

결측값이 있는 row들을 모두 삭제한다.

df[컬럼명].fillna(원하는 값)

해당 컬럼에 결측값 대신에 원하는 값으로 채워짐

```python
df_clean = df.dropna()
df['A_filled'] = df['A'].fillna(0)
```

### df.duplicated()

- row에 중복값이 하나라도 있으면 True, 없으면 False
- subset=컬럼명
  - 해당 컬럼에 중복값이 있는 row면 True, 없으면 False
- keep=first | last | False
  - 데이터프레임을 보여줄때 첫번째 중복된 데이터는 보여주지않음.
  - first는 중복된 데이터가 있는 row들중 첫번째 위치해 있는 row는 중복된 값이 아니라고 해서 안나옴
  - last는 중복된 데이터가 있는 row들중 마지막에 위치한 row는 중복된 값이 아니라고 해서 안나옴
  - False는 중복된 데이터가 있는 모든 row를 보여줌
  - 즉 keep은 중복된 row중 해당 위치의 row는 남긴다는 뜻, 표시 안함

df[df.duplicated()]

- 중복값이 있는 데이터들만 보여줌

df.duplicated().sum()

- 중복된 row 개수 알려줌

### df.drop_duplicates()

중복된 row을 제거함. 하나의 값이라도 같으면 삭제함

- subset=컬럼명
  - 해당 컬럼만 중복되어야 삭제함
- keep=’first’ | ‘last’
  - 중복된 row들중 첫번째, 마지막만 삭제하지 않음

## 날짜/시간 처리

### pd.to_datetime()

datetime 데이터타입으로 변경해줌

### series.dt.\*

series.dt.day

series.dt.month

series.dt.year

series.dt.dayofweek → 요일별로 숫자로 표현함 월요일0, 일요일 6

```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
```

### pd.to_timedelta(series, unit=날짜)

series를 timedelta 데이터 타입으로 바꾸어줌
unit=’D’ | ‘min’ | ‘S’ | ‘Y’

- 일, 분,초 연으로 데이터를 바꾸어줌
- ex) 10 → 10분 or 10초 or 10년

```python
df['duration'] = pd.to_timedelta(df['duration_minutes'], unit='min')
```

### df.loc[날짜]

인덱스에 그 날짜에 해당하는 row들만 불러옴

- ex)
  - df.loc[’2015’]
  - df.loc[’2015-06-10’]
  - df.loc[’2015’:’2017’] → 2017도 포함됨

## 데이터 결합 및 그룹화

### pd.concat([left_df1, right_df2] ,ignore_index=True, axis=1)

두 데이터를 합쳐준다. defualt()

left_df1데이터프레임 밑에 right_df2가 들어간다.

- ignore_index=True | False
  - 합쳐줄때 각 데이터프레임의 인덱스를 합쳐서 붙여주지 못하게한다.
- axis=1
  - left_df1를 right_df2 오른쪽에 붙여준다.

```python
df_concat = pd.concat([df1, df2], axis=1, ignore_index=True)
```

### pd.merge(left_df, right_df, on, how, suffix, left_on, right_on, left_index, right_index)

left_df을 right_df에 합쳐준다.(인덱스를 기준으로)

- on=컬럼명
  - 합쳐서 새롭게 만든 데이터프레임의 인덱스를 정해준다.
  - 당연히 고유한 값이여야한다
- how=’inner’(default) | ’left’ | ‘right’ | ‘outter’
  - 두 데이터에 모두 있는 값만 남긴다.
  - inner → 두 데이터에 인덱스가 공통된 값만 나온다.
  - left → 왼쪽에 있는 데이터의 인덱스의 row들을 모두 넣어준다. 이후 오른쪽 데이터중 인덱스 값이 같으면, 오른쪽의 col 값이 들어간다. 없으면 결측값이 된다.
  - right → 오른쪽에 있는 인덱스의 row 값들을 모두 넣어준다. 왼쪽의 데이터의 인덱스중 공통된 인덱스가 있으면 오른쪽의 col값이 들어간다. 없으면 결측값이 된다.
  - outter → 두 데이터의 인덱스가 모두 합쳐져서 들어간다.
- suffixes=(’왼쪽 추가 네이밍’, ‘오른쪽 추가 네이밍’)
  - 만약 두 데이터의 공통된 컬럼명이 있으면 각각 네이밍을 붙여준다.
- left_on=인덱스로 사용할 컬럼명, right_on=인덱스로 사용할 컬럼명
  - 만약 두 데이터의 값은 같지만 key값이 될 컬럼명이 다르다면, left_on, right_on에 각자 key로 설정할 컬럼명을 넣어주면된다.
- left_index, right_index = True | False
  - `left_index`라는 값을 `True`로 설정하면 왼쪽 데이터에 있는 인덱스를 키 값으로 사용하고, `right_index`라는 값을 `True`로 설정하면 오른쪽 데이터에 있는 인덱스를 키 값으로 사용합니다.
  - 둘중 하나만 True로 사용해도 된다. 예를 들어 `left_index`만 `True`로 설정하면 왼쪽 데이터만 인덱스를 키 값으로 사용하고, 오른쪽 데이터는 특정 컬럼을 키 값으로 사용하게 되는데요. 이때, 반드시 `right_on` 옵션을 함께 사용해서 어떤 컬럼을 키 값으로 할 건지 정해 줘야 합니다. 반대로 `right_index`만 `True`로 설정할 땐, `left_on` 옵션을 함께 사용해야 하고요.

```python
df_merge = pd.merge(df1, df2, on='id', how='inner')
```

![image.png](attachment:e04baf9d-6650-4587-8ca8-e54c7e33402f:image.png)

![image.png](attachment:fd965834-e3c6-4f69-afe3-cdcafd9cdf7b:image.png)

![image.png](attachment:a05865c4-f43c-4d9e-b879-a2fdf43bcd1d:image.png)

### left_df.join(right_df, on, lsuffix, rsuffix, how,)

- left_df와 right_df의 데이터를 합쳐준다.
- lsuffix, rsuffix → 공통된 컬럼명을 각각 추가해서 다르게 만들어준다.
- on= 왼쪽 데이터에서 키 값으로 사용할 컬럼명
  - ex)
  ```python
  employee_df.join(survey_df, on='id', rsuffix='_x')
  ```
  - `employee_df` 데이터의 `id` 컬럼과 `survey` 데이터의 인덱스를 기준으로 조인 연산을 수행합니다.
  - 이때 주의할 점은 왼쪽 데이터만 인덱스와 컬럼 중에서 자유롭게 키 값을 선택할 수 있고, 오른쪽 데이터는 인덱스만 키 값으로 사용할 수 있다는 것입니다.
- how=`’inner`’ | `left` | `right` | `ouuter`
  - inner → 두 데이터에 인덱스가 공통된 값만 나온다.
  - left → 왼쪽에 있는 데이터의 인덱스의 row들을 모두 넣어준다. 이후 오른쪽 데이터중 인덱스 값이 같으면, 오른쪽의 col 값이 들어간다. 없으면 결측값이 된다.
  - right → 오른쪽에 있는 인덱스의 row 값들을 모두 넣어준다. 왼쪽의 데이터의 인덱스중 공통된 인덱스가 있으면 오른쪽의 col값이 들어간다. 없으면 결측값이 된다.
  - outter → 두 데이터의 인덱스가 모두 합쳐져서 들어간다.

### df.groupby(그룹화할 열)

원하는 컬럼의 데이터를 인덱스로 만들어서 통계정보를 알수있다.

열(예: '매출', '점수')의 합계(`sum()`), 평균(`mean()`) 등을 계산한다.
그룹별 행 개수, 평균, 최소/최대/평균 값이 출력된다.

```python
group_size = df.groupby('지역').size()
group_mean = df.groupby('지역')['매출'].mean()
group_stats = df.groupby('지역')['매출'].agg(['min', 'max', 'mean'])
```

### df.groupby(그룹화할 열).size()

각 그룹별로 row가 몇개인지 나온다.

### df.groupby(그룹화할 열).count()

지정한 컬럼을 기준으로 데이터를 그룹화한 뒤, 각 그룹에서 **결측값이 아닌 데이터 개수**를 세어 반환한다.

### df.groupby(그룹화할 열).min(numeric_only=True)

각 그룹에서 **숫자형 또는 정렬 가능한 값(예: 날짜, 문자열 등)의 최소값**을 반환한다.

- `numeric_only=True`
  - 숫자형만 나온다.

### df.groupby(그룹화할 열).max(),mean(),sum()

### df.groupby(그룹화할 열)[집계할 열]

집계할 열 하나만 통계정보를 얻을수 있다.

### df.groupby([그룹화할 열,그룹화할 열])

멀티인덱스를 만들어준다.

```python
ex)
df.groupby(['지역', '상품']).loc[('서울', '전자')]
df.groupby(['지역', '상품']).loc[(['서울','대전','대구'], '전자'),:]

['지역', '상품']
데이터를 그룹화할 기준이 되는 열들을 지정한다.
예를 들어, '지역'과 '상품'을 기준으로 그룹화한다.

(['서울', '대전', '대구'], '전자')
그룹화 후 선택할 특정 그룹을 지정하는 값이다.
첫 번째 요소는 '지역'에 해당하며, ['서울', '대전', '대구'] 중 하나의 값을 의미한다.
두 번째 요소는 '상품'에 해당하며, '전자'라는 특정 값을 의미한다.

':'
조건에 맞는 그룹의 모든 열을 선택한다.
```

### df.groupby(그룹화할 열)[집계할 열].agg(min | max | mean | …)

- sum → 집계할 열의 합계를 구해준다
- max → 집계할 열의 곱를 구해준다

### df.groupby(그룹화할 열)[집계할 열].agg([{ ‘score’ : ‘mean’, ‘runtime’: [’min’, ‘max’]})

- score는 평균을
- runtime은 최솟값과 최댓값을 구해준다.

### pd.pivot_table(df, values, index, columns)

지정한 행과 열 기준에 따라 데이터를 재배열해, 쉽게 분석할 수 있는 요약표를 생성한다.

- **df**: 원본 데이터가 저장된 데이터프레임이다
- **values**: value에 있는 데이터를 요약한다.
- **index**: 행 방향으로 그룹화할 기준 열을 지정한다.
- **columns**: 열 방향으로 그룹화할 기준 열을 지정한다.

```python
pivot = pd.pivot_table(df, values='매출', index='지역', columns='상품', aggfunc='mean')
```

```python
pivot_table = pd.pivot_table(
    df,
    values=['매출', '수량'],          # 대상 값: 매출과 수량 데이터를 요약한다.
    index=['지역', '날짜'],           # 행 그룹화: 지역과 날짜별로 데이터를 그룹화한다.
    columns='상품',                  # 열 그룹화: 상품 종류별로 데이터를 분리한다.
    aggfunc={'매출': 'sum', '수량': 'mean'}  # 집계 함수: 매출은 합계, 수량은 평균을 계산한다.
)
```

## 시리즈 고유값 및 개수 확인

### series.unique()

시리즈 값들의 종류를 배열로 가져온다.

```python
unique_vals = df['A'].unique()
```

### series.value_counts(dropna, normalize)

시리즈 내 각 고유 값의 개수를 Series로 반환한다.

옵션

- dropna=False
  - Nan 개수도 알려준다
- normalize=True
  - 값들의 종류의 비율을 알려준다

```python
counts = df['A'].value_counts(dropna=False)
print(counts)

```

## 데이터 인덱싱 및 쿼리

### df[] 인덱싱

- 단일 컬럼명을 입력하면 Series, 리스트 입력 시 DataFrame을 반환한다.
  예제:

결과: Series와 DataFrame의 타입이 각각 출력된다.

    ```python
    col_series = df['A']
    col_df = df[['A']]
    print(type(col_series), type(col_df))

    ```

### df.query(비교연산자)

ex) ‘income > 5000’

변수는 @ 써줘야함 ex) income > @income_mean

내부에 문자열있으면 겹치지 않게 다른 따옴표 쓰기

in도 사용가능 ex area in [’city’, ‘Suburb’]

다중 연산자는 괄호로 묶어주기

- ex) "(married == 'Y') & (income > @income_mean)”

```python
income_mean = df['income'].mean()
result = df.query("income > @income_mean and area in ['city', 'Suburb']")
```

## 행 및 컬럼 삭제, 이름 변경

### df.drop(value)

value 값과 같은 인덱스 row 삭제된다.

- columns=컬럼명
  - 컬럼삭제
- axis=1
  - 컬럼삭제
  - 위의 columns와 같음
  - ex df.drop(’컬럼명’ ,axis=1)
- ex)
  - df.drop(’LP1003’) → LP1003 row 삭제
  - df.drop(’income’, axis=1) → income 컬럼 삭제
    - `df.drop(columns=’income’)` = `df.drop(’income’, axis=1)` 두 코드가 같음
- 컬럼명, axis=1 | columns =컬럼명 col 삭제

```python
df_drop_row = df.drop('LP1003')
df_drop_col = df.drop('income', axis=1)
```

### df.rename()

컬럼명 또는 인덱스의 이름을 변경한 DataFrame을 반환한다.

```python
df_renamed = df.rename(columns={'old_name': 'new_name'})
```

## 범주형 데이터 처리 및 함수 적용

### pd.cut(series, bins, right, labels)

연속형 데이터를 구간별로 나누어 범주형 Series를 반환한다.

- bins=[10,20,…]
  - bins의 각 인덱스 별로의 사이 구간을 설정해 속해있는것으로 나눈다.
- right=True | False 오른쪽값을 포함한 구간인지 아닌지 나누어준다
  - ex) 10 ~20에서 20이하인지 미만인지 True면 이하
- labels=[’10s’, ’20s’, …] 각 구간별로 네이밍해준다.

```python
df['age_group'] = pd.cut(df['age'], bins=[10,20,30,40,50], right=False, labels=['10s','20s','30s','40s'])
```

### series.apply()

- 시리즈의 각 요소에 사용자 정의 함수를 적용하여 새로운 Series를 반환한다.
  예제:

결과: 함수 적용 후 그룹화된 Series 출력이다.

    ```python
    def group_age(x):
        if 10 <= x < 20:
            return '10s'
        elif 20 <= x < 30:
            return '20s'
        elif 30 <= x < 40:
            return '30s'
        elif 40 <= x < 50:
            return '40s'
        elif 50 <= x < 60:
            return '50s'
        else:
            return '60s'

    df['age_group'] = df['age'].apply(group_age)
    print(df[['age', 'age_group']])

    ```

### series.select()

- 조건에 맞는 시리즈 요소를 선택하여 반환한다(사용자 정의 함수일 수 있다).
  예제:

결과: 조건을 만족하는 값들이 출력된다.

    ```python
    def custom_select(series, condition):
        return series[condition]

    selected = custom_select(df['A'], df['A'] > 10)
    print(selected)

    ```

## 기타 DataFrame 기본 속성 및 함수

### df.sort_values(by='컬럼명', ascending=True)

지정한 컬럼을 기준으로 정렬

- **옵션:**
  - `by='컬럼명'` → 정렬 기준이 되는 컬럼 지정
  - `ascending=True` → 오름차순 정렬 (기본값)
  - `ascending=False` → 내림차순 정렬

### df.columns

DataFrame의 컬럼 이름들을 배열로 반환한다.

### df.info()

DataFrame의 행 개수, 컬럼명, 데이터타입, 결측치 등의 정보를 출력한다.
예제:

결과: DataFrame의 상세 정보가 콘솔에 출력된다.

```python
df.info()

```

### df.shape

(row, col) 개수를 튜플로 반환한다.
