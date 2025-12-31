- [기초통계와 데이터 시각화(2) - 데이터 그래프 종류 및 시각화 with searborn](#기초통계와-데이터-시각화2---데이터-그래프-종류-및-시각화-with-searborn)
  - [데이터 그래프](#데이터-그래프)
    - [**Boxplot (박스플롯)**](#boxplot-박스플롯)
    - [**Stripplot (스트립플롯)**](#stripplot-스트립플롯)
    - [Swarmplot (스웜플롯)](#swarmplot-스웜플롯)
    - [Swarmplot 예제 코드](#swarmplot-예제-코드)
    - [Violinplot (바이올린 플롯)](#violinplot-바이올린-플롯)
    - [Regplot (회귀선 플롯)](#regplot-회귀선-플롯)
  - [상관관계 시각화](#상관관계-시각화)
    - [회귀선 + 산점도](#회귀선--산점도)
    - [히트맵](#히트맵)

<br>

# 기초통계와 데이터 시각화(2) - 데이터 그래프 종류 및 시각화 with searborn

<mark>**Seaborn**</mark>은 <mark>**Python의 데이터 시각화 라이브러리**</mark>로, <mark>**Matplotlib**</mark> 기반에서 더 세련되고 통계적인 그래프를 쉽게 만들 수 있도록 도와준다.

<br>

<mark>**Matplotlib보다 직관적인 인터페이스**</mark>

<mark>**Pandas DataFrame과 연동이 쉬움**</mark>

<mark>**기본적으로 세련된 스타일 제공**</mark>

<mark>**다양한 통계 그래프 지원**</mark> (히트맵, 박스플롯, 바이올린 플롯 등)

<br>

Jupyter에 기본적으로 설치가 안되어있어서 설치해주어야한다.

```python
!conda install --yes seaborn
import seaborn as sns
```

<br>

예제 코드

```python
sale_df = pd.read_csv('user.csv')

# 요일별 총 지출 분포 시각화
sns.boxplot(data=sale_df, x="day", y="total_bill", hue='sex')
plt.show()
```

위와 같이 data에 df를넣어주면 x와y축의 각각 원하는 컬럼을 넣어주면된다.

hue는 <mark>**데이터를 그룹별로 색상으로 구분**</mark> 할 때 사용하는 옵션이다.

<br>

## 데이터 그래프

### <mark>**Boxplot (박스플롯)**</mark>

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/1.png>)

<mark>**데이터의 분포와 이상치를 한눈에 보여주는 그래프**</mark>

- <mark>**중앙값(median), 사분위수(Q1, Q3), 최소/최대값, 이상치(outlier)**</mark> 등을 시각화
- <mark>**데이터의 퍼짐 정도**</mark>와 <mark>**이상치**</mark>를 쉽게 파악 가능

<br>

<mark>**예제 코드**</mark>

```python
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips)
plt.show()

```

<br>

### <mark>**Stripplot (스트립플롯)**</mark>

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/2.png>)

<mark>**각 데이터 포인트를 점으로 표현하는 산점도형 그래프**</mark>

- 개별 데이터를 확인하는 데 유용
- 같은 값을 가지는 데이터들이 겹칠 수 있음 → `jitter=True` 옵션 사용 가능

<br>

<mark>**예제 코드**</mark>

```python
sns.stripplot(x="day", y="total_bill", hue="sex", data=tips, jitter=True, dodge=True)
plt.show()

```

- `jitter=True` → 점들을 약간 흩어지게 해서 겹침 방지
- `dodge=True` → hue 값(성별)에 따라 점을 옆으로 분리

<br>

### Swarmplot (스웜플롯)

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/3.png>)

<mark>**Stripplot과 비슷하지만, 데이터가 겹치지 않도록 정렬하는 그래프**</mark>

- <mark>**각 데이터 포인트를 점으로 표현**</mark>하면서 <mark>**겹치지 않게 정렬**</mark>
- Stripplot보다 <mark>**데이터 밀도를 더 명확하게 표현**</mark> 가능
- <mark>**소규모 데이터셋**</mark>에서 개별 데이터 분포를 보기 좋음
- <mark>**큰 데이터셋에서는 성능 저하 가능**</mark> (점이 많아질수록 계산량 증가)

<br>

### Swarmplot 예제 코드

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 샘플 데이터 로드
tips = sns.load_dataset("tips")

# 요일별 총 지출 분포 시각화 + 성별 구분
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips, dodge=True)

plt.show()

```

- `hue="sex"` → 성별에 따라 색상을 다르게 표시
- `dodge=True` → 성별별 데이터를 <mark>**옆으로 나눠서 비교**</mark>

<br>

### Violinplot (바이올린 플롯)

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/4.png>)

<mark>**Boxplot + KDE(커널 밀도 추정) 그래프**</mark>

- 박스플롯처럼 데이터 분포를 보여주지만, <mark>**밀도(분포 형태)까지 표현**</mark>
- 데이터가 어느 값에서 많이 몰려 있는지 한눈에 확인 가능

<br>

<mark>**예제 코드**</mark>

```python
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)
plt.show()

```

- `split=True` → hue 값(성별)별로 분포를 한 그래프에서 비교

<br>

### Regplot (회귀선 플롯)

<mark>**Seaborn에서 산점도(Scatter Plot) + 회귀선(Regression Line)을 그려주는 그래프**</mark>

- 데이터의 관계를 시각적으로 보여주고, <mark>**선형 회귀 모델**</mark>을 통해 경향성을 확인할 수 있음
- `lmplot()`과 유사하지만, `regplot()`은 개별 축(`ax`)에 그릴 수 있음

<br>

Regplot 예제코드

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 샘플 데이터 로드
tips = sns.load_dataset("tips")

# 총 지출(total_bill)과 팁(tip) 간의 관계 + 회귀선
sns.regplot(x="total_bill", y="tip", data=tips)

plt.show()

```

<br>

## 상관관계 시각화

각 데이터 별로 상관관계를 알고싶을때 사용할 수 있는 대표적인 그래프로 <mark>**산점도가있다.**</mark>

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/5.png>)

위와같이 점들이 잘 뭉쳐있으면 상관관계가 높은것이고, 흩어져있으면 상관관계가 낮다는 의미이다.

상관관계를 구체적인 수치로 표현할수있다.

상관관계 > 0 이라면 어떤 값이 커질때 다른값도 함께 커진것을 뜻한다.

상관관계 < 0 이라면 어떤 값이 작아질때 다른값도 함께 작아진다는 것을 뜻한다.

<br>

상관관계 점수가 클수록 관계강도가 더 높다는 뜻이다.

```python
sns.scatterplot(data=wine_df, x='price',y='points')
```

<br>

### 회귀선 + 산점도

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/6.png>)

회귀선은 두 값의 상관관계를 요약해서 표현하는것이다.

```python
sns.regplot(data=wine_df, x='price',y='points')
```

<br>

상관관계를 데이터해서 볼수도 있다

```python
bike_df.corr(numberic_only=True)
```

<br>

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/7.png>)

- `numberic_only=True` 을 통해 숫자 데이터 컬럼만 표시해준다.

<br>

### 히트맵

<mark>**행렬 형태의 데이터를 색상으로 표현하는 그래프**</mark>

- 데이터 값이 클수록 진한 색상, 작을수록 연한 색상으로 표시됨
- <mark>**상관관계(correlation), 빈도수, 분포 분석 등에 유용**</mark>

```python
sns.heatmap(bike_df.corr(), annot=True) # 상관계수 값 표시
```

![image.png](<../images/AI%20엔지니어%20기초%20지식/기초통계와%20데이터%20시각화(2)/8.png>)
