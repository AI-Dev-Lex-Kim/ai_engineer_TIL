- [datetime](#datetime)
  - [datetime.now()](#datetimenow)
  - [datetime.now().timestamp()](#datetimenowtimestamp)
  - [datetime.fromtimestamp(timestamp)](#datetimefromtimestamptimestamp)
  - [`datetime` 객체 간의 뺄셈:](#datetime-객체-간의-뺄셈)
  - [`timedelta` 객체에서 차이 추출하기:](#timedelta-객체에서-차이-추출하기)

## datetime

시간 관련 모듈

<br>

### datetime.now()

<br>

현재 시간

### datetime.now().timestamp()

`timestamp()` 메소드는 **1970년 1월 1일** 이후부터 현재까지의 경과 시간을 초 단위로 반환합니다. 반환 값은 실수로, 소수점 아래 부분은 마이크로초를 나타냅니다.

```python
dt = datetime(2025, 3, 17, 14, 30, 45)
timestamp = dt.timestamp()
timestamp = 1678934400  # 예시 타임스탬프 (2025-03-17 00:00:00)
```

<br>

### datetime.fromtimestamp(timestamp)

이 메소드는 타임스탬프를 입력받아 해당하는 `datetime` 객체를 반환합니다

```python
timestamp = 1678934400

# 타임스탬프를 datetime 객체로 변환
dt = datetime.fromtimestamp(timestamp)
# 2025-03-17 00:00:00

```

<br>

### `datetime` 객체 간의 뺄셈:

`datetime` 객체를 빼면, 그 결과는 `timedelta` 객체입니다. `timedelta` 객체는 날짜와 시간 간의 차이를 나타내며, 이 객체에서 **초 단위**, **분 단위**, **시간 단위** 등을 추출할 수 있습니다.

```python
# 첫 번째 타임스탬프 (2025-03-17 14:00:00)
timestamp_1 = 1678933200
# 두 번째 타임스탬프 (2025-03-17 14:30:00)
timestamp_2 = 1678935000

# 타임스탬프를 datetime 객체로 변환
dt1 = datetime.fromtimestamp(timestamp_1)
dt2 = datetime.fromtimestamp(timestamp_2)

# 두 datetime 객체의 차이 계산
time_diff = dt2 - dt1

# 차이 결과 출력
print(f"두 datetime 객체 간의 차이는 {time_diff}입니다.")
```

<br>

### `timedelta` 객체에서 차이 추출하기:

`timedelta` 객체는 날짜, 시간, 분, 초 등을 포함하고 있으며, 이를 통해 두 날짜 간의 차이를 다양한 단위로 확인할 수 있습니다.

```python

복사편집
# 두 datetime 객체의 차이로 얻은 timedelta 객체에서 초 단위, 분 단위 추출
seconds_diff = time_diff.total_seconds()  # 초 단위 차이
minutes_diff = seconds_diff / 60  # 분 단위 차이
hours_diff = seconds_diff / 3600  # 시간 단위 차이

print(f"초 차이: {seconds_diff}초")
print(f"분 차이: {minutes_diff}분")
print(f"시간 차이: {hours_diff}시간")

```

<br>

출력:

```

복사편집
초 차이: 1800.0초
분 차이: 30.0분
시간 차이: 0.5시간

```
