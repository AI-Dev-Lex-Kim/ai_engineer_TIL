- [**데이터 입출력 및 생성**](#데이터-입출력-및-생성)
- [**인덱싱 및 슬라이싱**](#인덱싱-및-슬라이싱)
- [**데이터 수정 및 추가**](#데이터-수정-및-추가)
- [**통계 및 요약 함수**](#통계-및-요약-함수)
- [**데이터 시각화**](#데이터-시각화)
- [**데이터 타입 및 문자열 처리**](#데이터-타입-및-문자열-처리)
- [**데이터 형 변환 및 날짜/시간 처리**](#데이터-형-변환-및-날짜시간-처리)
- [**데이터 결합 및 그룹화**](#데이터-결합-및-그룹화)
- [**시리즈 고유값 및 개수 확인**](#시리즈-고유값-및-개수-확인)
- [**데이터 인덱싱 및 쿼리**](#데이터-인덱싱-및-쿼리)
- [**행 및 컬럼 삭제, 이름 변경**](#행-및-컬럼-삭제-이름-변경)
- [**범주형 데이터 처리 및 함수 적용**](#범주형-데이터-처리-및-함수-적용)
- [**기타 DataFrame 기본 속성 및 함수**](#기타-dataframe-기본-속성-및-함수)

### <mark>**데이터 입출력 및 생성**</mark>

- <mark>**pd.read_csv()**</mark> → CSV 파일 읽어 DataFrame 반환 (옵션: header, index_col, parse_dates 등 사용)
- <mark>**pd.DataFrame()**</mark> → 딕셔너리 등 다양한 데이터를 기반으로 DataFrame 생성

### <mark>**인덱싱 및 슬라이싱**</mark>

- <mark>**df.iloc[]**</mark> → 정수 기반 인덱싱; 단일 요소 선택 시 Series, 다중 선택 시 DataFrame 반환
- <mark>**df.loc[]**</mark> → 라벨 기반 인덱싱; 단일 행은 Series, 다중 선택은 DataFrame 반환
- <mark>**df[] 인덱싱 및 불린 인덱싱**</mark> →
  - 단일 컬럼명을 문자열로 입력하면 Series
  - 리스트 입력 시 DataFrame 반환
  - 불린 조건 선택 시 조건에 맞는 행을 포함한 DataFrame 반환

### <mark>**데이터 수정 및 추가**</mark>

- <mark>**행 값 변경**</mark> → `df.loc['row_label'] = [...]`로 특정 행의 값을 수정
- <mark>**컬럼 값 변경**</mark> → `df['컬럼'] = [...]` 또는 단일 값 할당으로 컬럼 전체 변경
- <mark>**행/컬럼 추가**</mark> → `df.loc['new_row'] = [...]`와 `df['새컬럼'] = [...]`로 추가
- <mark>**마스킹을 통한 조건부 할당**</mark> → `df.loc[조건, '컬럼'] = 값`으로 조건에 맞는 값 수정

### <mark>**통계 및 요약 함수**</mark>

- <mark>**df.describe()**</mark> → 수치형 데이터의 기본 통계정보(평균, 표준편차 등) 반환
- <mark>**df.quantile()**</mark> → 지정 분위수 값 반환
- <mark>**df.var()**</mark> → 각 컬럼의 분산 값을 Series로 반환
- <mark>**df.std()**</mark> → 각 컬럼의 표준편차 값을 Series로 반환
- <mark>**누적 연산 (cumsum(), cumprod())**</mark> → 누적 합과 누적 곱 계산하여 Series 반환

### <mark>**데이터 시각화**</mark>

- <mark>**df.plot()**</mark> → DataFrame 데이터를 다양한 그래프로 시각화하며 matplotlib Axes 객체 반환
- <mark>**seaborn 시각화 함수 (sns.barplot, sns.lineplot 등)**</mark> → DataFrame 기반의 시각화, 적절한 Axes 또는 FacetGrid 반환

### <mark>**데이터 타입 및 문자열 처리**</mark>

- <mark>**df[컬럼].astype()**</mark> → 특정 컬럼의 데이터타입 변경 후 Series 반환
- <mark>**문자열 관련 함수 (str.lower, str.upper, str.capitalize, str.split, str.strip, str.replace)**</mark> → 문자열 처리 후 새로운 Series 반환
- <mark>**round()**</mark> → 수치 데이터를 지정 소숫점 자리까지 반올림하여 Series 또는 DataFrame 반환

<mark>**결측치 및 중복 데이터 처리**</mark>

- <mark>**df.isna(), df.isna().sum(), df.isna().any()**</mark> → 결측치 여부, 개수, 존재 여부를 Boolean DataFrame/Series로 반환
- <mark>**df.dropna(), df[컬럼].fillna()**</mark> → 결측치가 있는 행 제거 및 지정 값으로 결측치 채움
- <mark>**df.duplicated(), df.drop_duplicates()**</mark> → 중복 여부 확인 및 중복 행 제거

### <mark>**데이터 형 변환 및 날짜/시간 처리**</mark>

- <mark>\*\*pd.to_datetime()와 series.dt.\*_</mark>_ → 문자열이나 Series를 datetime 타입으로 변환하고, 날짜, 월, 연도 등 정보 추출
- <mark>**pd.to_timedelta()**</mark> → 분 단위 등 값을 timedelta 타입으로 변환

### <mark>**데이터 결합 및 그룹화**</mark>

- <mark>**pd.concat()**</mark> → 여러 DataFrame을 행 또는 열 방향으로 연결
- <mark>**pd.merge()**</mark> → 공통 컬럼이나 인덱스를 기준으로 DataFrame 병합
- <mark>**left_df.join()**</mark> → 왼쪽 DataFrame의 컬럼 또는 인덱스를 기준으로 결합
- <mark>**df.groupby()와 집계 함수**</mark> → 지정 컬럼으로 그룹화 후, 크기, 평균, 최소/최대 등 집계 함수 적용
- <mark>**pd.pivot_table()**</mark> → 행, 열, 값 및 집계 함수를 사용하여 피벗 테이블 형태의 DataFrame 생성

### <mark>**시리즈 고유값 및 개수 확인**</mark>

- <mark>**series.unique()**</mark> → 시리즈 내 고유한 값들을 numpy array로 반환
- <mark>**series.value_counts()**</mark> → 각 고유 값의 빈도수를 Series로 반환

### <mark>**데이터 인덱싱 및 쿼리**</mark>

- <mark>**df[] 인덱싱**</mark> → 단일 컬럼명을 입력하면 Series, 리스트 입력 시 DataFrame 반환
- <mark>**df.query()**</mark> → 문자열 조건에 따라 DataFrame의 행을 선택하여 반환

### <mark>**행 및 컬럼 삭제, 이름 변경**</mark>

- <mark>**df.drop()**</mark> → 지정한 인덱스나 컬럼을 삭제한 DataFrame 반환
- <mark>**df.rename()**</mark> → 컬럼명 또는 인덱스의 이름을 변경한 DataFrame 반환

### <mark>**범주형 데이터 처리 및 함수 적용**</mark>

- <mark>**pd.cut()**</mark> → 연속형 데이터를 구간별로 나누어 범주형 Series 반환
- <mark>**series.apply()**</mark> → 시리즈의 각 요소에 사용자 정의 함수 적용하여 새로운 Series 생성
- <mark>**series.select()**</mark> → 조건에 맞는 시리즈 요소 선택

### <mark>**기타 DataFrame 기본 속성 및 함수**</mark>

- <mark>**df.columns**</mark> → DataFrame의 컬럼 이름들을 Index 객체로 반환
- <mark>**df.info()**</mark> → DataFrame의 행 개수, 컬럼명, 데이터타입, 결측치 등의 정보를 출력
- <mark>**df.shape**</mark> → DataFrame의 (행, 열) 개수를 튜플로 반환
