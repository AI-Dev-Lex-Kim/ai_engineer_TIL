- [dataset load\_dataset()](#dataset-load_dataset)
  - [hugging face dataset load\_dataset](#hugging-face-dataset-load_dataset)
    - [**포맷 및 빌더 선택**](#포맷-및-빌더-선택)
    - [**스플릿 결정**](#스플릿-결정)
    - [**중첩구조**](#중첩구조)
    - [**멀티프로세싱(Multi Processing)**](#멀티프로세싱multi-processing)
    - [**Specify Features**](#specify-features)
    - [load nested json data](#load-nested-json-data)
    - [load\_dataset 동작원리](#load_dataset-동작원리)
    - [파라미터](#파라미터)

---

# dataset load_dataset()

## hugging face dataset load_dataset

hugging face의 load_dataset을 통해 데이터를 쉽게 불러올 수 있다.

```python
from datasets import load_dataset

dataset = load_dataset("json", data=files="my_file.json")
```

### **포맷 및 빌더 선택**

`load_dataset`은 데이터셋 폴더 안의 파일들을 살펴보고 가장한 흔한 데이터 포맷을 찾아서 그 형식으로 데이터를 불러온다. 예를들어 json 파일이 많으면 json 빌더를 선택해서 가져온다.

사용자가 `“json”`, `“csv”` 같이 넣어주어 해당 빌더를 지정할 수 있다.

<br>

### **스플릿 결정**

파일이나 폴더 이름을 보고 train data인지 test data인지 자동으로 구분한다.

예를 들어서 `train.json`, `test.json` 파일이 있으면 알아서 `train`, `test`로 스플릿 해준다. 이것도 `data_files` 파라미터에서 수동으로 지정 가능하다.

<br>

### **중첩구조**

만약 nested filed로 되어있다면, `field` 파라미터를 사용해야한다.

```python
"""
json 파일 내부 구조
{"version": "0.1.0",
 "data": [{"a": 1, "b": 2.0, "c": "foo", "d": false},
          {"a": 4, "b": -5.5, "c": null, "d": true}]
}
"""
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_file.json", field="data")

```

<br>

### **멀티프로세싱(Multi Processing)**

데이터셋의 크기가 매우 크다면(수십 GB) 데이터셋을 불러오는데 오랜시간이 걸린다.

하나의 프로세스로(워커)만 작업 한다면 여러개의 코어가 있어도 순차적으로 처리해 작업시간이 오래걸린다.

멀티 프로세싱으로 `num_proc=4`와 같이 파라미터를 지정해주면된다.

<br>

100GB라는 데이터셋의 크기가 있다면 이를 1GB로 100개 나누어 저장할 수 있다.

1GB 하나하나를 샤드(Shard)라고 부른다. `num_proc=10` 이라면 하나의 프로세스당 10개의 샤드를 담당한다.

이후 병렬적으로 처리한뒤 결과를 모두 하나로 합쳐서 데이터셋 객체를 만든다.

`data_load` 뿐만 아니라 `.map()` 에서도 사용할 수 있다.

```python
from datasets import load_dataset

imagenet = load_dataset("timm/imagenet-1k-wds", num_proc=8)
ml_librispeech_spanish = load_dataset("facebook/multilingual_librispeech", "spanish", num_proc=8)
```

<br>

### **Specify Features**

dataset 라이브러리를 사용해서 로컬 파일을 Dataset으로 만들때, 각 컬럼(key)의 데이터타입을 자동으로 추론한다. 예를 들어서 어떤 컬럼에 숫자만 있으면 `int` 타입, 글자만 있으면 `string` 타입으로 인식한다.

<br>

하지만 자동으로 추론을 해주기 때문에 항상 원하는 타입으로만 추론하지 않는다.

이럴때 컬럼의 타입을 직접 지정해주어야한다.

1. 다양한 타입 지정
   - 흔하게 label 컬럼은 로컬 파일에서 `0`, `1`, `2` 같은 숫자로 저장되어있다. 자동으로 추론했을때 `int` 타입으로 인식을 한다. 하지만 이 숫자들을 ‘부정’, ‘중립’, ‘긍정’ 같은 시멘틱 정보로 부여해야한다. 이럴때 `ClassLabel` 피처를 사용해서 숫자와 실제 클래스 이름을 매핑해줄 수 있다.
2. 타입 추론 오류 방지
   - 데이터 타입을 잘못 추론하는 경우도 있다. 예를들어서 우편번호 처럼 숫자 형태이지만 실제로는 문자열로 다루어야하는 컬럼이다. 이럴때 자동으로 `int` 타입으로 인식한다. 이런 오류를 막을 수 있다.
3. 데이터 일관성 확보
   - 여러개의 파일에 나뉘어 저장된 데이터셋을 불러올때, 모든 파일이 동일한 데이터 스키마(구조, Schema)를 갖도록 강제해서 데이터 일관성을 보장 할 수 있다.

<br>

```python
# 클래스 이름 정의
# 라벨을 0, 1, 2 에 순서대로 매핑될 실제 클래스 이름을 리스트로 만든다.
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

emotion_feature = Features({
		# text 컬럼을 string 타입으로 명시한다.
		'text' : Value('string'),

		# label 컬럼을 ClassLabel 타입으로 명시한다.
		'label' : ClassLabel(names=class_names),
})

dataset_custom = load_dataset(
    'json',
    data_dir='path/to/local/my_dataset.json', # 데이터 폴더안의 파일 가져옴
    # data_files='path/to/local/my_dataset.json' 파일 하나 가능
    features=emotion_features  # ***** 우리가 만든 features 객체를 지정 *****
)

출력
->
"""
{
	'text': Value(dtype='string', id=None),
	'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
}
"""

```

`ClassLabel`은 정수와 문자열 라벨의 매핑을 해주어주는 특별한 Feature 타입이다.

이제 **Transformer Dataset** 라이브러리에서 `0`은 ‘sadness’, `1`이 ‘joy’가 된다.

<br>

### load nested json data

아래 데이터중에서 RawText와 Analysis 객체 안에 한 단계 더 들어가 있는 GeneralPolrarity을 Dataset에 가져오고 싶다.

```python
[
  {
    "Index": "119058",
    "RawText": "가성비 좋은 노트북 찾다가 구매했습니다...",
    "Source": "SNS",
    "Analysis": {
      "GeneralPolarity": "1",
      "Confidence": 0.98,
      "Aspects": [
        { "Aspect": "가격", "SentimentPolarity": "1" },
        { "Aspect": "기능", "SentimentPolarity": "1" }
      ]
    }
  }
]
```

<br>

**중첩된 Feature 객체 정의**

```python
From datasets import load_dataset, Features, Value, ClassLabel

nested_features = Features({
    'RawText': Value('string'),
    'Analysis': {
        'GeneralPolarity': ClassLabel(names=['0', '1'])
    }
})
```

<br>

`load_dataset`으로 중첩된 구조 그대로 데이터셋을 불러온다.

```python
dataset = load_dataset(
    'json',
    data_files=your_nested_data.json,
    features=nested_features, # Features 전달
    split='train'
)

# output:
# Dataset({
#     features: {'RawText': Value(dtype='string', id=None), 'Analysis': {'GeneralPolarity': ClassLabel(num_classes=2, names=['0', '1'], id=None)}},
#     num_rows: (전체 샘플 개수)
# })
```

<br>

**`.map()`**으로 애널리시스 객체안의 제너럴폴라리티 값을 꺼내어 최상위 계층으로 옮긴다.

```python
def flatten_structure(example):
    polarity_value = example['Analysis']['GeneralPolarity']

    return {
        'RawText': example['RawText'],
        'GeneralPolarity': polarity_value
    }

flattened_dataset = nested_dataset.map(
    flatten_structure,
    remove_columns=['Analysis']
)

# output:
# Dataset({
#     features: {'RawText': Value(dtype='string', id=None), 'GeneralPolarity': ClassLabel(num_classes=2, names=['0', '1'], id=None)},
#     num_rows: 2
# })
```

<br>

### load_dataset 동작원리

1. load_dataset 함수 호출
2. 지정된 캐시 디렉토리에서 데이터셋에 대한 캐시가 있는지 확인한다.
   - 없으면 다음단계를 진행한다.
   - 있으면 3~6 단계들을 스킵하고 7단계로 넘어가 `Dataset` 객체를 즉시 생성해 반환한다.
3. 원본 파일을 읽는다.
4. 파일 내용을 한줄 씩 파싱해서 각 컬럼의 데이터 타입을 추론한다.
5. 모든 정보를 바탕으로 데이터를 아파치 애로우 포맷으로 변환한다.
6. 변환된 애로우 테이블을 지정된 캐시 디렉토리에 저장한다.
7. 저장된 파일을 불러와 Dataset 객체를 생성해 반환하다.

<br>

### 파라미터

load_dataset은 다양한 파라미터들이 있다.

```python
(
		path: str
		name: typing.Optional[str] = None
		data_dir: typing.Optional[str] = None
		data_files: typing.Union[str, collections.abc.Sequence[str], collections.abc.Mapping[str, typing.Union[str, collections.abc.Sequence[str]]], NoneType] = None
		split: typing.Union[str, datasets.splits.Split, list[str], list[datasets.splits.Split], NoneType] = None
		cache_dir: typing.Optional[str] = None
		features: typing.Optional[datasets.features.features.Features] = None
		download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None
		download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None
		verification_mode: typing.Union[datasets.utils.info_utils.VerificationMode, str, NoneType] = None
		keep_in_memory: typing.Optional[bool] = None
		save_infos: bool = Falserevision: typing.Union[str, datasets.utils.version.Version, NoneType] = None
		token: typing.Union[bool, str, NoneType] = None
		streaming: bool = Falsenum_proc: typing.Optional[int] = None
		storage_options: typing.Optional[dict] = None**config_kwargs
)

return → Dataset or DatasetDict
```

<br>

**path**

데이터셋의 경로 또는 포맷을 지정한다.

- 허깅페이스 허브 경로: `‘username/dataset_name’` 처럼 고유 주소를 입력한다.
- 로컬 디렉토리: `‘./path/to/my_data/’` 처럼 로컬에 있는 파일들이 담긴 폴더 경로를 입력한다.
- 빌더이름(포맷): `‘csv’`, `‘json’` 처럼 데이터 포맷을 지정해 해당 포맷 전용 빌더를 불러온다. 이 경우는 반드시 `data_files` 파라미터로 파일 위치를 알려줘야한다.

<br>

**data_dir**

데이터 파일들이 모여있는 **디렉토리 경로**를 지정한다.

`data_files`를 지정하지 않고 `data_dir`만 지정하면, 해당 디렉토리 안에 있는 모든 파일을 데이터셋으로 불러온다. `data_files`를 일일이 지정하기 번거로울 때 유용하다.

<br>

**data_files**

불러올 **데이터 파일의 정확한 경로**를 지정한다.

`{'train': 'train.csv', 'test': 'test.csv'}`처럼 딕셔너리 형태로 전달하여 데이터셋을 `train`, `test` 스플릿으로 나누는것이 가장 흔하다.

<br>

여러 파일들 경로를 넣어준다.

- 리스트로 여러 파일:

  ```python
  train_files_list = [
      './my_data/train_part_1.csv',
      './my_data/train_part_2.csv',
      './my_data/train_part_3.csv'
  ]

  train_dataset = load_dataset('csv', data_files=train_files_list, split='train')
  ```

- 딕셔너리 형태로 `train`, `test`, `valid` 등 스플릿

  ```python
  # 파일
  file_dictionary = {
      'train': './my_data/training_data.csv',
      'test': './my_data/testing_data.csv',
      'validation': './my_data/validation_data.csv'
  }

  # 디렉토리
  forders_dictionary = {
      'train': './my_data/train/',
      'test': './my_data/test/',
      'validation': './my_data/valid/'
  }

  dataset_dict = load_dataset('csv', data_files=file_dictionary)
  dataset_dict = load_dataset('csv', data_files=forders_dictionary)
  ```

<br>

**split**

데이터를 train/test/valid 등 스플릿을 선택한다.

- `split='train'`: dataset을 `train`에 저장한다.
- `split=None` (기본값): 모든 스플릿(`train`, `test` 등)을 스플릿을 나누지 않고 저장한다.

<br>

**cache_dir**

처리된 데이터(애로우 테이블)를 **저장하거나 읽어올 캐시 디렉토리**를 지정한다.

기본값은 `~/.cache/huggingface/datasets`이며, 한 번 불러온 데이터셋은 이곳에 저장되어 다음 로드 시 매우 빠르게 불러온다.

<br>

**features**

데이터셋의 **스키마(구조와 타입)를 지정한다.**

이전에 설명했듯이, `ClassLabel`을 사용해 레이블에 의미를 부여하거나, 데이터 타입을 강제하고 싶을 때 사용한다.

```python
emotion_feature = Features({
		# text 컬럼을 string 타입으로 명시한다.
		'text' : Value('string'),

		# label 컬럼을 ClassLabel 타입으로 명시한다.
		'label' : ClassLabel(names=class_names),
})

dataset_custom = load_dataset(
    'json',
    data_dir='path/to/local/my_dataset.json', # 데이터 폴더안의 파일 가져옴
    # data_files='path/to/local/my_dataset.json' 파일 하나 가능
    column_names=['text', 'label'],
    features=emotion_features  # ***** 우리가 만든 features 객체를 지정 *****
)
```

<br>

**download_mode**

데이터를 허브같은 곳에서 다운로드하고 생성하는 **방식을 결정한다.**

- `REUSE_DATASET_IF_EXISTS` (기본값): 캐시가 있으면 재사용한다.
- `FORCE_REDOWNLOAD`: 캐시가 있어도 무시하고 강제로 다시 다운로드 및 처리한다.

<br>

**verification_mode**

다운로드한 데이터의 **무결성을 검증하는 수준**을 정한다.

체크섬, 파일 크기 등을 검사하여 데이터가 손상되지 않았는지 확인한다. 보통 기본값으로 충분하다.

<br>

**keep_in_memory**

데이터셋 전체를 복사하여 **RAM(메모리)에 저장시키고 유지할지** 결정한다.

데이터셋이 작을 경우, 메모리에 올려두고 사용하면 디스크 I/O가 없어 처리 속도가 더 빨라진다.

메모리 사용량과 속도를 맞바꾸는 옵션이다.

<br>

**revision**

허브에 있는 데이터셋의 **특정 버전을 지정한다.**

Git의 커밋 해시(commit SHA)나 태그(tag)를 사용하여, 특정 시점의 데이터셋을 불러올 수 있어 연구나 실험의 **재현성**을 보장하는 데 중요하다.

<br>

**token**

허브에 있는 **비공개(private) 데이터셋**에 접근하기 위한 **인증 토큰**을 전달한다.

`True`로 설정하면 로컬에 저장된 토큰(`~/.huggingface`)을 자동으로 사용한다.

<br>

**streaming**

`True`로 설정하면, 데이터를 미리 다운로드하지 않고 **실시간으로 스트리밍한다.**

필요할때마다 데이터를 실시간으로 가져와서 처리한다.

`for` 루프를 통해서 샘플에 접근할때마다 필요한 만큼 데이터를 가져온다.

매우 큰 데이터셋을 다룰 때 메모리와 디스크 공간을 절약할 수 있다.

<br>

**num_proc**

데이터를 처리할 때 사용할 **병렬 프로세스의 개수**를 지정한다.

CPU 코어를 여러 개 활용하여 데이터 전처리 속도를 크게 향상시킬 수 있다.

ex) `num_proc=8`

<br>

**`*config_kwargs` (additional keyword arguments)**

위에 명시되지 않은, **특정 데이터셋 빌더(Builder)에만 필요한 추가적인 키워드 인자**들을 전달할 때 사용됩니다.

<br>

**반환 값 (Returns)**

이러한 파라미터 조합에 따라 `load_dataset`은 다음과 같은 객체를 반환한다.

- `split` 지정 O, `streaming=False` -> `Dataset`
- `split` 지정 X, `streaming=False` -> `DatasetDict`
- `split` 지정 O, `streaming=True` -> `IterableDataset`
- `split` 지정 X, `streaming=True` -> `IterableDatasetDict`

<br>

참고

- [Hugging Face load_dataset Docs](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/loading_methods#loading-methods)
