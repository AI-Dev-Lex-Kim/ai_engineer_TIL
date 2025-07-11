- [Hugging Face Dataset.map](#hugging-face-datasetmap)
  - [Batch Mapping](#batch-mapping)
    - [map 함수 파라미터(batch)](#map-함수-파라미터batch)
    - [Feature](#feature)
    - [batch\_size](#batch_size)
    - [num\_proc](#num_proc)

# Hugging Face Dataset.map

## Batch Mapping

`datasets` 라이브러리에서 배치 매핑은 `batched=True` 파라미터를 설정한다.

`batch_size=100(defaults: 1000)`와 같이 배치 사이즈도 정한다.

데이터셋의 각 샘플을 개별적으로 처리하는 대신, **여러 샘플을 묶은 배치 단위로 함수를 적용한다.**

<br>

배치 매핑을 하는 가장 큰 이유는 처리 속도를 높이는 것이다.

단일 샘플로 처리하면 샘플 개수가 10만개면, 함수를 10만번 불러야 하지만 배치단위는 훨씬 줄어든다.

또한 데이터 메모리와 CPU 사이를 오가는 횟수를 줄인다.

단일 샘플로 처리하는 것보다 **배치단위로 데이터를 한번에 처리할때 훨씬 빠르다.**

```python
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
class_names = ['부정', '중립', '긍정']

features = Features({
    'input_ids': Sequence(Value('int32')),
    'attention_mask': Sequence(Value('int32')),
    'token_type_ids': Sequence(Value('int32')),
    'label': ClassLabel(names=class_names)
})

def tokenize_batch(batch):
	"""
	batch = {
	    'RawText': ["안녕하세요", "반갑습니다", "즐거운 하루"], # <-- 리스트 형태
	    'GeneralPolarity': ['1', '0', '1']                  # <-- 리스트 형태
	}
	"""
  labels = []
  cleaned_texts = []
  for i, text in enumerate(batch['RawText']):
    cleaned_text = clean_text(text)
    cleaned_texts.append(cleaned_text)

    label = batch['GeneralPolarity'][i]
    labels.append(int(label) + 1)

  tokened_batch = tokenizer(cleaned_texts, max_length=256, truncation=True)

  tokened_batch['label'] = labels

  return tokened_batch

type_dataset = filtered_dataset['train'].map(tokenize_batch,
                                             features=features,
                                             remove_columns=['RawText', 'GeneralPolarity'],
                                             batched=True,
                                             batch_size=1024,
                                             num_proc=4)
```

<br>

### map 함수 파라미터(batch)

map에 들어가는 함수의 파라미터는 batch 크기만큼의 샘플들이 들어온다.

예를들어

```python
batch = {
	    'RawText': ["안녕하세요", "반갑습니다", "즐거운 하루"], # <-- 리스트 형태
	    'GeneralPolarity': ['1', '0', '1']              # <-- 리스트 형태
}
```

<br>

### Feature

`Features`는 단일 샘플에서의 컬럼 타입을 정해주면 된다.

- `input_ids`, `attention_mask`, `token_type_ids`, → `Sequence(Value(’int32’))`
- `label` → `ClassLabel`

주의해야할 점은 `feature` 스키마를 적용했으니, 반드시 `remove_columns`를 해주어야한다.

그렇지 않으면 기존의 컬럼 이름이 `dataset`의 피처로 나온다.

따라서 feature 스키마에는 `label`,`input_ids`, `attention_mask`, `token_type_ids` 4개가 있지만, 기존의 `RawText`, `GeneralPolarity` 가 추가로 있어 예상하지 못해 에러가 난다.

<br>

### batch_size

한번에 처리할 배치 사이즈를 지정해준다.

실험해본 결과 단일 샘플로 맵핑을 할경우 `4408/4408 [00:19<00:00, 540.18 examples/s]` 19초가 걸렸다.

1024 배치로 맵핑을 할경우 `4408/4408 [00:08<00:00, 1060.46 examples/s]` 8초가 걸렸다.

8, 16, 32, 64, 128, 256, 512, 1024 순으로 적용해보니 클수록 점점 빨라졌다.

초반에 데이터를 탐색하는 시간이 오래걸리고, 어느정도 탐색하면 배치 크기수 만큼 빠르게 처리한다.

즉 데이터셋의 크기가 작다면 그렇게 큰 의미가 없다.

**큰 데이터셋일 경우는 점점 더 빠르게 처리한다.**

<br>

### num_proc

처리할 프로세스 개수를 지정해준다.

<br>

참고

- [Hugging Face Dataset.map Docs](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map)
