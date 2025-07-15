- [skt/koBERT Full Fine-Tuning/LoRA 감정분석 모델 구현](#sktkobert-full-fine-tuninglora-감정분석-모델-구현)
  - [Load data](#load-data)
  - [**Preprocessing**](#preprocessing)
  - [Dataset, Tokenizer](#dataset-tokenizer)
  - [DataLoader](#dataloader)
    - [Collator](#collator)
  - [Full Fine-Tuning Model](#full-fine-tuning-model)
    - [Train Loop](#train-loop)
    - [Evaluate Loop](#evaluate-loop)
    - [Train/Eval Loop](#traineval-loop)
    - [학습 분석](#학습-분석)
  - [LoRA 모델](#lora-모델)
    - [LoRA 설정](#lora-설정)
    - [Train Loop](#train-loop-1)
    - [Evaluate Loop](#evaluate-loop-1)
    - [Train/Eval Loop](#traineval-loop-1)
    - [학습 분석](#학습-분석-1)
    - [LoRA에 의해 업데이트된 파라미터 확인](#lora에-의해-업데이트된-파라미터-확인)

# skt/koBERT Full Fine-Tuning/LoRA 감정분석 모델 구현

<br>

## Load data

```python
root_path = "/content/drive/MyDrive/ai_enginner/codeit/스프린트 미션/mission13"
data_path = f'{root_path}/data/SNS/04. IT기기'
cache_dir = f'{root_path}/data/cache'

class_names = ['부정', '중립', '긍정']

features = Features({
    'RawText': Value('string'),
    'GeneralPolarity': ClassLabel(names=class_names)
})

dataset = load_dataset('json',
                       data_dir=data_path,
                       cache_dir=cache_dir,
                       num_proc=4,
          )
```

<br>

## **Preprocessing**

```python
remove_dataset = dataset.remove_columns(['Index', 'Source', 'Domain', 'MainCategory', 'ProductName', 'ReviewScore', 'Syllable', 'Word', 'RDate', 'Aspects'])

# DatasetDict({
#     train: Dataset({
#         features: ['RawText', 'GeneralPolarity'],
#         num_rows: 5004
#     })
# })
```

<br>

```python
def filter_valid_polarity(example):
    # GeneralPolarity' 키가 딕셔너리에 존재하는지 확인
    if 'GeneralPolarity' not in example:
        return False

    polarity_value = example['GeneralPolarity']

    # 값이 None인지 확인
    if polarity_value is None:
        return False

    # 값이 빈 문자열인지 확인
    if polarity_value == '':
        return False

    # 값이 -1, 0, 1 인지 확인
    if str(polarity_value) not in ['-1', '0', '1']:
        return False # 유효한 숫자 범위에 없으면 유효하지 않음

    return True

filtered_dataset = remove_dataset.filter(filter_valid_polarity)

# DatasetDict({
#     train: Dataset({
#         features: ['RawText', 'GeneralPolarity'],
#         num_rows: 4408
#     })
# })
```

`filter()`을 사용해 `Generlarity`의 이상치, 결측치를 제거해줬다.

<br>

## Dataset, Tokenizer

이제는 각 샘플들에 여러가지 전처리를 추가적으로 해주어야한다.

- 타입 명시
- 토크나이저
- 텍스트 정제

<br>

[`.map()`](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map)을 사용하면 된다.

```python
def clean_text(text):
    # 예시: HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', text)
    # 특수문자 제거(단, 문장부호는 남김)
    text = re.sub(r'[^0-9가-힣A-Za-z\.\,\?\!]+', ' ', text)
    # 중복 공백 하나로
    text = re.sub(r'\s+', ' ', text).strip()
    return text

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
class_names = ['부정', '중립', '긍정']

features = Features({
    'input_ids': Sequence(Value('int32')),
    'attention_mask': Sequence(Value('int32')),
    'token_type_ids': Sequence(Value('int32')),
    'label': ClassLabel(names=class_names)
})

def tokenize_fn(example):
  text =  clean_text(example['RawText'])
  label = example['GeneralPolarity']

  tokend_example = tokenizer(text, max_length=256, truncation=True)
  tokend_example['label'] = int(label) + 1

  return tokend_example

type_dataset = filtered_dataset['train'].map(tokenize_fn,
                                             features=features,
                                             remove_columns=['RawText', 'GeneralPolarity'])
```

**타입 명시**

Features를 통해 아웃풋의 타입을 명시해준다.

- `input_ids`, `attention_mask`, `token_type_ids`, → `Sequence(Value(’int32’))`
- `label` → `ClassLabel`

주의해야할 점은 `feature` 스키마를 적용했으니, 반드시 `remove_columns`를 해주어야한다.

그렇지 않으면 기존의 컬럼 이름대로 dataset의 피처로 나온다.

따라서 feature 스키마에는 `label`,`input_ids`, `attention_mask`, `token_type_ids` 4개가 있지만, 기존의 `RawText`, `GeneralPolarity` 가 그대로 남아있어 예상하지 못해 에러가 난다.

<br>

**토크나이저**

토크나이저는 Hugging Face `AutoTokenizer`을 통해 받는다.

2020년 8월 이후 업데이트가 중단된 `skt/KoBERT` github의 토크나이저를 받게된다면,

최신 python 및 딥러닝 프레임 워크 환경과 충돌이 발생할 수 있다.

<br>

현재 가장 권장되는 방식으로, 모든 하위 의존성을 관리하는 hugging face의 transformers 라이브러리을 사용하는 것이다.

**초창기 transformers (4년 전) 에서는 custom code를 이용한 Auto mapping이 불가하여 skt/kobert-base-v1에서 파이썬 라이브러리 형태로만 제공을 했다.**

이제는 \*\*\*\*`monologg/kobert`를 통해 바로 호출 가능하게 수정 되었다.

`AutoTokenizer.from_pretrained(”monologg/kobert", trust_remote_code=True”)` 코드로 KoBERT의 복잡한 토크나이저를 추상화해서 불러올 수 있다.

<br>

**텍스트 정제**

clean function를 만들어서 HTML 태그, 특수문자, 중복 공백 제거를 해주었다.

<br>

**배치매핑**

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

이제 데이터를 train과 test로 스플릿 해준다.

만약 데이터를 불러올때 해주었다면 안해줘도 된다.

```python
train_val_dataset = type_dataset.train_test_split(test_size=0.2, shuffle=True)

# DatasetDict({
#     train: Dataset({
#         features: ['input_ids', 'attention_mask', 'token_type_ids', 'label'],
#         num_rows: 3526
#     })
#     test: Dataset({
#         features: ['input_ids', 'attention_mask', 'token_type_ids', 'label'],
#         num_rows: 882
#     })
# })
```

<br>

## DataLoader

DataLoader 클래스가 요구하는 메서드는 두가지가 있다.

1. `__len__`
2. `__getitem__(index)`

이 두가지만 있다면 어떤 데이터셋이든 DataLoader는 전달받아 처리할 수 있다.

허깅페이스의 dataset 라이브러리는 arrow dataset 타입이다. 이 두 가지 메소드를 모두 구현되어 있어 DataLoader에서 작업을 수행 할 수 있다.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=2, pin_memory=True)
```

<br>

### Collator

자연어처리를 할때는 배치 단위로 모델에 입력하려면, 모든 문장의 길이를 동일하게 맞춰야한다.

허깅페이스의 `DataCollatorwithPadding`을 사용하면 좋다.

<br>

`DataCollatorWithPadding`

```python
from transformer import DataCollatorWithPadding

(
		tokenizer: PreTrainedTokenizerBase
		padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True
		max_length: typing.Optional[int] = None
		pad_to_multiple_of: typing.Optional[int] = None
		return_tensors: str = 'pt'
)
```

**tokenizer**

토크나이저를 넣어주면 된다.

이 콜레이터는 토크나이저로부터

- 패딩에 사용할 토큰 ID가 무엇인지,
- 패딩을 문장의 왼쪽에 할지 오른쪽에 할지 등의 정보를 얻는다.

<br>

**padding(default: True)**

시퀀스를 어떻게 패딩할지 정한다.

- `True` or `‘longest’`(기본값): 배치 내에서 가장 긴 시퀀스의 길이에 맞춰 나머지 시퀀스를 패딩한다.
- `‘max_length’`: `max_length` 인자로 지정된 최대 길이에 맞춰 모든 시퀀스를 패딩한다. `max_length`가 주어지지 않으면 해당 모델이 수용할 수 있는 가장 긴 길이로 패딩한다.
- `False` or `‘do_not_pad’`: 패딩을 전혀하지 않는다.

<br>

**max_length**

타입: `int`

반환될 시퀀스의 최대 길이를 지정한다.

`padding=’max_length’`으로 사용할때 패딩의 길이 기준이다.

<br>

**pad_to_multiple_of**

시퀀스의 길이를 주어진 값의 배수가 되도록 패딩한다.

예를들어서 이값을 8로 설정하고 패딩 후 시퀀스 길이가 22가 되었다면, 길이를 8의 배수인 24로 맞추기 위해 추가적인 패딩을 더한다.

<br>

**return_tensors
반환활 텐서의 종류를 지정한다.**

- `“pt”`: 파이토치 텐서로 반환(기본값)
- `“tf”`: 텐서플로우 텐서로 반환
- `“np”`: 넘파이 배열로 반환

<br>

우리 코드에 적용은 아래와 같이한다.

```python
dataCollatorwithPadding = DataCollatorwithPadding(tokenizer=tokenzier, padding=True, return_tensors="pt")
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=True, collate_fn=dataCollatorwithPadding)
test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=2, pin_memory=True collate_fn=dataCollatorwithPadding)
```

<br>

## Full Fine-Tuning Model

허깅페이스에서 태스크에 맞는 모델(예: `AutoModelForSequenceClassification`)로 따로 부르면 너무 추상화 되어있어서 어렵고, 커스텀 모델을 만드니깐 내가 작업하기도 편하다.

예를 들어, 분류 레이어를 `Linear -> ReLU -> Linear` 처럼 2단으로 쌓거나, 드롭아웃(Dropout) 비율을 세밀하게 바꾸거나, 어텐션 풀링(Attention Pooling) 같은 다른 풀링(pooling) 방식을 적용하기가 까다롭다.

데이터의 흐름과 각 레이어의 역할을 명확히 코드로 정의하기 때문에 디버깅이 쉽고 모델의 동작을 완벽하게 이해할 수 있다.

<br>

허깅페이스에서 베이스 모델만 부르고(`AutoModel.from_pretrained`) 다운스트림 태스크에 맞는 커스텀 모델을 만들면 된다.

```python
kobert_base_model = AutoModel.from_pretrained(MODEL_NAME)

class CustomKoBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        # 불러온 kobert_base_model을 내부 레이어로 사용
        self.kobert = base_model
        self.dropout = nn.Dropout(p=0.1)
        # KoBERT의 출력(768)을 받아 우리가 원하는 클래스 수로 변환
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # 베이스 모델로부터 출력을 얻음
        outputs = self.kobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [CLS] 모든 배치에 대해서 0번째 토큰의 768차원 벡터만 선택함.
        # 0번째 토큰은 항상 문장의 시작을 알리는 [CLS] 토큰이다.
        # 문장 전체의 의미를 압축한 대표 벡터가 되도록 학습한다.
        cls_token_output = outputs.last_hidden_state[:, 0, :]

        # 문장의 대표 벡터인 [CLS] 토큰의 출력을 드롭 아웃 레이어에 통과시켜 과적합 방지
        final_output = self.dropout(cls_token_output)

        # 각 클래스에 대한 최종 예측 점수인 로짓(logit)을 계산한다.
        logits = self.classifier(final_output)

        return logits

my_model = CustomKoBERTClassifier(kobert_base_model, NUM_CLASSES)

```

<br>

### Train Loop

```python
def train_loop():
    koBERT_classifier_model.train()

    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = koBERT_classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    return avg_train_loss
```

<br>

### Evaluate Loop

```python
def eval_loop():
    koBERT_classifier_model.eval()

    all_predictions = []
    all_labels = []

    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          token_type_ids = batch['token_type_ids'].to(device)
          labels = batch['labels'].to(device)

          logits = koBERT_classifier_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
          )

          loss = criterion(logits, labels)

          total_val_loss += loss.item()
          preds = torch.max(logits, 1)[1]

          all_predictions.extend(preds.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_dataloader)

    results = metric.compute(
      predictions=all_predictions,
      references=all_labels,
      average="macro"  # 클래스 불균형이 있을 때 각 클래스를 동일한 가중치로 평균
    )

    return avg_val_loss, results
```

<br>

### Train/Eval Loop

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("monologg/kobert")
koBERT_classifier_model = KoBERTClassifier(model, 3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(koBERT_classifier_model.parameters(), lr=5e-5)
num_epochs = 25
history = {
    'train_loss': [],
    'val_loss': [],
    'precision': [],
    'recall': [],
    'f1': []
}
metric = evaluate.combine(["precision", "recall", "f1"])

for epoch in range(num_epochs):
    avg_train_loss = train_loop()
    eval_loss, metric_results = eval_loop()

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(eval_loss)

    precision = metric_results['precision']
    recall = metric_results['recall']
    f1 = metric_results['f1']

    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)

    print(f"--- Epoch {epoch + 1}/{num_epochs} Results ---")
    print(f"Train Loss:      {avg_train_loss:.4f}")
    print(f"Validation Loss: {eval_loss:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}\n")

PATH = f"{root_path}/chkpt/kobert_classifier_full_fine_tuining_model_weights.pt"
torch.save(koBERT_classifier_model.state_dict(), PATH)

plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.grid(True)
plt.show()
```

- 결과 로그

  ```python
  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.67it/s]
  --- Epoch 1/25 Results ---
  Train Loss:      0.8308
  Validation Loss: 0.6169
  Precision:       0.7282
  Recall:          0.7244
  F1 Score:        0.7262

  100%|██████████| 111/111 [00:31<00:00,  3.53it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.64it/s]
  --- Epoch 2/25 Results ---
  Train Loss:      0.6151
  Validation Loss: 0.5730
  Precision:       0.7291
  Recall:          0.7186
  F1 Score:        0.7230

  100%|██████████| 111/111 [00:31<00:00,  3.53it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.61it/s]
  --- Epoch 3/25 Results ---
  Train Loss:      0.5014
  Validation Loss: 0.6107
  Precision:       0.7757
  Recall:          0.7125
  F1 Score:        0.7239

  100%|██████████| 111/111 [00:31<00:00,  3.53it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.66it/s]
  --- Epoch 4/25 Results ---
  Train Loss:      0.3931
  Validation Loss: 0.6451
  Precision:       0.7302
  Recall:          0.6812
  F1 Score:        0.6992

  100%|██████████| 111/111 [00:31<00:00,  3.53it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.65it/s]
  --- Epoch 5/25 Results ---
  Train Loss:      0.3027
  Validation Loss: 0.7563
  Precision:       0.7293
  Recall:          0.6980
  F1 Score:        0.7111

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.65it/s]
  --- Epoch 6/25 Results ---
  Train Loss:      0.2301
  Validation Loss: 0.7715
  Precision:       0.7392
  Recall:          0.7274
  F1 Score:        0.7329

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.62it/s]
  --- Epoch 7/25 Results ---
  Train Loss:      0.1749
  Validation Loss: 0.8662
  Precision:       0.7354
  Recall:          0.6964
  F1 Score:        0.7071

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.64it/s]
  --- Epoch 8/25 Results ---
  Train Loss:      0.1776
  Validation Loss: 0.9243
  Precision:       0.7336
  Recall:          0.6872
  F1 Score:        0.7001

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.60it/s]
  --- Epoch 9/25 Results ---
  Train Loss:      0.1448
  Validation Loss: 0.8656
  Precision:       0.7411
  Recall:          0.7171
  F1 Score:        0.7247

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.60it/s]
  --- Epoch 10/25 Results ---
  Train Loss:      0.1038
  Validation Loss: 0.9703
  Precision:       0.7607
  Recall:          0.7319
  F1 Score:        0.7380

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.62it/s]
  --- Epoch 11/25 Results ---
  Train Loss:      0.0622
  Validation Loss: 1.1862
  Precision:       0.7072
  Recall:          0.7205
  F1 Score:        0.7058

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.59it/s]
  --- Epoch 12/25 Results ---
  Train Loss:      0.1041
  Validation Loss: 1.1202
  Precision:       0.7336
  Recall:          0.7282
  F1 Score:        0.7266

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.55it/s]
  --- Epoch 13/25 Results ---
  Train Loss:      0.0408
  Validation Loss: 1.2882
  Precision:       0.7629
  Recall:          0.6911
  F1 Score:        0.7123

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.59it/s]
  --- Epoch 14/25 Results ---
  Train Loss:      0.0506
  Validation Loss: 1.2249
  Precision:       0.7197
  Recall:          0.6722
  F1 Score:        0.6903

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.60it/s]
  --- Epoch 15/25 Results ---
  Train Loss:      0.0709
  Validation Loss: 1.2927
  Precision:       0.7414
  Recall:          0.7239
  F1 Score:        0.7203

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.63it/s]
  --- Epoch 16/25 Results ---
  Train Loss:      0.0408
  Validation Loss: 1.5118
  Precision:       0.6774
  Recall:          0.7339
  F1 Score:        0.6899

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.63it/s]
  --- Epoch 17/25 Results ---
  Train Loss:      0.0558
  Validation Loss: 1.2706
  Precision:       0.7383
  Recall:          0.7252
  F1 Score:        0.7200

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.64it/s]
  --- Epoch 18/25 Results ---
  Train Loss:      0.0618
  Validation Loss: 1.3364
  Precision:       0.7495
  Recall:          0.6681
  F1 Score:        0.6929

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.55it/s]
  --- Epoch 19/25 Results ---
  Train Loss:      0.0523
  Validation Loss: 1.3592
  Precision:       0.7283
  Recall:          0.6846
  F1 Score:        0.7017

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.62it/s]
  --- Epoch 20/25 Results ---
  Train Loss:      0.0314
  Validation Loss: 1.4030
  Precision:       0.6900
  Recall:          0.6897
  F1 Score:        0.6885

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.57it/s]
  --- Epoch 21/25 Results ---
  Train Loss:      0.0351
  Validation Loss: 1.2727
  Precision:       0.7309
  Recall:          0.7242
  F1 Score:        0.7255

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.57it/s]
  --- Epoch 22/25 Results ---
  Train Loss:      0.0394
  Validation Loss: 1.3963
  Precision:       0.7314
  Recall:          0.6823
  F1 Score:        0.6984

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.57it/s]
  --- Epoch 23/25 Results ---
  Train Loss:      0.0865
  Validation Loss: 1.1937
  Precision:       0.7006
  Recall:          0.6857
  F1 Score:        0.6914

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.55it/s]
  --- Epoch 24/25 Results ---
  Train Loss:      0.0634
  Validation Loss: 1.1686
  Precision:       0.7416
  Recall:          0.7033
  F1 Score:        0.7190

  100%|██████████| 111/111 [00:31<00:00,  3.52it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.31it/s]
  --- Epoch 25/25 Results ---
  Train Loss:      0.0373
  Validation Loss: 1.3051
  Precision:       0.7107
  Recall:          0.7179
  F1 Score:        0.7107
  ```

### 학습 분석

![image.png](../images/자연어/skt%20koBERT%20Full%20Fine-Tuning%20LoRA%20감정분석%20모델%20구현/image1.png)

1. train loss
   - 0.8308에서 시작해서 0.0351까지 매우 꾸준하고 빠르게 감소하고있다.
   - 모델이 주어진 훈련데이터에 학습을 잘하고 있음을 보여준다.
2. eval loss
   - 에폭 2에서 0.5730으로 최저점을 기록했다.
   - 그 이후 에폭 3부터 0.6107로 다시 증가하고 이후 지속 상승했다.
   - 에폭 3이후부터 오버피팅의 신호라고 할 수 있다. 모델이 훈련 데이터를 너무 과하게 학습해서, 검증 데이터에 대한 일반화 성능을 잃기 시작했다.
3. F1 Score
   - 로그를 보면 에폭 10에서 0.7380으로 최고점을 기록했다.
   - 그 이후 F1 Score가 지속적으로 떨어지거나 불안정하게 변동했다.
   - 모델의 실질적인 예측 성능은 에폭 10에서 정점인것을 알 수 있다.
4. 결론
   - 가장 낮은 검증 손실 에폭 2
   - 가장 높은 F1 Score 에폭 10
   - 일반적으로 검증 데이터에 대해 F1 Score가 가장 높은 모델을 최종 모델로 선택한다. 따라서 에폭 10의 모델이 최상의 모델일 가능성이 높다.
5. 향후 개선 방향
   - 데이터 증강이나 드롭 아웃 비율을 증가와 같은 오버피팅 완화 기법을 적용하면 최고 F1 Score을 더 높일 수 있을 것이다.

<br>

- Full Fine-Tuning 전체 코드

  ```python
  # !pip install -U evaluate
  !pip install -U datasets huggingface_hub fsspec
  !pip install evaluate

  import os
  import json
  import torch
  import torch.nn as nn
  from torch.optim import AdamW
  from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
  from datasets import Dataset, load_dataset, Features, ClassLabel, Value, Sequence
  import pandas as pd
  import re
  import evaluate
  import numpy as np
  from tqdm import tqdm
  from torch.utils.data import DataLoader

  root_path = "/content/drive/MyDrive/ai_enginner/codeit/스프린트 미션/mission13"
  data_path = f'{root_path}/data/SNS/04. IT기기'
  cache_dir = f'{root_path}/data/cache'

  dataset = load_dataset('json',
                         data_dir=data_path,
                         cache_dir=cache_dir,
                         num_proc=4,
                         )

  remove_dataset = dataset.remove_columns(['Index', 'Source', 'Domain', 'MainCategory', 'ProductName', 'ReviewScore', 'Syllable', 'Word', 'RDate', 'Aspects'])
  remove_dataset

  def filter_valid_polarity(example):
      # GeneralPolarity' 키가 딕셔너리에 존재하는지 확인
      if 'GeneralPolarity' not in example:
          return False

      polarity_value = example['GeneralPolarity']

      # 값이 None인지 확인
      if polarity_value is None:
          return False

      # 값이 빈 문자열인지 확인
      if polarity_value == '':
          return False

      # 값이 -1, 0, 1 인지 확인
      if str(polarity_value) not in ['-1', '0', '1']:
          return False # 유효한 숫자 범위에 없으면 유효하지 않음

      return True

  filtered_dataset = remove_dataset.filter(filter_valid_polarity)
  filtered_dataset

  def clean_text(text):
      # 예시: HTML 태그 제거
      text = re.sub(r'<[^>]+>', ' ', text)
      # 특수문자 제거(단, 문장부호는 남김)
      text = re.sub(r'[^0-9가-힣A-Za-z\.\,\?\!]+', ' ', text)
      # 중복 공백 하나로
      text = re.sub(r'\s+', ' ', text).strip()
      return text

  tokenzier = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
  class_names = ['부정', '중립', '긍정']
  features = Features({
      'input_ids': Sequence(Value('int32')),
      'attention_mask': Sequence(Value('int32')),
      'token_type_ids': Sequence(Value('int32')),
      'labels': ClassLabel(names=class_names)
  })

  def tokenize_batch(batch):
    labels = []
    cleaned_texts = []
    for i, text in enumerate(batch['RawText']):
      cleaned_text = clean_text(text)
      cleaned_texts.append(cleaned_text)

      label = batch['GeneralPolarity'][i]
      labels.append(int(label) + 1)

    tokened_batch = tokenzier(cleaned_texts, max_length=256, truncation=True)

    tokened_batch['labels'] = labels

    return tokened_batch

  tokened_dataset = filtered_dataset['train'].map(tokenize_batch,
                                               features=features,
                                               remove_columns=['RawText', 'GeneralPolarity'],
                                               batched=True,
                                               batch_size=512,
                                               num_proc=2)

  train_test_dataset = tokened_dataset.train_test_split(test_size=0.2, shuffle=True)
  train_dataset = train_test_dataset['train']
  test_dataset = train_test_dataset['test']
  train_test_dataset

  dataCollatorWithPadding = DataCollatorWithPadding(tokenizer=tokenzier, padding=True, return_tensors="pt")

  train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=True, collate_fn=dataCollatorWithPadding)
  test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=2, pin_memory=True, collate_fn=dataCollatorWithPadding)

  class KoBERTClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
      super().__init__()
      self.bert = base_model
      self.dropout = nn.Dropout(p=0.1)
      self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
      output = self.bert(input_ids, attention_mask, token_type_ids)

      # [CLS] 토큰
      cls_token_output = output.last_hidden_state[:, 0, :]
      drop_cls_token = self.dropout(cls_token_output)
      logits = self.classifier(drop_cls_token)

      return logits

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AutoModel.from_pretrained("monologg/kobert")
  koBERT_classifier_model = KoBERTClassifier(model, 3).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=5e-5)

  num_epochs = 25

  koBERT_classifier_model.train()

  for epoch in range(num_epochs):

    for batch in tqdm(train_dataloader):
      optimizer.zero_grad()

      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      token_type_ids = batch['token_type_ids'].to(device)
      labels = batch['labels'].to(device)

      logits = koBERT_classifier_model(input_ids, attention_mask, token_type_ids)
      loss = criterion(logits, labels)

      loss.backward()
      optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

  metric = evaluate.combine(["precision", "recall", "f1"])

  # 모델을 평가 모드로 설정한다.
  model.eval()

  # 모든 예측과 라벨을 저장할 빈 리스트를 만든다.
  all_predictions = []
  all_labels = []

  # 그래디언트 계산을 비활성화한다.
  with torch.no_grad():
      # 평가용 데이터로더에서 데이터를 가져온다.
      for batch in tqdm(test_dataloader, desc="Evaluating"):
          # 라벨과 입력값을 분리한다. (필요시 to(device) 추가)
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          token_type_ids = batch['token_type_ids'].to(device)
          labels = batch['labels'].to(device)

          # 모델에 입력을 전달하여 logits를 얻는다.
          # **batch는 딕셔너리의 내용을 인자로 풀어주는 역할을 한다.
          logits = koBERT_classifier_model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
          )

          # logits에서 가장 높은 값의 인덱스를 예측 결과로 선택한다.
          predictions = torch.argmax(logits, dim=1)

          # 예측 결과와 실제 라벨을 리스트에 추가한다.
          # GPU 사용 시 .cpu()를 붙여야 numpy로 변환이 가능하다.
          all_predictions.extend(predictions.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())

  # compute 함수에 예측값과 정답(references) 리스트를 전달하여 최종 점수를 계산한다.
  results = metric.compute(
      predictions=all_predictions,
      references=all_labels,
      average="macro"
  )

  # 결과 출력
  precision = results['precision']
  recall = results['recall']
  f1 = results['f1']

  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1 Score: {f1:.4f}")
  ```

<br>

## LoRA 모델

```python
# AutoModel을 사용하기 위해, 기존 풀파인튜닝 코드에 있었던 분류기 클래스를 다시 정의한다.
class KoBERTClassifier(nn.Module):
  def __init__(self, base_model, num_classes):
    super().__init__()
    # 이 클래스는 베이스 BERT 모델과 분류를 위한 선형 레이어를 포함한다.
    self.bert = base_model
    self.dropout = nn.Dropout(p=0.1)
    # BERT 모델의 출력 차원(hidden_size)을 입력으로 받고, 클래스 개수(num_classes)만큼 출력하는 선형 레이어를 정의한다.
    self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

  def forward(self, input_ids, attention_mask, token_type_ids):
    # 베이스 BERT 모델을 통과시켜 출력을 얻는다.
    output = self.bert(input_ids, attention_mask, token_type_ids)

    # BERT 출력의 last_hidden_state에서 [CLS] 토큰에 해당하는 부분(첫 번째 토큰)을 추출한다.
    cls_token_output = output.last_hidden_state[:, 0, :]
    # 드롭아웃을 적용하여 과적합을 방지한다.
    drop_cls_token = self.dropout(cls_token_output)
    # 분류기(선형 레이어)를 통과시켜 최종 로짓(logits)을 계산한다.
    logits = self.classifier(drop_cls_token)

    return logits
```

<br>

### LoRA 설정

```python
# AutoModel을 사용하여 분류 헤드가 없는 순수 KoBERT 모델을 로드한다.
base_model = AutoModel.from_pretrained("monologg/kobert")

# LoRA 설정을 정의한다.
peft_config = LoraConfig(
    # 커스텀 헤드를 사용하므로 task_type을 명시적으로 지정할 필요가 없다.
    # 추론 모드가 아님을 명시한다. (학습 시에는 False)
    inference_mode=False,
    # LoRA 어댑테이션 매트릭스의 랭크(rank)를 설정한다.
    r=8,
    # LoRA 스케일링 파라미터.
    lora_alpha=16,
    # LoRA 레이어에 적용할 드롭아웃 비율이다.
    lora_dropout=0.1,
    # LoRA를 적용할 모듈의 이름을 리스트로 지정한다.
    target_modules=["query", "value"],
    # 편향(bias) 파라미터는 학습하지 않도록 설정한다.
    bias="none"
)

# get_peft_model 함수를 사용하여 '베이스 모델'에 LoRA 설정을 적용한다.
# 이 과정을 통해 base_model의 query, value 레이어는 LoRA 레이어로 대체되고 나머지 파라미터는 동결된다.
peft_base_model = get_peft_model(base_model, peft_config)

# 학습 가능한 파라미터의 수와 그 비율을 출력한다.
# 이 값은 LoRA 레이어의 파라미터 수만 나타낸다. (분류기 헤드는 아직 포함되지 않음)
peft_base_model.print_trainable_parameters()

# LoRA가 적용된 베이스 모델(peft_base_model)을 사용하여 KoBERTClassifier 인스턴스를 생성한다.
koBERT_classifier_model = KoBERTClassifier(peft_base_model, 3).to(device)

```

<br>

### Train Loop

```python
def lora_train_loop():
    koBERT_classifier_model.train()

    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = koBERT_classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    return avg_train_loss
```

<br>

### Evaluate Loop

```python
def lora_eval_loop():
    koBERT_classifier_model.eval()

    all_predictions = []
    all_labels = []

    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = koBERT_classifier_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = criterion(logits, labels)

            total_val_loss += loss.item()
            predictions = torch.max(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_dataloader)

    metric_results = metric.compute(
        predictions=all_predictions,
        references=all_labels,
        average="macro"  # 클래스 불균형이 있을 때 각 클래스를 동일한 가중치로 평균
    )

    return avg_val_loss, metric_results
```

<br>

### Train/Eval Loop

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("monologg/kobert")
koBERT_classifier_model = KoBERTClassifier(model, 3).to(device)

criterion = nn.CrossEntropyLoss()
# 옵티마이저는 전체 모델(koBERT_classifier_model)의 파라미터를 전달받는다.
# 이렇게 하면 동결되지 않은 파라미터, 즉 'LoRA 레이어'와 '분류기 헤드'의 파라미터만 학습 대상이 된다.
optimizer = AdamW(koBERT_classifier_model.parameters(), lr=5e-4)
num_epochs = 25
history = {
    'train_loss': [],
    'val_loss': [],
    'precision': [],
    'recall': [],
    'f1': []
}
metric = evaluate.combine(["precision", "recall", "f1"])

for epoch in range(num_epochs):
    train_loss = lora_train_loop()
    eval_loss, metric_results = lora_eval_loop()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(eval_loss)

    precision = metric_results['precision']
    recall = metric_results['recall']
    f1 = metric_results['f1']

    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)

    print(f"--- Epoch {epoch + 1}/{num_epochs} Results ---")
    print(f"Train Loss:      {train_loss:.4f}")
    print(f"Validation Loss: {eval_loss:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}\n")

PATH = f"{root_path}/chkpt/kobert_classifier_LoRA_model_weights.pt"
torch.save(koBERT_classifier_model.state_dict(), PATH)

plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.grid(True)
plt.show()
```

- 학습/평가 로그

  ```python
  100%|██████████| 111/111 [00:22<00:00,  4.93it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00,  9.98it/s]
  --- Epoch 1/25 Results ---
  Train Loss:      0.8703
  Validation Loss: 0.6569
  Precision:       0.7010
  Recall:          0.6598
  F1 Score:        0.6657

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.15it/s]
  --- Epoch 2/25 Results ---
  Train Loss:      0.6240
  Validation Loss: 0.6273
  Precision:       0.7403
  Recall:          0.6850
  F1 Score:        0.6976

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.14it/s]
  --- Epoch 3/25 Results ---
  Train Loss:      0.5360
  Validation Loss: 0.6304
  Precision:       0.7541
  Recall:          0.6759
  F1 Score:        0.6992

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.09it/s]
  --- Epoch 4/25 Results ---
  Train Loss:      0.4498
  Validation Loss: 0.6108
  Precision:       0.7443
  Recall:          0.7197
  F1 Score:        0.7242

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.13it/s]
  --- Epoch 5/25 Results ---
  Train Loss:      0.3683
  Validation Loss: 0.6465
  Precision:       0.7413
  Recall:          0.7280
  F1 Score:        0.7315

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.21it/s]
  --- Epoch 6/25 Results ---
  Train Loss:      0.2822
  Validation Loss: 0.6990
  Precision:       0.6905
  Recall:          0.7301
  F1 Score:        0.7007

  100%|██████████| 111/111 [00:21<00:00,  5.11it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.18it/s]
  --- Epoch 7/25 Results ---
  Train Loss:      0.1967
  Validation Loss: 0.7961
  Precision:       0.7259
  Recall:          0.7350
  F1 Score:        0.7300

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.18it/s]
  --- Epoch 8/25 Results ---
  Train Loss:      0.1286
  Validation Loss: 0.8886
  Precision:       0.7351
  Recall:          0.7150
  F1 Score:        0.7240

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.15it/s]
  --- Epoch 9/25 Results ---
  Train Loss:      0.0837
  Validation Loss: 1.0432
  Precision:       0.7274
  Recall:          0.7195
  F1 Score:        0.7227

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.18it/s]
  --- Epoch 10/25 Results ---
  Train Loss:      0.0668
  Validation Loss: 1.0418
  Precision:       0.7274
  Recall:          0.7231
  F1 Score:        0.7250

  100%|██████████| 111/111 [00:21<00:00,  5.11it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.21it/s]
  --- Epoch 11/25 Results ---
  Train Loss:      0.0290
  Validation Loss: 1.2242
  Precision:       0.7378
  Recall:          0.7002
  F1 Score:        0.7146

  100%|██████████| 111/111 [00:21<00:00,  5.11it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.21it/s]
  --- Epoch 12/25 Results ---
  Train Loss:      0.0306
  Validation Loss: 1.3130
  Precision:       0.7176
  Recall:          0.7401
  F1 Score:        0.7255

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.16it/s]
  --- Epoch 13/25 Results ---
  Train Loss:      0.0442
  Validation Loss: 1.2828
  Precision:       0.7374
  Recall:          0.7270
  F1 Score:        0.7317

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.10it/s]
  --- Epoch 14/25 Results ---
  Train Loss:      0.0294
  Validation Loss: 1.2973
  Precision:       0.7311
  Recall:          0.7332
  F1 Score:        0.7321

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.11it/s]
  --- Epoch 15/25 Results ---
  Train Loss:      0.0102
  Validation Loss: 1.3346
  Precision:       0.7506
  Recall:          0.7206
  F1 Score:        0.7335

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.19it/s]
  --- Epoch 16/25 Results ---
  Train Loss:      0.0044
  Validation Loss: 1.3995
  Precision:       0.7396
  Recall:          0.7306
  F1 Score:        0.7344

  100%|██████████| 111/111 [00:21<00:00,  5.11it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.13it/s]
  --- Epoch 17/25 Results ---
  Train Loss:      0.0046
  Validation Loss: 1.4565
  Precision:       0.7459
  Recall:          0.7295
  F1 Score:        0.7369

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.18it/s]
  --- Epoch 18/25 Results ---
  Train Loss:      0.0013
  Validation Loss: 1.4624
  Precision:       0.7301
  Recall:          0.7215
  F1 Score:        0.7256

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.16it/s]
  --- Epoch 19/25 Results ---
  Train Loss:      0.0006
  Validation Loss: 1.5013
  Precision:       0.7402
  Recall:          0.7264
  F1 Score:        0.7328

  100%|██████████| 111/111 [00:21<00:00,  5.11it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.18it/s]
  --- Epoch 20/25 Results ---
  Train Loss:      0.0004
  Validation Loss: 1.5345
  Precision:       0.7421
  Recall:          0.7281
  F1 Score:        0.7346

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.14it/s]
  --- Epoch 21/25 Results ---
  Train Loss:      0.0003
  Validation Loss: 1.5610
  Precision:       0.7372
  Recall:          0.7276
  F1 Score:        0.7320

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00,  9.95it/s]
  --- Epoch 22/25 Results ---
  Train Loss:      0.0003
  Validation Loss: 1.5797
  Precision:       0.7377
  Recall:          0.7296
  F1 Score:        0.7334

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.10it/s]
  --- Epoch 23/25 Results ---
  Train Loss:      0.0002
  Validation Loss: 1.5979
  Precision:       0.7387
  Recall:          0.7304
  F1 Score:        0.7342

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.12it/s]
  --- Epoch 24/25 Results ---
  Train Loss:      0.0002
  Validation Loss: 1.6194
  Precision:       0.7404
  Recall:          0.7314
  F1 Score:        0.7354

  100%|██████████| 111/111 [00:21<00:00,  5.10it/s]
  Evaluating: 100%|██████████| 28/28 [00:02<00:00, 10.15it/s]
  --- Epoch 25/25 Results ---
  Train Loss:      0.0002
  Validation Loss: 1.6324
  Precision:       0.7364
  Recall:          0.7292
  F1 Score:        0.7324
  ```

<br>

![image.png](../images/자연어/skt%20koBERT%20Full%20Fine-Tuning%20LoRA%20감정분석%20모델%20구현/image2.png)

### 학습 분석

1. Train Loss
   - 0.8703에서 시작해서 0.0002 까지 매우 꾸준하고 빠르게 감소했다.
   - 모델이 훈련 데이터에 학습을 잘 진행하는것을 보여준다.
2. Eval Loss
   - 에폭 4에서 0.6108로 최저점을 기록했다.
   - 그 이후 에폭 5부터 다시 증가하기 시작해서 지속 상승 했다.
   - 이것은 에폭 5부터 오버피팅이 시작됐음을 알수있다.
3. F1 Score
   - 에폭 17에서 0.7369 최고점을 기록했다.
   - 검증 손실이 증가한후 F1 Score는 한동안 개선되는 경향을 보였다.
4. 결론
   - 검증 손실은 에폭 4가 가장 낮다.
   - 하지만 일반화가 잘된 모델이 아니라 가장 감정 분류를 정확하게 하는 모델이 최종 목표이다.
   - 따라서 실제 성능 지표인 F1 Score가 가장 높은 에폭 17의 모델을 최종 모델로 선택해야한다.
5. LoRA vs Full Fine-Tuning
   - LoRA의 최고 F1 Score 7369는는 Full Fine-Tuning 0.7380과 거의 비슷한 성능을 보여줬다.
   - LoRA는 훨씬 적은 파라미터만으로도 비슷한 성능을 달성했으므로, 학습 효율성이 압도적으로 높는것을 알 수 있다.
   - 오히려 LoRA는 기존 모델을 거의 건드리지 않기 때문에, 사전 훈련으로 학습된 일반적인 지식을 덜 잊는다는 장점도 있다.

<br>

### LoRA에 의해 업데이트된 파라미터 확인

```python
import peft

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def find_lora_parameters(model):
    lora_params = 0
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.Linear):
            lora_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, peft.utils.ModulesToSaveWrapper):
            lora_params += sum(p.numel() for p in module.parameters())

    return lora_params

total_params = count_parameters(model)
lora_params = find_lora_parameters(model)

print(f"Total Parameters: {total_params}")
print(f"LoRA Parameters: {lora_params}")
print(f"LoRA Ratio: {(lora_params / total_params) * 100:.2f}%")

Total Parameters: 92484099
LoRA Parameters: 14469120
LoRA Ratio: 15.64%
```

<br>

참고

- [Hugging Face Load Docs](https://huggingface.co/docs/datasets/loading#json)
- [KoBERT로 7가지 감정 분류 모델 구현 블로그](https://jinya.tistory.com/2)
- [KoBERT를 활용한 감정분류 모델 구현 with Colab](https://bbarry-lee.github.io/ai-tech/KoBERT%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B0%90%EC%A0%95%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84.html)
- [LoRA 이번만큼 완벽하게 이해해보자. with 실습 코드 Colab 블로그](https://colab.research.google.com/drive/1LvZH-Mo0Lu6uc-R-i6BIdb8K8btXrGQe?usp=sharing)
- [https://developerahjosea.tistory.com/entry/LoRA-이번만큼-완벽하게-이해해보자-with-실습-코드](https://developerahjosea.tistory.com/entry/LoRA-%EC%9D%B4%EB%B2%88%EB%A7%8C%ED%81%BC-%EC%99%84%EB%B2%BD%ED%95%98%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%B4%EB%B3%B4%EC%9E%90-with-%EC%8B%A4%EC%8A%B5-%EC%BD%94%EB%93%9C)
