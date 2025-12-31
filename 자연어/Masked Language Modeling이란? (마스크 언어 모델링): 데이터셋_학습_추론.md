- [Masked Language Modeling이란? (마스크 언어 모델링): 데이터셋/학습/추론](#masked-language-modeling이란-마스크-언어-모델링-데이터셋학습추론)
  - [데이터 준비 과정](#데이터-준비-과정)
  - [학습](#학습)
  - [추론](#추론)
  - [Summarization task](#summarization-task)

# Masked Language Modeling이란? (마스크 언어 모델링): 데이터셋/학습/추론

<mark>**Masked language modeling**</mark>은 문장에서 일부 단어를 숨기고 맞추는 학습 방식이다.

모델은 숨겨진 단어의 왼쪽과 오른쪽 문맥을 모두 참고할 수 있다.

BERT가 이런 마스크 언어 모델의 대표적인 예로, 문장 전체 이해가 필요한 작업에 적합하다.

<br>

반대로, 문장 속 일부 단어를 가리고(MASK) 해당 단어를 맞추는 방식이다.

즉, 문장 전체를 보고 빈칸을 예측한다.

특징

- 입력 문장 전체를 고려
- 문장 이해, 분류, 정보 추출에 적합
- BERT 계열 모델들이 이 방식 사용

```python
문장: 나는 오늘 아침에 [MASK] 먹었다.
출력: 밥
```

모델은 앞과 뒤 문맥을 모두 참조해서 [MASK]를 예측한다.

순서를 반드시 지키지 않아도 된다.

<br>

실제 활용 사례

1. 문장 분류: 감정분석
2. 특정 단어 찾기
3. 정보 추출
4. BERT, RoBERTa

<br>

## 데이터 준비 과정

> <mark>**Masked Language Modeling**</mark>의 멋진 점은 데이터셋에 레이블(label)이 필요하지 않다는 점이다.
> 다음 단어가 레이블(label)이기 때문이다.

<br>

데이터셋을 Dataset 객체로 변환

```python
# answers.text가 리스트 형태인 예시 데이터
data = {
    "answers.text": [
        ["우리 은하와 가장 가까운 은하가 있다.", "그 은하 이름은 안드로메다 은하이다."], # 문장2
        ["어제 엄마한테 혼났다.", "다음에는 나쁜짓을 안해야겠다."] # 문장2
    ]
}

# Dataset 객체 생성
dataset = Dataset.from_dict(data)

# 확인
print(dataset)

# Dataset({
#    features: ['answers.text'],
#    num_rows: 2
# })
```

<br>

전처리 함수 정의

```python
def preprocess_function(examples):
    # 리스트 안의 문장을 하나로 합쳐서 토큰화
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

# "우리 은하와 가장 가까운 은하가 있다. 그 은하 이름은 안드로메다 은하이다."
# "어제 엄마한테 혼났다. 다음에는 나쁜짓을 안해야겠다."
```

- answer.text에 있는 각각 배열 하나마다, 문맥이 이어지는 하나의 문장으로 이루어져 있는 문장이다. 따라서 `“ ”.join()`으로 배열안의 요소들을 하나의 문맥으로 이어준다.

<br>

map으로 토큰화 적용

```python
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 확인
print(tokenized_dataset)
```

`batched=True`

- `preprocess_function`이 단일 예제(x) 하나씩 받는 게 아니라 <mark>**배치 단위의 예제 리스트**</mark>를 받도록 바뀐다.
- 그래서 `" ".join(x) for x in examples["answers.text"]` 같은 코드가 가능해진다.
- 만약 False면 `examples`가 단일 예제가 되므로 리스트 컴프리헨션이 오류 날 수 있다.

<br>

`num_proc=4`

- 병렬 처리. 4개의 프로세스를 사용해서 속도를 올린다.
- 데이터셋이 크면 여러 CPU를 동시에 쓰는 게 유리하다.

<br>

`remove_columns=dataset.column_names`

- 원래 데이터셋 안의 모든 컬럼을 제거한다.
- 따라서 결과에는 <mark>**`preprocess_function`에서 리턴한 토큰화 결과만 남는다**</mark>.
- 즉 `"answers.text"`를 합쳐서 토큰화한 값만 남게 된다.

<br>

반환되는 데이터셋

```python
{
    "input_ids": [[101, 1234, 5678, ...], [101, 2345, 6789, ...]],
    "attention_mask": [[1,1,1, ...], [1,1,1, ...]]
}
```

- 원래 다른 컬럼들은 전부 제거된다
- 예제 단위가 유지되면서, 모델에 바로 넣을 수 있는 형태가 된다.
- 컬럼이란?

  ```python
  data = {
      "id": [1, 2],
      "question": ["오늘 날씨는?", "내일 일정은?"],
      "answers.text": [["좋다"], ["회의 있다"]]
  }

  dataset = Dataset.from_dict(data)
  print(dataset)

  # 출력 결과
  Dataset({
      features: ['id', 'question', 'answers.text'],
      num_rows: 2
  })
  ```

  - 여기서 컬럼은 `'id'`, `'question'`, `'answers.text'`
  - `preprocess_function`이 반환하는 값(`input_ids`, `attention_mask`)만 남게 됨
  - “컬럼을 제거한다”는 말은 <mark>**원래 객체에 있던 id, question, answers.text 등의 필드를 삭제**</mark>
  - `train_test_split(test_size=0.2)`을 사용해서 학습, 테스트 세트 분할 가능

  ```python
  dataset = dataset.train_test_split(test_size=0.2)
  ```

<br>

토큰화

```python
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mls=True, mlm_probability=0.15)
```

<mark>**1. pad_token을 eos_token으로 설정하는 이유**</mark>

토크나이저(tokenizer)는 텍스트를 토큰 단위로 바꿔주는 역할이다.

GPT 같은 <mark>**causal LM 모델**</mark>은 원래 문장의 끝을 알리는 특별한 토큰(eos_token)만 가지고 있음

그런데 학습할 때 여러 문장을 <mark>**같은 길이**</mark>로 맞춰야 한다.

<br>

일반적으로 문장을 일정 길이로 맞추기 위해 길이가 짧은 문장은 뒤를 빈 공간(padding)으로 채워야 한다.

`pad_token`은 패딩할 때 쓰는 특별한 토큰이다.

causal LM(GPT 계열)에는 원래 pad_token(<mark>**패딩용 토큰)이 정의되어 있지 않은 경우가 많음**</mark>

<mark>**모델이 이미 아는 eos_token(**</mark>End Of Sequence)<mark>**을 패딩용으로 재활용**</mark>한다.

즉, “문장의 끝”을 의미하는 토큰을 “빈 칸”처럼 사용한다고 이해하면 된다.

```python
tokenizer.eos_token  # '<|endoftext|>'
tokenizer.pad_token  # None -> pad_token = '<|endoftext|>'
```

이제 배치에서 짧은 문장을 패딩할 때도 <mark>**'<|endoftext|>'**</mark> 토큰을 사용

<br>

<mark>**2. DataCollatorForLanguageModeling의 역할**</mark>

`DataCollatorForLanguageModeling`은 <mark>**배치(batch)를 만들면서 언어 모델 학습에 필요한 처리를 자동으로 해주는 클래스**</mark>

1. 토크나이저

   - `tokenizer=tokenizer` → 토크나이저를 사용하여 배치 데이터를 숫자 토큰으로 변환

   입력 문장:

   ```python
   ["나는 오늘 아침에 밥 먹었다.", "오늘 날씨가 좋다."]
   ```

   토큰화 후:

   ```python
   [[101, 102, 103, 104], [101, 105, 106]]
   ```

2. 문장 길이 맞추기

   - 배치 안에 있는 문장들의 길이가 다 다르다.
   - 짧은 문장은 pad_token(eos_token)으로 채워서 모두 같은 길이로 맞춘다.
   - 입력이 모두 같은 길이가 아니면 배치의 최대 길이에 동적으로 패딩

   패딩 후(batch 길이 4):

   ```python
   [[101, 102, 103, 104],
    [101, 105, 106, 104]]  # 104는 eos_token으로 패딩

   ```

3. `mls=True`일때

   - 일부 토큰만 mask 처리 → 예측 대상은 <mark>**mask 토큰뿐**</mark>
   - 나머지(non-masked) 토큰 위치는 -100으로 처리

   ```python
   input_ids = [101, 11, 22, 33]
   mask       = [False, True, False, False]  # 11만 mask
   labels     = [-100, 11, -100, -100]
   ```

   모델은 <mark>**mask된 11만 예측**</mark>하고, 나머지 토큰은 학습에서 무시

4. `mlm_probability=0.15`
   - 마스킹될 토큰의 비율을 지정
   - 0.15 라면, 전체 토큰중 15%를 랜덤하게 [MASK]로 바꾼다는 의미
   - 선택된 토큰만 모델이 예측하도록 라벨(labels)로 지정된다.
   - 나머지 85%의 토큰은 그대로 모델 입력에 들어가지만, loss 계산에서 무시된다.
   - 너무 적으면 학습 신호가 부족해서 모델이 충분히 학습하지 못한다.
   - 너무 많으면 문맥이 손상되서 모델이 문장의 의미를 잃는다.
   - 따라서 대부분의 Hugging Face의 BERT 기반 모델에서 0.15를 기본값으로 사용한다.

<br>

## 학습

모델 로드

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

<br>

하이퍼파라미터 정의

필수 파라미터는 output_dir으로 모델을 저장할 위치를 정한다.

```python
training_args = TrainingArguments(
    output_dir="my_awesome_eli5_mlm_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

<br>

평가

```python
import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

<br>

## 추론

방법1: pipeline 추론

```python
text = "The Milky Way is a <mask> galaxy."

from transformers import pipeline

mask_filler = pipeline("fill-mask", "username/my_awesome_eli5_mlm_model")
mask_filler(text, top_k=3)
```

<br>

결과

```python
[{'score': 0.5150994658470154,
  'token': 21300,
  'token_str': ' spiral',
  'sequence': 'The Milky Way is a spiral galaxy.'},
 {'score': 0.07087188959121704,
  'token': 2232,
  'token_str': ' massive',
  'sequence': 'The Milky Way is a massive galaxy.'},
 {'score': 0.06434620916843414,
  'token': 650,
  'token_str': ' small',
  'sequence': 'The Milky Way is a small galaxy.'}]
```

<br>

방법2:

텍스트 토큰화 후 input_ids을 pytorch 텐서로 반환. `<mask>` 토큰의 위치를 지정

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
```

<br>

모델에 입력 전달후 마스킹된 토큰의 logits을 반환

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]
```

<br>

확률이 가장 높은 세개의 마스킹된 토큰을 반환

```python
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
```

```python
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```

<br>

## Summarization task

```python
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamW

# 1. 학습 데이터 예시 (Summarization)
train_data = [
    {
        "article": "The first human went into space and orbited the Earth on April 12, 1961.",
        "summary": "First human in space on April 12, 1961."
    },
    {
        "article": "Apollo 11 was the first mission to land humans on the Moon.",
        "summary": "Apollo 11 first humans on Moon."
    }
]

# 2. Dataset 객체 생성
dataset = Dataset.from_list(train_data)

# 3. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token  # 필요 시 pad_token 설정

# 4. preprocess_function 정의
def preprocess_function(batch):
    """
    input_ids: 기사(article)를 토크나이즈
    labels: summary를 토크나이즈
    """
    input_ids_list = []
    labels_list = []

    for article, summary in zip(batch["article"], batch["summary"]):
        input_encoded = tokenizer(
            article,
            truncation=True,
            max_length=128,
        )["input_ids"]
        label_encoded = tokenizer(
            summary,
            truncation=True,
            max_length=32
        )["input_ids"]

        input_ids_list.append(torch.tensor(input_encoded))
        labels_list.append(torch.tensor(label_encoded))

    return {"input_ids": input_ids_list, "labels": labels_list}

# 5. map 적용
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 6. DataCollator 생성 (padding 처리)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None  # 모델 지정 안하면 일반 padding
)

# 7. DataLoader 준비
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator
)

# 8. 모델 로드 (Seq2Seq)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 옵티마이저
optimizer = AdamW(model.parameters(), lr=5e-5)

# 9. 학습 루프 예시 (1 epoch)
for batch in dataloader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print("Batch loss:", loss.item())

```

Summarization task에서는 <mark>**두 가지**</mark>를 만들어야 한다

1. <mark>**input_ids**</mark> → 모델 입력(input), 여기서는 기사(article)
2. <mark>**labels**</mark> → 모델이 예측해야 하는 정답(output), 여기서는 요약(summary)

<br>

참고

- [<mark>**Hugging Face Masked language modeling Docs**</mark>](https://huggingface.co/docs/transformers/v4.56.2/en/tasks/masked_language_modeling#masked-language-modeling)
