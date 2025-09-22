- [Causal language modeling이란? (인과적 언어 모델링): 데이터셋/학습/추론](#causal-language-modeling이란-인과적-언어-모델링-데이터셋학습추론)
  - [데이터 준비 과정](#데이터-준비-과정)
    - [Transformer 모델 내부 label 처리](#transformer-모델-내부-label-처리)
  - [훈련](#훈련)
    - [훈련 코드](#훈련-코드)
  - [추론](#추론)

# Causal language modeling이란? (인과적 언어 모델링): 데이터셋/학습/추론

언어 모델링(language modeling)은 주어진 텍스트에서 다음 단어나 토큰을 예측하는 것을 말한다.

즉, 언어 모델은 문장의 의미와 구조를 이해해서 어떤 단어가 올지 확률적으로 예측하는 능력을 갖고 있다.

<br>

예를 들어서

```python
문장: 나는 오늘 아침에 ______ 먹었다.
```

모델은 빈칸에 들어갈 단어를 확률적으로 예측할 수 있다.

이때 어떤방식으로 학습하냐에 따라 언어 모델링 종류가 나뉜다.

1. Causal Language Model
2. Masked Language Model

<br>

Causal Language Model은

문장에서 앞의 단어만 보고 다음 단어를 예측하는 방식이다.

즉, 미래 정보를 보지않고 과거 정보만 이용한다.

<br>

특징

- 학습시 순서를 지키면서 이전 단어들만 입력으로 사용
- 자동완성, 텍스트 생성에 적합
- GPT 계열 모델들이 이 방식 사용

```python
입력: 나는 오늘 아침에
출력(예측): 밥
```

실제 활용 사례

1. 텍스트 생성: GPT, ChatGPT, Copilot
2. 코드 자동 완성: Copilot, CodeParrot
3. 스토리나 게임 시나리오 생성

<br>

## 데이터 준비 과정

> Causal Language Model의 멋진 점은 데이터셋에 레이블(label)이 필요하지 않다는 점이다.
> 다음 단어가 레이블(label)이기 때문이다.

<br>

데이터셋을 Dataset 객체로 변환

LLM 학습에서 중요한 점은 사용할 **모델이 실제 추론 때 받게 될 입력과 동일한 형식으로 학습 데이터**를 준비하는 것이다.

<br>

skt/A.X-4.0-Light 모델은 다음과 같은 입력 형식을 받는다.

```python

messages = [
    {"role": "system", "content": "당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다."},
    {"role": "user", "content": "The first human went into space and orbited the Earth on April 12, 1961."},
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
```

`“role” : “system”` 과 같은 형식으로 데이터를 준비하면 된다.

<br>

```python
from datasets import Dataset

train_data = [
    [
        {"role": "system", "content": "당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다."},
        {"role": "user", "content": "The first human went into space and orbited the Earth on April 12, 1961."},
        {"role": "assistant", "content": "1961년 4월 12일, 최초의 인간이 우주로 가서 지구를 공전했습니다."}
    ],
    [
        {"role": "system", "content": "당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다."},
        {"role": "user", "content": "Apollo 11 was the first mission to land humans on the Moon."},
        {"role": "assistant", "content": "아폴로 11호는 인간을 달에 착륙시킨 최초의 임무였습니다."}
    ]
]

# Hugging Face Dataset용 딕셔너리로 변환
dataset_dict = {"messages": train_data}

# 확인
print(dataset)

# Dataset({
#    features: ['answers.text'],
#    num_rows: 2
# })
```

- Hugging Face `datasets.Dataset`에서 `from_dict`를 사용할 때는 **딕셔너리 형태**가 필요하다.
- 즉, `{"key1": [값1, 값2, ...], "key2": [값1, 값2, ...]}` 형태여야 한다.
- `train_data`가 **리스트 안에 리스트 + 딕셔너리 구조**로 되어 있으면, 바로 `from_dict(train_data)`는 오류가 난다.
- 이렇게 하면 `dataset[0]["messages"]`로 각 대화에 접근할 수 있다.
- `apply_chat_template` 같은 토크나이저 변환 함수에 바로 넣을 수 있다.

<br>

`train_test_split(test_size=0.2)`을 사용해서 학습, 테스트 세트 분할 가능

```python
dataset = dataset.train_test_split(test_size=0.2)
```

<br>

전처리 함수 정의

```python
def preprocess_function(batch):
    """
    batch는 {'messages': [샘플1, 샘플2, ...]} 형태
    DataCollator에서 padding과 attention_mask 처리 예정
    """
    input_ids_list = []

    for messages in batch["messages"]:
        # apply_chat_template로 프롬프트 문자열 → 토큰화
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).squeeze(0)  # 1차원 텐서로 변환

        input_ids_list.append(encoded)

    return {"input_ids": input_ids_list}  # attention_mask는 DataCollator가 처리
```

- answer.text에 있는 각각 배열 하나마다, 문맥이 이어지는 하나의 문장으로 이루어져 있는 문장이다. 따라서 `“ ”.join()`으로 배열안의 요소들을 하나의 문맥으로 이어준다.

<br>

map으로 토큰화 적용

```python
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names
)

# 확인
print(tokenized_dataset)
```

`batched=True`

- `preprocess_function`이 단일 예제(x) 하나씩 받는 게 아니라 **배치 단위의 예제 리스트**를 받도록 바뀐다.
- 그래서 `" ".join(x) for x in examples["answers.text"]` 같은 코드가 가능해진다.
- 만약 False면 `examples`가 단일 예제가 되므로 리스트 컴프리헨션이 오류 날 수 있다.

<br>

`num_proc=4`

- 병렬 처리. 4개의 프로세스를 사용해서 속도를 올린다.
- 데이터셋이 크면 여러 CPU를 동시에 쓰는 게 유리하다.

<br>

`remove_columns=dataset.column_names`

- 원래 데이터셋 안의 모든 컬럼을 제거한다.
- 따라서 결과에는 **`preprocess_function`에서 리턴한 토큰화 결과만 남는다**.
- 즉 `"answers.text"`를 합쳐서 토큰화한 값만 남게 된다.

<br>

반환되는 데이터셋

```python
{
    "input_ids": [[101, 1234, 5678, ...], [101, 2345, 6789, ...]],
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
  - “컬럼을 제거한다”는 말은 **원래 객체에 있던 id, question, answers.text 등의 필드를 삭제**

<br>

토큰화

```python
from transformers import DataCollatorForLanguageModeling

# 1. 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

# 2. 토큰화, 라벨, 패딩 처리
data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False # 배치 패딩과 attention_mask 자동 처리
)
```

**1. pad_token을 eos_token으로 설정하는 이유**

토크나이저(tokenizer)는 텍스트를 토큰 단위로 바꿔주는 역할이다.

GPT 같은 **causal LM 모델**은 원래 문장의 끝을 알리는 특별한 토큰(eos_token)만 가지고 있음

그런데 학습할 때 여러 문장을 **같은 길이**로 맞춰야 한다.

<br>

일반적으로 문장을 일정 길이로 맞추기 위해 길이가 짧은 문장은 뒤를 빈 공간(padding)으로 채워야 한다.

`pad_token`은 패딩할 때 쓰는 특별한 토큰이다.

causal LM(GPT 계열)에는 원래 pad_token(**패딩용 토큰)이 정의되어 있지 않은 경우가 많음**

**모델이 이미 아는 eos_token(**End Of Sequence)**을 패딩용으로 재활용**한다.

즉, “문장의 끝”을 의미하는 토큰을 “빈 칸”처럼 사용한다고 이해하면 된다.

```python
tokenizer.eos_token  # '<|endoftext|>'
tokenizer.pad_token  # None -> pad_token = '<|endoftext|>'
```

이제 배치에서 짧은 문장을 패딩할 때도 **'<|endoftext|>'** 토큰을 사용

<br>

**2. DataCollatorForLanguageModeling의 역할**

`DataCollatorForLanguageModeling`은 **배치(batch)를 만들면서 언어 모델 학습에 필요한 처리를 자동으로 해주는 클래스**

1. 문장 길이 맞추기

   - 배치 안에 있는 문장들의 길이가 다 다르다.
   - 짧은 문장은 pad_token(eos_token)으로 채워서 모두 같은 길이로 맞춘다.
   - 입력이 모두 같은 길이가 아니면 배치의 최대 길이에 동적으로 패딩

   패딩 후(batch 길이 4):

   ```python
   [[101, 102, 103, 104],
    [101, 105, 106, 104]]  # 104는 eos_token으로 패딩
   ```

2. 라벨 생성
3. MLM 여부 결정
   - `mlm=False` → masked LM이 아닌 Causal LM용
     - Causal LM은 항상 다음 단어를 에측하는 것이 목적이다.
     - 문장 일부를 가리지 않고, 순차적으로 다음 단어 예측을 학습한다.
     - `labels` = `input_ids` 그대로 해서 라벨 생성
     - 마지막 패딩 부분은 예측할 필요가 없으므로 학습에서 무시(-100)한다.
   - `mls=True`일때
     - 일부 토큰만 mask 처리 → 예측 대상은 **mask 토큰뿐**
     - 나머지(non-masked) 토큰 위치는 -100으로 처리
     ```python
     input_ids = [101, 11, 22, 33]
     mask       = [False, True, False, False]  # 11만 mask
     labels     = [-100, 11, -100, -100]
     ```
     모델은 **mask된 11만 예측**하고, 나머지 토큰은 학습에서 무시
4. DataCollator에서 tokenizer를 넣는 이유

   DataCollator는 **padding token ID를 알아야** 다음을 수행할 수 있다

   1. 배치 패딩
      - padding 토큰으로 **짧은 시퀀스를 채우기**
   2. attention mask 생성
      - padding 위치를 0, 실제 토큰 위치를 1로 표시
   3. (MLM이면) 마스킹
      - masked LM 학습 시 **mask 토큰을 선택**

   - 즉, tokenizer 없이는 DataCollator가 **어떤 숫자로 패딩할지, 어떤 토큰을 마스킹할지** 알 수 없다.

<br>

정리

1. GPT 계열 causal LM 모델은 **MLM이 필요 없고, 순차적 생성이 목적**이므로 `mlm=False` 설정
2. 배치에서 문장 길이가 달라도 **패딩 필요** → pad_token 지정
3. 토큰화와 라벨 생성을 **자동화** → `DataCollatorForLanguageModeling` 사용
4. 학습 시 **loss 계산에서 패딩 무시** → 정확한 다음 단어 예측 학습 가능

<br>

정리하면, 이 코드는 **GPT 계열 모델을 학습할 때 데이터 배치를 준비하는 표준적인 방법**

- `pad_token = eos_token` → 패딩 문제 해결
- `mlm=False` → causal LM용 라벨 생성

<br>

### Transformer 모델 내부 label 처리

내부 구현을 보면 **shift는** `forward()` **모델 내부에서 자동으로 수행한다.**

그래서 **collator에서 label을 따로 오른쪽으로 밀 필요가 없다**.

- causal LM에서는 labels가 **입력 시퀀스를 오른쪽에서 왼쪽으로 한 칸씩 밀어서 다음 단어 예측**용으로 만들어진다.
  - 왜냐하면 모델이 현재 보고 있는 위치의 토큰을 가지고 바로 그 토큰을 맞추는 게 아니라 그 다음 위치에 나올 토큰을 맞춰야 하기 때문이다.
  - 예를 들어 입력이 ["나는", "밥을", "먹는다"]라는 토큰 시퀀스라고 하자.(원래는 토큰화로 숫자여야함)
  - 그렇다면 labels는 ["밥을", "먹는다", -100]처럼 만들어지는데 여기서 한 칸 오른쪽에서 왼쪽으로 밀려 있다는 것을 볼 수 있다.
  - $t_1$ 시점에 입력으로 “나는”이 들어 왔으면, 그 다음 단어를 예측 한다. 그 다음 단어로 “밥을”이 와야한다. 그래서 오른쪽에서 왼쪽으로 밀어서 label 배열을 만들어 준것이다.
- 모델이 시점 t에서 출력한 분포와 label로 정렬된 t 시점의 정답(원래 시퀀스의 t+1 토큰)을 비교해 loss를 계산하고, 이 loss를 전체 시퀀스에 대해 합산하거나 평균내어 하나의 스칼라 손실값을 얻는다.손실값을 미분해서(기울기 계산) 모델 파라미터를 업데이트하는 것이 학습의 핵심

<br>

## 훈련

모델 로드

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

<br>

훈련 하이퍼파라미터 정의

필수로 정의해야하는 파라미터는 `output_dir`으로 모델을 저장할 위치를 지정하는것이다.

```python
training_args = TrainingArguments(
    output_dir="my_awesome_eli5_clm-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
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

훈련이 완료되면 evaluate() 메서드를 사용해서 모델 평가

```python
import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

<br>

### 훈련 코드

```python
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AdamW

# 1. 학습 데이터 예시
train_data = [
    [
        {"role": "system", "content": "당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다."},
        {"role": "user", "content": "The first human went into space and orbited the Earth on April 12, 1961."},
        {"role": "assistant", "content": "1961년 4월 12일, 최초의 인간이 우주로 가서 지구를 공전했습니다."}
    ],
    [
        {"role": "system", "content": "당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다."},
        {"role": "user", "content": "Apollo 11 was the first mission to land humans on the Moon."},
        {"role": "assistant", "content": "아폴로 11호는 인간을 달에 착륙시킨 최초의 임무였습니다."}
    ]
]

# 2. Dataset 객체 생성
dataset_dict = {"messages": train_data}
dataset = Dataset.from_dict(dataset_dict)

# 3. 토크나이저 로드 및 pad_token 설정
tokenizer = AutoTokenizer.from_pretrained("qwen-7b-chat")  # 예시 모델
tokenizer.pad_token = tokenizer.eos_token  # CLM에서는 eos_token을 pad_token으로 사용

# 4. preprocess_function 정의
def preprocess_function(batch):
    input_ids_list = []
    for messages in batch["messages"]:
        # messages를 프롬프트 문자열로 변환 후 토크나이즈
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).squeeze(0)
        input_ids_list.append(encoded)
    return {"input_ids": input_ids_list}

# 5. map 적용
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 6. DataCollator 생성 (CLM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # causal LM
)

# 7. DataLoader 준비
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator
)

# 8. 모델 로드
model = AutoModelForCausalLM.from_pretrained("qwen-7b-chat")
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 옵티마이저
optimizer = AdamW(model.parameters(), lr=5e-5)

# 9. 학습 루프 예시 (1 epoch)
for batch in dataloader:
    # batch의 각 요소를 device로 이동
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

<br>

## 추론

방법1: pipeline 추론

```python
prompt = "Somatic hypermutation allows the immune system to"

from transformers import pipeline

generator = pipeline("text-generation", model="username/my_awesome_eli5_clm-model")
generator(prompt)

```

```python
[{'generated_text': "Somatic hypermutation allows the immune system to be able to effectively reverse the damage caused by an infection.\n\n\nThe damage caused by an infection is caused by the immune system's ability to perform its own self-correcting tasks."}]
```

<br>

방법2:

토큰화후 input_ids 반환

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_clm-model")
inputs = tokenizer(prompt, return_tensors="pt").input_ids
```

<br>

generate() 메서드로 텍스트 생성

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("username/my_awesome_eli5_clm-model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
```

<br>

생성된 토큰 ID를 다시 텍스트로 디코딩

```python
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

```python
["Somatic hypermutation allows the immune system to react to drugs with the ability to adapt to a different environmental situation. In other words, a system of 'hypermutation' can help the immune system to adapt to a different environmental situation or in some cases even a single life. In contrast, researchers at the University of Massachusetts-Boston have found that 'hypermutation' is much stronger in mice than in humans but can be found in humans, and that it's not completely unknown to the immune system. A study on how the immune system"]
```

참고

[**Hugging Face Causal language modeling Docs**](https://huggingface.co/docs/transformers/v4.56.2/en/tasks/language_modeling)
