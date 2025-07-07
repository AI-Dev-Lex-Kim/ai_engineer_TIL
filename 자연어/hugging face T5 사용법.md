# hugging face T5 사용법

## 데이터셋 불러오기 및 전처리

```python
def split_dataset(data_path):
  with open(data_path, 'r') as f:
    data = json.load(f)

  example = []
  for doc in data['documents']:
    text = doc['text']
    abstr = doc['abstractive']
    sentence = []
    for arr1 in text:
      for obj in arr1:
        clean_sentence = clean_text(obj['sentence'])
        sentence.append(clean_sentence)

    full_text = ' '.join(sentence)
    clean_abstr = clean_text(abstr[0])
    obj = {
        'full_text': full_text,
        'clean_abstr': clean_abstr
    }
    example.append(obj)

  return example

train_inputs = chage_example(news_train_path)
dataset_train_input = Dataset.from_list(train_inputs)
```

토크나이저에 들어가기 위해서는 hugging face Dataset 타입으로 들어가야한다.

배열을 허깅페이스 데이터셋 타입으로 바꾸려면 `Dataset.from_list(input)`를 사용하면 된다.

배열 안에는 샘플 하나당 하나의 객체가 들어있어야 한다.

여기에 input으로 들어 오려면 아래와 같은 형식이어야한다.

```python
from datasets import Dataset
my_list = [{"a": 1}, {"a": 2}, {"a": 3}]
dataset = Dataset.from_list(my_list)
print(dataset)

"""
Dataset({
   features: ['full_text', 'clean_abstr'],
   num_rows: 10
})
"""
```

<br>

## 모델 가져오기

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base-ko")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-base-ko").to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

허깅 페이스에 공개된 사전 학습 된 모델 “KETI-AIR/ke-t5-base-ko” 을 그대로 입력해주면 가져올 수 있다.

<br>

## 토크나이저(Tokenizer)

[토크나이저 파라미터 공식 docs](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__)

모델은 문자열 텍스트를 직접 처리 할 수 없어서 숫자로 변환해주어야한다.

토크나이저는 텍스트를 토큰이라 불리는 개별 단어를 나누는 과정이며, 이 토큰들이 최종적으로 숫자로 변환 되어 모델에 입력 된다.

<br>

해당 모델이 사용한 토크나이저도 모델 이름만 입력하면 자동으로 가져온다.

사전 학습된 모델과 동일한 토크나이저를 사용하는 것이 중요하다.

텍스트를 나누는 방식이 모델이 학습할 때 사용한 방식과 정확히 일치해야하기 떄문이다.

```python
text = 'i love you'
target = 'i love you too'
inputs = tokenizer(text, return_tensors="pt")
# 'input_ids': tensor([[  3,  23, 333,  25,   1]])
```

출력 값으로 input_ids에 5개의 토큰이 나왔다.

i, love, girl 이라는 3개의 토큰과 시작, 끝 토큰 5개가 된것으로 생각할 수 있다.

<br>

### map() 함수

`map()`함수는 각 데이터 샘플이 아니라 배치 단위로 토크나이저를 적용하기 때문에 훨씬 빠르다.

`batched=True`로 설정하면 여러 샘플을 한꺼번에 처리한다.

```python
def tokenization(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenization, batched=True)
```

<br>

Trainer에서는 데이터로더 형식으로 학습을 한다.

데이터로더는 배열 안에 각 요소가 샘플로 배치크기 만큼 존재한다.

따라서 `data[i]` 와 같이 샘플을 접근한다.

하지만 tokenizer을 사용한 아웃풋은 객체값을 반환한다.

따라서 토크나이즈만 하고 모델에 넣으면 동작을 하지 않는다.

`.map` 을 사용해서 각 배치별로 데이터셋을 나누어준다.(Dataset 역활과 똑같음)

```python
def tokenize_fn(exmaple):
  inputs = tokenizer(exmaple['full_text'],
                     text_target=exmaple['clean_abstr'],
                     max_length=512,
                     truncation=True)

  return inputs

token_model_inputs = dataset_train_input.map(tokenize_fn,
						                                 batched=True,
						                                 # 이걸 해주어야 'full_text", 'clean_abstr'이 사라져서 올바르게 패딩이 들어감
						                                 remove_columns=dataset_train_input.column_names,
						                                 )

print(token_model_inputs)
"""
Dataset({
    features: ['full_text', 'clean_abstr', 'input_ids', 'attention_mask', 'labels'],
    num_rows: 10
})
"""

```

`truncation=True`는 토크나이저가 생성한 토큰 시퀀스가 모델이 허용하는 최대길이(max_length)를 넘으면 잘라서 버린다.

<br>

`text_target`

정답 텍스트를 넣어준다.

<br>

`remove_columns=dataset_train_input.column_names`

이걸 해주어야 'full_text", 'clean_abstr'이 사라져서 올바르게 패딩이 들어감

<br>

### set_format()

토크나이저로 나온 데이터들이 list 형태로 나온다.

데이터셋을 모델에 학습하려면 tensor 형태여야한다.

`set_format()`으로 바꿔주면된다.

`input_ids", "token_type_ids", "attention_mask", "label”` 이 tensor형태로 나온다.

```python
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
"""
{
  "input_ids": Tensor([...]),
  "attention_mask": Tensor([...]),
  "labels": Tensor([...])    # 혹은 "label"
}
"""
```

<br>

**토큰화 확인**

```python
ids = train_token_inputs['input_ids'][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(ids)

pd.DataFrame(
    [
      ids, tokens
    ], index=('ids', 'tokens')
)
```

|            | **0** | **1** | **2** | **3**     | **4** | **5** | **6**     | **7** | **8** | **9** | **...** | **502** | **503** | **504** | **505** | **506** |
| ---------- | ----- | ----- | ----- | --------- | ----- | ----- | --------- | ----- | ----- | ----- | ------- | ------- | ------- | ------- | ------- | ------- |
| **ids**    | 19531 | 157   | 97    | 1868      | 77    | 556   | 26928     | 7     | 21    | 23253 | ...     | 0       | 0       | 0       | 0       | 0       |
| **tokens** | ▁박재 | 원    | ▁기자 | ▁대한민국 | ▁5    | G     | ▁홍보대사 | ▁     | 를    | ▁자처 | ...     | <pad>   | <pad>   | <pad>   | <pad>   | <pad>   |

<br>

## Model

모델을 확인해보면 임베딩 차원과 어휘 사전 크기를 알 수 있다.

```python
model
#  (lm_head): Linear(in_features=768, out_features=64128, bias=False)
```

가장 마지막 분류 헤드에 벡터는 768차원이고 출력 차원이 어휘사전 크기인 64128이 된다.

<br>

### 포워딩 테스트

커스텀 모델이 아닌, 이미 만들어져 있는 모델은 아키텍처와 출력이 의도대로 나오는지 확인해야한다.

이런 작업은 모델이 제대로 작동하는지, 어떤 구조로 되어있는지 쉽게 이해할 수 있다.

<br>

영어 문장을 예제로 만들어서 테스트 해본다.

```python
eng = ["i love to travel abroad."]
ko = ["나는 해외 여행을 좋아한다."]

encoder_inputs = tokenizer(eng, return_tensors="pt")['input_ids'].to(deivce)
decoder_targets = tokenizer(ko, return_tensors="pt")['input_ids'].to(deivce)
```

<br>

영어 문장은 인코더의 입력(`encoder_inputs`)이 되어 attention 작업을 거치게 된다.

→ 영어 문장 축약 정보 생성

<br>

한국어 문장은 디코더의 타겟(`decoder_targets`)이 되어 attention 작업으로 학습을 하게된다.

디코더는 teacher forcing을 위해서 입력 값(`decoder_inputs`)도 필요하다.

<br>

**teacher forcing**

디코더에서 이전 단어들을 참조하여 다음 단어를 예측한다.

하지만 학습 초기에는 모델이 생성하는 단어들이 대부분 틀릴 가능성이 높다.

그 다음 단어도 이전 단어들을 보고 학습을 하게 될때, 틀린 단어들이 가득한 단어들로 지속적으로 학습하게 된다.

이 상태에서 계속해서 예측된 단어를 기반으로 학습하게 되면, **잘못된 예측이 누적되면서 학습이 왜곡**될 수 있다.

이를 방지하기 위해 **teacher forcing**을 사용한다.

예측된 단어(토큰) 대신, 실제 정답 단어(target)을 다음 입력으로 사용하여 학습한다.

<br>

그럼 다음 타임스텝에서 이전 단어들을 참조해서 다음 단어를 예측할때 올바른 단어를 바탕으로 예측해 학습할 수 있다.

모델이 실제로 예측한 단어(토큰)은 다음 입력으로는 사용되지 않고, **loss 계산에만 활용**된다.

<br>

`decoder_inputs` 생성 방법은 `model._shift_right`를 사용해서 `decoder_target` 인덱스를 오른쪽으로 한칸씩 이동시킨다.(shift right)

```python
decoder_inputs = model._shift_right(decoder_targets)
```

<br>

오른쪽으로 이동시킨 결과는 `<pad>`가 제일 첫번째 토큰으로 생기게 된다.

마지막에 존재 했던 `</s>` SEP 토큰은 없어진다.

```python
pd.DataFrame(
    [
        tokenizer.convert_ids_to_tokens(decoder_inputs[0].tolist()),
        tokenizer.convert_ids_to_tokens(decoder_targets[0].tolist())
    ], index=('decoder_inputs_tokens', 'decoder_targets_tokens')
)
```

|                     | 0     | 1     | 2       | 3         | 4         | 5    |
| ------------------- | ----- | ----- | ------- | --------- | --------- | ---- |
| **decoder_inputs**  | <pad> | ▁나는 | ▁해외   | ▁여행을   | ▁좋아한다 | .    |
| **decoder_targets** | ▁나는 | ▁해외 | ▁여행을 | ▁좋아한다 | .         | </s> |

<br>

teacher forcing을 진행할때, `decoder_input`이 예측을 진행하며, 다음 토큰으로는 `decoder_targets`이 들어가게 된다.

```python
outputs = model(input_ids=encoder_inputs,
      labels=decoder_targets,
      decoder_input_ids=decoder_inputs)
```

<br>

output.keys는 아래와 같은 키들이 존재한다

```python
outputs.keys()
# odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])
```

<br>

손실 함수 값은 `outputs.loss`로 확인 할 수 있다.

```python
outputs.loss
# tensor(177.6366, grad_fn=<NllLossBackward0>)
```

<br>

인코더의 마지막 hidden state는 (1, 7, 768) 이다.

샘플 수, 토큰 수(타임 스텝 수), 모델 사이즈를 나타낸다.

즉, 인코더에 들어가는 토큰은 7개이며 768 차원 벡터로 인코딩 되어있다.

```python
outputs['encoder_last_hidden_state'].shape
# torch.Size([1, 7, 768])
```

<br>

디코더가 타임 스텝별로 다음 단어를 예측한다.

첫 시작은 `decoder_inputs`의 `<pad>` 토큰을 입력을 받아 다음 단어를 예측한다.

그 다음 타임스텝은 `decoder_inputs`의 다음 단어를 입력으로 받아 또 다음 단어를 예측하며 반복한다.

<br>

가장 마지막 타임스텝에서는 `decoder_target`이 마지막 단어가 `</s>` 이다보니, 학습한 뒤을 마치면 마지막 단어를 `</s>`으로 예측하게 된다.

모든 단어를 예측을 하게되면 6개의 토큰이 나오게 된다.

`decoder_inputs`은 정답 시퀀스의 길이와 동일하게 6개이고 그 6개의 입력에 대해 각각 다음 토큰을 예측하니깐 6개 만큼의 예측 토큰을 만든다.

<br>

샘플(문장) 한개가 가지고 있는 6개의 토큰이 64128개 단어에 대한 확률 값이 들어 있다.

```python
outputs['logits'].shape
# torch.Size([1, 6, 64128])
```

<br>

logit에 argmax를 씌워서 토큰화 시켜보면 다음과 같다.

```python
tokenizer.convert_ids_to_tokens(torch.argmax(outputs['logits'][0], axis=1).cpu().numpy())
# ['OVER', '▁카이버', '▁카이버', '▁선후배', '▁카이버', 'OVER']
```

마지막 헤더가 학습 되지 않았기 때문에 적절한 아웃풋이 나오지 않았다.

입력과 출력의 텐서 모양을 보면 포워드 패스가 제대로 작동한것을 알 수 있다.

<br>

## Collator

데이터를 미니배치 형태로 모아주는 collator이다.

모델을 학습시킬떄는 `DataLoader`을 사용한다.

데이터로더를 통해 `for` 루프를 돌면서 데이터 셋으로 부터 샘플을 미니 배치 수 만큼 가져온다.

이때 샘플을 미니 배치 수 만큼 무작위로 가져온다.

서로 다른 크기의 샘플을 가져오게 되니 패딩을 통해 길이를 맞춰주는 작업이 필요하다. 이 작업을 `collate_fn` 으로 하게된다.

<br>

시퀀스 투 시퀀스 모델은 콜레이터 함수가

1. 입력 or 출력 문자열을 패딩
2. 디코더 타겟을 오른쪽으로 한칸 쉬프트 시켜 디코더 입력 생성
3. 패딩 인덱스 -100 처리해 loss에 반영되지 않게함(tokenizer에서 `paddin=True` 넣어주면 -100안됨)
4. 파이썬 리스트에서 tensor로 변환해줌.

이런 작업을 자동으로 처리해주는 클래스는 `DataCollatorForSeq2Seq` 이다.

```python
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

<br>

`data_collator`에 들어갈려면 각 샘플별로 `input_ids`, `attention_mask`..등등 하나의 객체 형식으로 묶여 있어야한다.

```python
[{'input_ids': [4014, 322, 3170, 147, 67, 23274, 3, 1], # 문단1
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1],# 문단1
  'labels': [6842, 404, 951, 5767, 15387, 27, 831, 800, 4378, 15, 1587, 3, 1]},# 문단1
 {'input_ids': [11783, 4412, 96, 6556, 709, 1632, 3, 1], # 문단2
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1], # 문단2
  'labels': [9881, 18590, 3837, 70, 4341, 1086, 677, 35, 426, 2255, 3, 1]}] # 문단2

[
	# 문단1
	{'input_ids': [...], # 문단1
	  'attention_mask': [...],# 문단1
	  'labels': [...]}, # 문단1
	},
	# 문단2
	{'input_ids': [...], # 문단2
	  'attention_mask': [...],# 문단2
	  'labels': [...]}, # 문단2
	}
]

```

<br>

`data_collator`의 결과는 아래와 같다.

```python
data_collator(res)

{
		'input_ids':
				tensor([[   83,  2177,    10,  4366, 18904,     3,     1],
				        [   83,   297, 21122, 19989,     1,     0,     0]]),

		'attention_mask':
				tensor([[1, 1, 1, 1, 1, 1],
				      [1, 1, 1, 1, 1, 1]]),

		'labels':
				tensor([[  600,   722,  7545,  2932,  1008,  4919, 19779,  9763,  1168,    153,     1],
		            [  600,  1638, 10448, 16492,     3,     1,  -100,  -100,  -100,  -100, -100,  -100]])
		'decoder_input_ids':
				tensor([[    0,   600,   722,  7545, 16492,     3],
				      [    0,   600,  1638, 10448, 16492,     3]])
}
```

콜레이터 함수(`DataCollatorForSeq2Seq`)가 다음과 같은것을 자동으로 해준다고 했다.

1. 입력 or 출력 문자열을 패딩
2. 디코더 타겟을 오른쪽으로 한칸 쉬프트 시켜 디코더 입력 생성
3. labels 패딩 인덱스 -100 처리해 loss에 반영되지 않게함
4. 파이썬 리스트에서 tensor로 변환해줌.

<br>

→

<br>

1. 입력 문자열(`input_ids`)에 패딩을 주어 마지막에 0이 2개 들어가 있다.
2. `label`(디코더 타겟)을 오른쪽으로 한칸 쉬프트 시켜 `decoder_input_ids`을 생성했다.
3. lables 마지막에 패딩 인덱스 -100을 주어 loss에 반영되지 않게 했다.
4. 파이썬 리스트에서 tensor로 변환해줌.

<br>

### 패딩 인덱스 -100 주기위한 조건(중요)

tokenizer에서 토큰화을 해주었을때, `return_tensor=”pt”`, `padding=True` 을 해준다면, 이미 시퀀스에는 패딩이 들어가 있다.

패딩 인덱스 -100을 주는것은 콜레이터 함수가 -100 패딩을 직접 넣어주어야한다.

따라서 이미 존재하면 -100으로 바꿔주지 않는다.

<br>

따라서 tokenizer로 토큰화를 진행할때 `return_tensor=”pt”`, `padding=True` 을 빼주어 일반 파이썬 리스트 형태로 나오게 해야한다.

- 콜레이터 함수에서 패딩 -100과 tensor 형태로 바꿔주니 위의 작업을 tokenizer에서 할 필요가 없다.
- 또는 `set_format()`을 통해 tensor 형태로 바꿔주면된다.

<br>

## Metrix

모델이 올바르게 학습되었는지 평가한다.

번역 모델은 주로 BLEU 점수를 사용한다. 생성한 번역문장이 레퍼런스(실제) 문장과 얼마나 비슷한지 측정해 점수를 매긴다. 같은 단어가 반복되거나, 실제 문장보다 너무 짧으면 패널티를 부여한다. 따라서 레퍼런스 문장과 길이가 최대한 비슷하고 다양한 단어를 생성하면 점수가 높다.

[evaluate](https://huggingface.co/docs/evaluate/index), [sacarebleu](https://huggingface.co/spaces/evaluate-metric/sacrebleu) 라이브러리로 평가 할 수 있다.

[sacarebleu](https://huggingface.co/spaces/evaluate-metric/sacrebleu) 라이브러리는 BLEU의 표준 라이브러리이며 각 모델이 다른 토크나이저를 쓰는 경우 BPE로 통일 시켜서 BLEU 점수를 계산한다.

<br>

evaluate 라이브러리로 sacrebleu를 불러온다.

```python
!pip install evaluate
!pip install sacrebleu

import evaluate

metric = evaluate.load("sacrebleu")
```

<br>

아래와 같이 predictions과 references을 비교해서 score를 계산한다.

```python
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)

{'score': 46.750469682990165,
 'counts': [11, 6, 4, 3],
 'totals': [12, 11, 10, 9],
 'precisions': [91.67, 54.54, 40.0, 33.33],
 'bp': 0.9200444146293233,
 'sys_len': 12,
 'ref_len': 13}
```

<br>

Trainer에 사용하는 전처리 및 디코딩 작업을 해주는 헬퍼 함수를 만들어 주어야한다.

```python
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    return result
```

Trainer가 내부적으로 (preds. labels) 튜플을 넘긴다.

pred는 모델의 출력, labels는 학습시 사용된 정답 토큰 id들이다.

<br>

예측된 토큰 id을 문자열로 변환 시켜준다.

`[101, 1045, 2293, 2017, 102]` → `"i love you"`

`skip_special_tokens=True` 덕분에 `<pad>`, `<eos>` 등은 제거됨

<br>

라벨(labels)에는 -100이 들어 있어 그대로 `batch_decode()`하면 오류 나거나 이상한 결과가 나온다.
그래서 -100을 `tokenizer.pad_token_id`로 바꿔야 한다.

> skip_special_tokens=True 옵션 덕분에 pad 토큰은 자동으로 제거된다.

<br>

문자열 앞뒤 공백제거와 리스트 형식으로 감싸주는 후처리를 해준다.

마지막으로 metric.compute()을 해주어 BLEU를 계산해준다.

`{"bleu": result["score"]}` 와 같은 형태를 반환해 Trainer은 이 딕셔너리를 log에 기록하고 `eval_results`로 반환한다.

<br>

즉, compute_metrics 함수는 모델 예측 → 디코딩 → 전처리 → metric 계산 과정을 깔끔하게 묶어서 Trainer에서 사용할 수 있게 해주는 함수이다.

<br>

## Trainer

학습을 간단하게 해주는 `SeqSeqTrainer` 클래스를 사용할 수 있다.

`Seq2SeqTrainingArguments` 를 사용해서 학습 세부조건을 설정 할 수 있다.

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{root_path}/chkpt", # 체크포인트, 결과물 저장될 디렉토리(경로 설정가능)
    learning_rate=0.0005,            # 옵티마이저 학습률
    weight_decay=0.01,               # 가중치 감쇠
    logging_steps=1000,              # 100스텝마다 로그 남김
    num_train_epochs=1,              # 에폭수, 1 에폭만 학습
    save_strategy="steps",           # or "epoch"
    save_steps=10000,                # ➜ 10000 step마다 저장
    logging_strategy="steps",        # "epoch" or "steps"로 하면 로그 남김
    eval_strategy="steps",           # or "epoch"
    eval_steps=15000,                # 15000 step마다 평가함
    per_device_eval_batch_size=16,   # 검증 시 배치 크기,
    predict_with_generate=True,      # 평가/예측 시 generate() 사용해 결과 생성(요약, 번역, 생성 등에서 필수)
    per_device_train_batch_size=8,   # 학습 배치 크기
    fp16=True,                       # float16 연산 사용 여부, True 하면 GPU 메모리 절약 및 속도 증가, False 하면 float32
    report_to="none",                # Wandb 로그 끄기 # 로그 리포팅 대상, "none"으로 하면 로그 사용안함(default: WandB, TensorBoarc 등)
    save_total_limit=2,              # 최근 2개의 체크포인트만 남김, 나머지 삭제
    gradient_accumulation_steps=2    # 배치 크기를 2배 늘려준다.
)
```

<br>

`gradient_accumulation_steps`

여러 개의 미니배치에 대한 손실(gradient)을 누적해서

한 번의 `optimizer.step()`(= 가중치 업데이트)을 수행하는 설정

메모리 적게 쓰면서도 batch size 늘리기

<br>

training_args를 만들고 아래와 같이 trainer를 생성한다.

지금 까지 준비했던 `model`, `training_args`, `tokenized_datasets`, `data_collator`, `tokenizer`, `compute_metrics` 를 전달해준다.(주의: 토큰화된 데이터셋 넘겨주어야함)

```python
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=token_model_inputs,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 학습 종료후 결과 저장
trainer.save_model("./results")
```

<br>

train_dataset은 세 가지 형식을 모두 지원한다.

- `torch.utils.data.Dataset`: 일반적인 PyTorch Dataset (indexing 가능: `__getitem__`, `__len__`)
- `torch.utils.data.IterableDataset`: 순회만 가능한 Dataset
- `datasets.Dataset`: Hugging Face의 `datasets` 라이브러리에서 제공하는 고수준 객체

<br>

`datasets.Dataset`인 경우, 모델의 `forward()` 함수에 맞지 않는 컬럼은 자동으로 제거됨

즉, `input_ids`, `attention_mask`, `labels` 등만 남기고 나머지는 무시함

<br>

팁

학습 중 GPU RAM이 OOM 될 수 있다.

- batch size 줄이기
- fp16=True
- `eval_dataset=converted_train_token_inputs` 사용 안하기(그냥 안넣으면됨)
- 더 작은 모델 사용 t5-base → t5-small

<br>

OOM이 되면 위의 방법을 사용했을때 적용이 안될것이다.

이미 학습시에 사용한 데이터들이 GPU RAM 있기때문이다.

따라서 비워줘야한다.

```python
import torch
import gc

gc.collect()                 # Python 객체 메모리 해제
torch.cuda.empty_cache()     # PyTorch 캐시 비우기
torch.cuda.ipc_collect()     # CUDA inter-process 캐시 정리
```

<br>

## Test

학습과 저장을 마치고 모델을 다음과 같이 부를수있다.

```python
model_dir = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

model.cpu();
```

<br>

토큰화

추론용 데이터는 반드시 Tensor 타입 이여야한다.

또한 학습을 위한 text_target도 필요없다.

```python
input_text = [
    "Because deep learning frameworks are well developed, in these days, machine translation system can be built without anyone's help.",
    "This system was made by using HuggingFace's T5 model for a one day"
]

def inference_tokenize_function(example):
    model_inputs = tokenizer(
        example["text"],
        max_length=512,
        truncation=True,
    )

    return model_inputs

test_datset = Dataset.from_list(input_text)
tokenized_datasets = test_datset.map(inference_tokenize_function, batched=True, remove_columns=test_datset.column_names)

val_loader = DataLoader(
    tokenized_datasets,
    batch_size=8,               # 메모리에 맞게 조절
    collate_fn=data_collator
)
```

DataLoader을 통해서 학습시 사용했던 배치사이즈 만큼 추론하게 한다.(일치해야함)

<br>

추론

`model.generate()` 메서드로 추론 한다.

```python
model.eval()
with torch.no_grad():
    for batch in tqdm(val_loader):
        batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}

        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=5
        )
```

`torch.Tensor.requires_grad`의 `defualt=True` 이다.

`torch.no_grad()`, `model.eval()`을 사용해 추론에서는 연산 그래프를 사용하지 않게해 더 빠른 추론이 되게한다.

항상 추론 단계에서 새로운 모델을 불러왔을때, model의 deivce 설정과 input의 device를 통일 시켜야함.

<br>

`num_beams=5`는 "빔 서치(Beam Search)"를 쓸 때 탐색 후보 수를 5개로 하겠다는 뜻이다.

<br>

```python
다음 가능한 토큰: ["you", "it", "this", "dogs", "cats"]
확률:             [0.4 , 0.3 , 0.1 , 0.1  , 0.1]
```

- `num_beams=1`이면: `"you"` 선택 → **그대로 계속**
- `num_beams=5`이면:

→ 확률 높은 5개를 동시에 살린다.

→ 각각의 경우에 대해 다음 토큰도 또 5개씩 확장해서 트리 구조 탐색한다.

→ 최종적으로 가장 높은 점수를 갖는 문장 경로를 선택한다.

<br>

`num_beams`가 크면 성능이 향상된다.

하지만 그만큼 연산량이 증가해 느려지고 메모리가 증가한다.

<br>

`tokenizer.batch_decode`을 사용해서 한글로 바꾸어 번역이 잘되었는지 확인한다.

```python
preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

for inp, pred in zip(inputs, preds):
    print('')
    print(f"입력 문장: {inp}")
    print(f"예측 요약: {pred}")
    print("-" * 50)
```

`skip_special_tokens=True`
→ `<pad>, <eos>, <s>` 같은 특수 토큰 제거

<br>

이후에는 `DataCollatorForSeq2Seq`나 `Seq2SeqTrainer` 를 직접 구현해서 더 상세히 이해해봐야겠다.

<br>

참고

- [https://metamath1.github.io/blog/posts/gentle-t5-trans/gentle_t5_trans.html?utm_source=pytorchkr&ref=pytorchkr](https://metamath1.github.io/blog/posts/gentle-t5-trans/gentle_t5_trans.html?utm_source=pytorchkr&ref=pytorchkr)
- [https://huggingface.co/learn/llm-course/chapter7/4?fw=pt#metrics](https://huggingface.co/learn/llm-course/chapter7/4?fw=pt#metrics)
- [https://wikidocs.net/166817](https://wikidocs.net/166817)
- [https://huggingface.co/docs/evaluate/transformers_integrations](https://huggingface.co/docs/evaluate/transformers_integrations)
- [https://huggingface.co/lcw99/t5-base-korean-text-summary](https://huggingface.co/lcw99/t5-base-korean-text-summary)
