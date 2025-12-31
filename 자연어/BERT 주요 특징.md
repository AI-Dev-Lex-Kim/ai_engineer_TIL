- [BERT 주요 특징](#bert-주요-특징)
  - [1. 완전한 양방향(bidirectional) 구조](#1-완전한-양방향bidirectional-구조)
    - [정리](#정리)
  - [2. 사전 학습 2가지 방식 적용](#2-사전-학습-2가지-방식-적용)
    - [Masked Language Modeling (MLM)](#masked-language-modeling-mlm)
    - [MLM이 전체 입력 토큰의 15%만 마스킹하는 이유](#mlm이-전체-입력-토큰의-15만-마스킹하는-이유)
    - [구현 코드](#구현-코드)
    - [Next Sentence Prediction (NSP)](#next-sentence-prediction-nsp)
    - [학습 방식](#학습-방식)
    - [NSP를 추가한 의도](#nsp를-추가한-의도)
    - [NSP가 downstream task에 주는 이점](#nsp가-downstream-task에-주는-이점)
    - [구현 코드](#구현-코드-1)
  - [3. 미세 조정으로 다양한 작업 활용](#3-미세-조정으로-다양한-작업-활용)
    - [결론](#결론)

# BERT 주요 특징

BERT는 대규모 말뭉치에서 양방향 문맥 정보를 학습하는 사전학습 기반 언어 모델로,

MLM과 NSP를 통해 일반적인 언어 이해 능력을 습득하고, 다양한 NLP 태스크에 파인튜닝으로 쉽게 적용된다.

<br>

BERT는 인코더만 존재하는 모델이다.

인코더만 존재하는 모델로서 문장, 토큰 이해에 최적화 된 인코더 모델이다.

아래와 같은 다운스트림 NLP 태스크를 적용할 수 있다.

1. 문장 분류
   - 감정 분석
   - 주제 분류
   - 스팸 탐지
2. 토큰 분류
   - 개체명 인식
   - 품사 태깅

등등 생성형 모델이 아닌 문장의 관계를 이해해 문장을 함축하는 모델이다.(분류)

<br>

아래는 BERT 주요 특징이다.

1. 완전한 양방향 구조
2. MLM(Masked Language Model)
3. NSP(Next Sentence Prediction)

<br>

## 1. 완전한 양방향(bidirectional) 구조

Left-to-Right Language Model (예: GPT)

```
입력: "나는 학교에"
모델이 예측할 단어: ?
모델이 볼 수 있는 정보: "나는 학교에" (왼쪽 → 오른쪽만)
```

- GPT류 모델은 현재 단어를 예측할 때 <mark>**이전 단어**</mark>들만 사용
- 문장의 오른쪽(미래)은 보지 못함.

<br>

Right-to-Left Language Model (예: ELMo 일부 구성)

```python
입력: "갔다 나는"
모델이 예측할 단어: ?
모델이 볼 수 있는 정보: "갔다 나는" (오른쪽 → 왼쪽)
```

- 일부 모델은 거꾸로 보지만, 여전히 <mark>**한 방향**</mark>으로만 처리함.

<br>

BERT는 Transformer의 <mark>**self-attention 구조**</mark>를 활용해 "<mark>**양쪽 문맥을 동시에**</mark>" 보고 단어를 이해합니다. 이것이 "<mark>**양방향성**</mark>"의 의미이다.

```python
입력: "나는 [MASK]에 갔다"
예측: [MASK] = "학교"

→ "나는", "에", "갔다" 모두를 활용해서 "학교"를 예측
```

<br>

| 항목           | 기존 LM (Unidirectional)                | BERT (Bidirectional)                              |
| -------------- | --------------------------------------- | ------------------------------------------------- |
| 문맥 이해      | 부분적 (한쪽 방향)                      | <mark>**완전한 양쪽 문맥**</mark>                              |
| 단어 의미 해석 | 모호할 수 있음                          | 더 정확한 의미 파악 가능                          |
| 예시           | "나는 [MASK]에 갔다" → [MASK] 예측 불가 | "나는 <mark>**학교**</mark>에 갔다" 가능                       |
| 적용 가능 영역 | 언어 생성 (GPT류)                       | 문장 분류, NER, QA 등 <mark>**이해 중심 태스크에 강력**</mark> |

### 정리

- <mark>**기존 LM**</mark>은 단어를 예측할 때 문장의 한쪽 방향만 참고함 → 문맥의 절반만 사용
- <mark>**BERT**</mark>는 한 단어를 예측할 때 <mark>**양쪽 문맥을 모두 사용**</mark>함 → 문장의 전체 정보를 고려
- 이는 <mark>**문맥 이해 능력 향상**</mark>, <mark>**정확한 의미 파악**</mark>, <mark>**다양한 NLP 태스크에 높은 성능**</mark>으로 이어짐

<br>

## 2. 사전 학습 2가지 방식 적용

### Masked Language Modeling (MLM)

문장 내에서 <mark>**일부 단어를 [MASK]로 가리고**</mark>, 그 가려진 단어를 맞추도록 훈련하는 방식

- 양방향 문맥을 활용하여 단어 의미를 <mark>**깊이 있게 이해**</mark>하게 만들기 위함.
- 기존 언어 모델(GPT 등)은 한 방향만 보고 단어를 예측하는데, BERT는 <mark>**좌우 모두**</mark> 봐야 함.

<br>

```python
입력: "나는 [MASK]에 갔다."
예측: [MASK] → "학교"
```

BERT는 "나는", "에", "갔다"라는 좌우 문맥을 동시에 보며 <mark>**학교**</mark>를 추론.

<br>

학습 방식

- 입력 문장에서 전체 토큰의 15%를 선택.
  - 그 중 80%는 `[MASK]`로 대체
  - 10%는 랜덤한 단어로 바꿈
  - 10%는 그대로 둠 (모델이 예측 정확성을 유지하도록)
- 예측 대상은 원래의 단어

<br>

### MLM이 전체 입력 토큰의 15%만 마스킹하는 이유

너무 많이 마스킹하면 문장 의미 훼손, 너무 적게하면 학습을 잘 못함

따라서 15%는 실험적으로 가장 효율적인 학습 성능을 보여줌.

<br>

10%보다 낮으면 학습이 느려지고, 최적 성능도 낮다.

20~30% 이상이면 문장 의미가 손상되어 예측 정확도가 하락한다.

<br>

마스킹된 토큰을 모두 [MASK]로 바꾸지 않는다.

80%는 실제 마스킹을 한다.

10%는 랜덤 토큰을 적용한다.

10%는 원래 단어를 유지한다.

```python
# 전체 시퀀스 중 15%만 MLM 손실에 사용
masking_rules = {
    "80%": "[MASK]",
    "10%": "random word",
    "10%": "original word 유지"
}
```

<br>

### 구현 코드

```python
import random

# ────────────────────────────────────────────────────────────
# 1) 마스킹 함수 정의
# ────────────────────────────────────────────────────────────
def mask_tokens(token_ids,           # List[int]: 입력 토큰 ID 리스트
                vocab_size,          # int     : 어휘 크기 (랜덤 토큰 생성용)
                mask_token_id,       # int     : [MASK] 토큰 ID
                pad_token_id,        # int     : [PAD] 토큰 ID
                mask_prob=0.15):     # float   : 마스킹 확률 (15%)
    """
    Args:
      token_ids    : [t0, t1, t2, ...] 형태의 원본 토큰 ID 리스트
      vocab_size   : 전체 어휘 수 (랜덤 토큰 샘플링에 사용)
      mask_token_id: [MASK] 토큰 ID
      pad_token_id : 패드 토큰 ID (마스킹 대상 제외)
      mask_prob    : 마스킹할 비율 (기본 0.15 = 15%)
    Returns:
      masked_input_ids : 마스킹된 토큰 ID 리스트
      labels           : 예측해야 할 원래 토큰 ID 리스트
                         (마스킹된 위치만 ID, 나머지는 -100)

		token_ids     = [2, 3, 0, 2, 5]   # 입력 토큰 ID 리스트
		vocab_size    = 6                 # 어휘 크기
		mask_token_id = 1                 # [MASK] 토큰 ID
		pad_token_id  = 0                 # [PAD] 토큰 ID
		mask_prob     = 0.15              # 마스킹 비율 15%

    """
    N = len(token_ids)
    # 1) labels 초기화: 원본 복사, 나중에 마스킹되지 않은 위치는 -100 처리
    labels = token_ids.copy()

    # 2) 마스킹할 인덱스 선정
    masked_indices = []
    for i in range(N):
        if token_ids[i] == pad_token_id:
            # 패드 토큰은 절대 마스킹하지 않음
            masked_indices.append(False)
        else:
            # 랜덤하게 mask_prob 확률로 True/False 결정
            masked_indices.append(random.random() < mask_prob)

    # 3) labels에서 마스킹되지 않은 위치는 -100으로 설정
    for i in range(N):
        if not masked_indices[i]:
            labels[i] = -100

    # 4) 마스킹된 위치에 대해 80/10/10 규칙 적용
    masked_input_ids = token_ids.copy()
    for i in range(N):
        if not masked_indices[i]:
            continue
        r = random.random()
        if r < 0.8:
            # 80% 확률로 [MASK] 토큰
            masked_input_ids[i] = mask_token_id
        elif r < 0.9:
            # 10% 확률로 랜덤 토큰
            masked_input_ids[i] = random.randrange(vocab_size)
        else:
            # 10% 확률로 원래 토큰 유지 (이미 token_ids[i] 상태)
            pass

    return masked_input_ids, labels

# ────────────────────────────────────────────────────────────
# 2) 데모 실행
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # (A) 어휘 및 특수 토큰 ID 정의
    vocab     = {'[PAD]':0, '[MASK]':1, 'the':2, 'dog':3, 'barked':4, 'loudly':5}
    pad_id    = vocab['[PAD]']
    mask_id   = vocab['[MASK]']
    vocab_sz  = len(vocab)

    # (B) 예제 토큰 시퀀스 (ID 리스트)
    #     예: "the dog barked loudly"
    orig = [vocab['the'], vocab['dog'], vocab['barked'], vocab['loudly']]

    # (C) 마스킹 수행
    masked_input, labels = mask_tokens(orig, vocab_sz, mask_id, pad_id)

    # (D) 결과 출력
    print("원본 토큰 IDs  :", orig)
    print("마스킹된 입력  :", masked_input)
    print("예측용 라벨    :", labels)

```

<br>

### Next Sentence Prediction (NSP)

두 문장이 주어졌을 때, <mark>**두 번째 문장이 첫 번째 문장의 다음 문장인지**</mark>를 예측하는 과제.

<br>

문장 간 관계(문맥 흐름, 연결성)를 학습하기 위해서이다.

질문-응답, 문서 요약 등 <mark>**문장 수준의 이해**</mark>가 필요한 태스크에 도움이 된다.

| 문장 A              | 문장 B                      | 정답    |
| ------------------- | --------------------------- | ------- |
| "나는 학교에 갔다." | "친구들을 만났다."          | IsNext  |
| "나는 학교에 갔다." | "로마는 이탈리아 수도이다." | NotNext |

<br>

### 학습 방식

- 50% 확률로 실제 다음 문장을 이어 붙임 (label: _IsNext_)
- 50% 확률로 무작위 문장을 붙임 (label: _NotNext_)
- BERT는 문장 A와 문장 B를 `[SEP]` 토큰으로 구분하고, `[CLS]` 토큰을 통해 관계를 예측

<br>

CLS 토큰(Classification 토큰)

문장의 맨 앞에 붙으며, 전체 문장(혹은 문장 쌍)의 <mark>**대표 임베딩 역할**</mark>을 한다.

```python
입력: [CLS] 나는 학교에 갔다. [SEP] 친구들을 만났다. [SEP]
↑ 이 토큰의 출력값을 가지고 NSP 결과 (IsNext or NotNext)를 예측함
```

<br>

SEP 토큰(Separator 토큰)

두 문장을 구분할 때 사용하다.

```python
입력:
[CLS] 나는 학교에 갔다. [SEP] 친구들을 만났다. [SEP]

→ 첫 번째 [SEP]: 문장1 끝
→ 두 번째 [SEP]: 문장2 끝 (문장 경계 명확하게)
```

```python
from torch.nn import CrossEntropyLoss
# [CLS] token output을 C라고 할 때, C를 이진 분류에 사용
logits = classification_layer(C)
loss = CrossEntropyLoss()(logits, labels)

```

<br>

BERT 입력: `[CLS] 문장 A [SEP] 문장 B [SEP]`

이 입력을 Transformer에 넣으면, `[CLS]` 위치에서 나오는 출력(hidden state)을 바탕으로

문장 B가 문장 A 다음인지 (`IsNext`) 아닌지 (`NotNext`)를 예측함.

| 항목        | MLM                  | NSP                     |
| ----------- | -------------------- | ----------------------- |
| 레벨        | 단어 수준            | 문장 수준               |
| 학습 내용   | 단어 의미, 문맥 파악 | 문장 간 관계, 문맥 흐름 |
| 결과        | 정확한 단어 임베딩   | 문장 이해력 강화        |
| 활용 태스크 | NER, 감성 분석 등    | QA, 문장 유사도 등      |

BERT는 이 두 과제를 통해 <mark>**단어+문장 수준의 문맥을 모두 학습한다.**</mark>

<br>

### NSP를 추가한 의도

기존 언어 모델은 문장 내 단어 간 관계는 잘 학습했지만, 두 문장 간의 관계는 전혀 학습하지 못했다.

<br>

BERT는 문장과의 관계, 연관성을 학습했다.

문장이 이어지는지 판단 할 수 있다.

문장관 논리적/의미적 관계를 파악한다. 또한 두 문장의 상호작용을 할 수 있다.

<br>

### NSP가 downstream task에 주는 이점

Downstream task란, 사전학습(pre-training)된 모델을 가져다가, 실제 우리가 풀고 싶은 NLP 문제에 적용하는 작업.

<br>

자연어 추론: 두 문장간 의미를 비교하고 추론하는 능력이 좋아진다.

QA: 문서 내 문장이 질문과 관계를 파악하는 능력이 좋아진다.

문장 유사도: 두 문장을 함께 이해하고 비교해서 두 문장의 유사도를 비교하는데 특화되어 있다.

<br>

### 구현 코드

```python
import torch
import torch.nn as nn

class BertConfig:
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, intermediate_size, max_position_embeddings):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

class BertForNextSentencePrediction(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.nsp_head = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = (self.token_embeddings(input_ids) +
             self.position_embeddings(positions) +
             self.segment_embeddings(token_type_ids))
        x = x.transpose(0, 1)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)
        cls_output = x[:, 0, :]
        logits = self.nsp_head(cls_output)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

# 데모 데이터 준비
input_ids = torch.tensor([[2,3,4,5],[2,3,6,7]])
token_type_ids = torch.tensor([[0,0,1,1],[0,0,1,1]])
attention_mask = torch.ones_like(input_ids)
labels = torch.tensor([1,0])

# 모델 초기화 및 실행
config = BertConfig(vocab_size=10, hidden_size=8, num_heads=2,
                    num_layers=2, intermediate_size=32, max_position_embeddings=10)
model = BertForNextSentencePrediction(config)
loss, logits = model(input_ids, token_type_ids, attention_mask, labels)

print("NSP Loss:", loss.item())
print("NSP Logits:", logits)
print("Predictions:", torch.argmax(logits, dim=1))

```

<br>

## 3. 미세 조정으로 다양한 작업 활용

미세조정 (Fine-tuning)

사전학습된 BERT를 <mark>**기초 모델로 사용한다.**</mark>

각 <mark>**NLP 태스크에 맞게 소량의 데이터로 재학습한다.**</mark>

입력 형식은 거의 같고, <mark>**출력층만 교체**</mark>하는 방식을 한다.

| 태스크                   | 출력층 구성                          | 입력 예                         |
| ------------------------ | ------------------------------------ | ------------------------------- |
| 문장 분류 (감성 분석 등) | `[CLS]` → Linear → Softmax           | `[CLS] 문장 [SEP]`              |
| 문장 관계 (NSP/NLI)      | `[CLS]` → Linear → Softmax           | `[CLS] 문장1 [SEP] 문장2 [SEP]` |
| 개체명 인식 (NER)        | 각 토큰 → Linear → Softmax (BIO tag) | `[CLS] 문장 [SEP]`              |
| 질문응답 (SQuAD)         | 각 토큰에 대해 시작/끝 위치 예측     | `[CLS] 질문 [SEP] 문서 [SEP]`   |

<br>

이런 방식이 좋은 이유

공통 모델을 바탕으로 태스크별 미세조정을 한다.

덕분에 적은 데이터로도 학습이 가능하며 사전학습으로 깊은 문맥을 이해할 수 있다.

<br>

### 결론

- BERT는 최초로 <mark>**완전 양방향 Transformer**</mark>를 사전학습한 언어 모델.
- <mark>**MLM**</mark>과 <mark>**NSP**</mark> 두 가지 과제를 통한 사전학습이 핵심.
- 사전학습된 BERT는 다양한 NLP 태스크에 거의 <mark>**추가 파라미터 없이 바로 적용 가능**</mark>.
- GLUE, SQuAD 등에서 <mark>**기존 최고 성능을 초월**</mark>하는 결과를 보여줌.
