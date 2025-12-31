- [BERT 전체 과정](#bert-전체-과정)
- [Input](#input)
    - [Token IDs](#token-ids)
    - [Segment IDs](#segment-ids)
    - [Position IDs](#position-ids)
- [Input Embedding(인풋 임베딩)](#input-embedding인풋-임베딩)
- [Encoder(인코더)](#encoder인코더)
  - [Multi-Head Self-Attention](#multi-head-self-attention)
    - [Q, K, V 생성](#q-k-v-생성)
    - [멀티-헤드 처리를 위해 분리](#멀티-헤드-처리를-위해-분리)
    - [각 Head에서 Scaled Dot-Product Attention 수행](#각-head에서-scaled-dot-product-attention-수행)
    - [Scaling](#scaling)
    - [어텐션 가중치 계산](#어텐션-가중치-계산)
    - [가중치 합 결과](#가중치-합-결과)
    - [헤드별 concat](#헤드별-concat)
    - [최종 선형 변환](#최종-선형-변환)
  - [Residual \& LayerNorm](#residual--layernorm)
  - [Feed-Forward Network (FFN)](#feed-forward-network-ffn)
  - [Residual \& LayerNorm](#residual--layernorm-1)
- [Output Representation(아웃풋 표현)](#output-representation아웃풋-표현)
- [BERT 구현 코드](#bert-구현-코드)
  - [Dataset](#dataset)

# BERT 전체 과정

<br>

1. input
   - Token IDs
   - Segment IDs
   - Position IDs
2. input embedding
3. Encoder
   1. Multi-Head Self-Attention
   2. Add & LayerNorm
   3. Feed-Forward Network (FFN)
   4. Add & LayerNorm
4. Output Representation

<br>

# Input

### Token IDs

문장을 토큰화 한뒤 각 토큰을 고유한 정수 ID로 변환한것

```python
Token IDs: [101, 2023, 2003, 1037, 7953, 102]
```

<br>

### Segment IDs

두 문장을 동시에 입력 받아서 구분하기 위해 토큰들이 어떤 문장에 속해 있는지에 대한 인덱스 정수

```python
Segment IDs: [0, 0, 0, 0, 0, 0]

# 문장 쌍이 주어졌다면 다음처럼 됨
입력: [CLS] A1 A2 A3 [SEP] B1 B2 [SEP]
세그먼트: 0   0  0  0   0     1  1   1
```

<br>

### Position IDs

말 그대로 토큰의 문장 내 위치정보를 나타내는 인덱스이다.

Transformer 기반 모델은 단어의 순서 정보를 반영하지 못한다.

즉, "나는 밥을 먹었다" vs. "밥을 나는 먹었다"를 입력하면 둘 다 동일하게 처리할 위험이 있다.

<br>

그래서 각 토큰이 문장에서 몇 번째인지 명시해주는 위치 인코딩이 필요하다.

```python
Input tokens:   [CLS] 나는 밥을 먹었다 [SEP]
Token IDs:         101  9521  7098  7824  6983   102
Position IDs:        0     1     2     3     4     5
```

<br>

# Input Embedding(인풋 임베딩)

각 3가지의 토큰을 임베딩해서 하나의 벡터로 만들어준다.

```python
InputEmbedding = TokenEmbedding + SegmentEmbedding + PositionEmbedding
```

| 임베딩 종류        | 설명                                       |
| ------------------ | ------------------------------------------ |
| Token Embedding    | 단어 ID → 의미 벡터로 변환                 |
| Segment Embedding  | 문장 A/B 구분용 임베딩                     |
| Position Embedding | 단어 순서를 반영하기 위한 위치 정보 임베딩 |

<br>

# Encoder(인코더)

BERT는 Decoder는 없고 Encoder만 존재한다.

<br>

각 인코더 블록은 다음 순서 같이 구성되어 있다.

1. Multi-Head Self-Attention
2. Add & LayerNorm
3. Feed-Forward Network (FFN)
4. Add & LayerNorm

이 블록을 12번(혹은 24번) 반복한다.

<br>

## Multi-Head Self-Attention

```python
input_embeddings: [batch_size, seq_len, hidden_size]
= [B, T, 768]   ← BERT-base 기준
```

인풋 임베딩이 인코더의 첫 블록에 들어간다.

<br>

### Q, K, V 생성

```python
Q = input_embeddings @ W_Q   # Query
K = input_embeddings @ W_K   # Key
V = input_embeddings @ W_V   # Value
```

여기서 `W_Q`, `W_K`, `W_V`는 각각 `[768, 768]` 크기의 학습 가능한 가중치 행렬이다.

즉, 인풋 임베딩 벡터 하나하나가 Q, K, V로 선형 변환(linear projection) 된다.

<br>

### 멀티-헤드 처리를 위해 분리

BERT-base는 12개 헤드이므로 hiddne size 768를 12개로 나눈다.

```python
Q: [B, T, 768] → [B, 12, T, 64]   # 64 = 768 / 12
K: [B, T, 768] → [B, 12, T, 64]
V: [B, T, 768] → [B, 12, T, 64]
```

<br>

### 각 Head에서 Scaled Dot-Product Attention 수행

유사도 계산

```python
attention_scores = Q @ K.transpose(-2, -1)  # shape: [B, T, T]
```

Query i번째 토큰과 Key j번째 토큰의 내적(dot product)을 구한다.

<br>

### Scaling

dot product 값이 커지면 gradient가 작아지는 문제 발생

방지 하기 위해 스케일링 적용

$d_k$: Key 벡터의 차원 (BERT-base에선 64)

$D = d_k = 64 → \sqrt{64} = 8$

```python
scaled_scores = attention_scores / math.sqrt(D)
```

<br>

### 어텐션 가중치 계산

```python
attn_weights = torch.softmax(scaled_scores, dim=-1)
```

shape[B, T, T] 각 토큰에 대해 얼마나 관계가 있는지 확률로 나타냄.

<br>

### 가중치 합 결과

```python
output = attn_weights @ V  # [B, T, T] @ [B, T, D] → [B, T, D]
```

attention weight를 기준으로 모든 Value 벡터를 가중합한다.

<br>

| Token i (Query) | Token j (Key) | Q·Kᵀ 값 | Softmax 후 Weight | Value V_j  | 곱합에 기여   |
| --------------- | ------------- | ------- | ----------------- | ---------- | ------------- |
| "나는"          | "나는"        | 1.5     | 0.45              | [0.2, 0.7] | [0.09, 0.315] |
| "나는"          | "밥을"        | 1.2     | 0.35              | [0.1, 0.2] | [0.035, 0.07] |
| "나는"          | "먹었다"      | 0.8     | 0.20              | [0.4, 0.6] | [0.08, 0.12]  |

<br>

### 헤드별 concat

```python
heads = [output₀, output₁, ..., output₁₁]  # 각각 [B, T, 64]
```

Head의 개수 만큼인 총 12개가 나오게 된다.

12개의 head 출력을 차원 기준으로 이어 붙인다.

```python
concat_output = torch.cat(heads, dim=-1)  # shape: [B, T, 768]
# [B, T, 64] + [B, T, 64] + ... ×12 → [B, T, 768]
```

<br>

### 최종 선형 변환

Head 들의 concat 결과를 다시 1개의 벡터로 통합하는 선형 계층을 통과한다.

```python
output = concat_output @ W_O + b_O  # W_O: [768, 768]
```

<br>

## Residual & LayerNorm

원래 문장 정보와 새로 계산한 정보(Multi-Head Attention 결과)를 더한다.(Residual)

새로운 정보를 참고하되, 기존 정보를 잊지 않게 된다.

<br>

너무 튀거나 이상한 값을 평평하게 정리해준다.(LayerNorm)

```python
# 임의의 예시 입력 텐서: X (배치=1, 시퀀스 길이=2, hidden size=4)
X = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]]])  # shape: [1, 2, 4]

# 서브 레이어 출력값 (예: Attention 결과)
sub_output = torch.tensor([[[0.5, 0.5, 0.5, 0.5],
                            [1.0, 1.0, 1.0, 1.0]]])  # shape: [1, 2, 4]

# Residual Add
added = X + sub_output

# LayerNorm 정의 (hidden size=4)
# [[[-1.3416, -0.4472, 0.4472, 1.3416],
#  [-1.3416, -0.4472, 0.4472, 1.3416]]]
layernorm = nn.LayerNorm(normalized_shape=4)

# LayerNorm 적용
output = layernorm(added)
```

<br>

## Feed-Forward Network (FFN)

Multi-Head Self-Attention은 “이 단어가 다른 단어들과 어떤 관련이 있는지” 토큰 간의 관계만 본다.

<br>

각 단어 자체의 의미를 깊게 처리하는 능력이 부족하다.

FFN은 Attention을 통해 주변 정보를 얻어 토큰 하나하나를 의미를 파악하게 해준다.

<br>

Attention 결과 값을 `Linear` → `ReLU` → `Linear` 을 거치게한다.

첫번째 Linear는 차원을 확장하여 더 많은 의미를 표현하게 한다.

ReLU는 비선형을 부여해서 복잡한 패턴을 학습 가능하게 한다.

두번째 Linear는 차원을 다시 줄여서 최종 출력을 한다.

```python
self.linear1 = nn.Linear(hidden_size, intermediate_size)
self.relu = nn.ReLU()
self.linear2 = nn.Linear(intermediate_size, hidden_size)

x1 = self.linear1(x)
x2 = self.relu(x1)
x3 = self.linear2(x2)
```

<br>

## Residual & LayerNorm

다시 한번 Residual & LayerNorm 과정을 거쳐 안정적이게 만든다

```python
class FeedForwardWithResidual(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # 1. FFN 통과
        x1 = self.linear1(x)
        x2 = self.relu(x1)
        x3 = self.linear2(x2)

        # 2. Residual Add
        added = x + x3

        # 3. LayerNorm
        out = self.layernorm(added)

        return out, x3, added
```

최종적으로 각 토큰에 대해 문맥을 반영한 벡터가 나온다.

<br>

# Output Representation(아웃풋 표현)

사전 학습시 [CLS] 토큰을 문장들의 전체 의미를 요약한다고 가정했다.

[CLS] 토큰 벡터 하나만으로도 문장 전체의 의미, 관계, 분위기 같은걸 파악할 수 있다.

<br>

예시)

```python
[CLS], I, love, you, ., [SEP]
```

여기 총 6개의 토큰 각각에 768차원 벡터가 각각 존재한다.

그런데 이 문장 하나만을 요약하고 싶다.

이때 [CLS]에 존재하는 정보만을 가져와 파악 할 수 있다.

<br>

[CLS] 토큰만 가져오는 코드는 아래와같다.

```python
cls_vector = last_hidden_state[:, 0, :]  # [batch, hidden_size]
```

`[:, 0, :]`

- `:` → 배치 다 가져오고
- `0` → <mark>**첫 번째 토큰**</mark>, 즉 `[CLS]` 위치
- `:` → 768차원 벡터 전체

즉, `[CLS]` 벡터 하나만 쏙 빼서 가져온다.

<br>

이후 [CLS] 벡터 하나만 뽑아서 분류기로 넘긴다.

```python
# batch_size, seq_len, hidden_size = 2, 10, 768
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states):
		    # 1) 첫 번째 토큰([CLS]) 벡터만 추출 → shape = [batch_size, hidden_size]
        cls_token = hidden_states[:, 0]

        # 2) 선형 변환 → shape = [batch_size, hidden_size]
        output = self.linear(cls_token)

        # 3) tanh 활성화 → 최종 pooled_output
        return self.tanh(output)

print(pooled_out.shape)  # → torch.Size([2, 768])
```

토큰별 문맥 벡터(`encoder_output`)

문장 전체 표현(`pooled_output`)

<br>

BERT 결과값을 바탕으로 다음과 같은 것도 답할 수 있다.

- 이 문장은 긍정이야, 부정이야?
- 이 문장은 스포츠야, 경제야, 정치야?
- 문장 A랑 B는 이어지는 내용이야? (NSP)
- 문장 A랑 B는 같은 의미야? (문장 유사도)
- 문서 안에서 핵심문장은 뭐야?

<br>

# BERT 구현 코드

```python
class BertConfig:
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 max_position_embeddings=512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        def shape(x):
            return x.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        query_layer = shape(self.query(hidden_states))
        key_layer = shape(self.key(hidden_states))
        value_layer = shape(self.value(hidden_states))

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, value_layer)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, config.hidden_size)
        return context

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attention_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.1),
            nn.LayerNorm(config.hidden_size)
        )
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Sequential(
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(0.1),
            nn.LayerNorm(config.hidden_size)
        )
        self.activation = nn.GELU()

    def forward(self, hidden_states, attention_mask=None):
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output(attn_output + hidden_states)
        intermediate_output = self.activation(self.intermediate(attn_output))
        layer_output = self.output(intermediate_output + attn_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0]
        return self.activation(self.dense(cls_token))

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

```

<br>

`BertConfig`

- 모델 전체의 하이퍼파라미터를 정의한다.
- 어휘 크기(`vocab_size`)
- 은닉 차원(`hidden_size`)
- 레이어 수(`num_hidden_layers`)
- 헤드 수(`num_attention_heads`)
- FFN 차원(`intermediate_size`)
- 최대 위치 임베딩 길이(`max_position_embeddings`) 등을 보관한다.

<br>

`BertEmbeddings`

- 입력 토큰 ID를 실제 벡터로 바꿔 주는 역할을 한다.
- 세 가지 임베딩을 더해서 최종 입력 임베딩을 만든다.
  1. 단어 임베딩 (`word_embeddings`)
  2. 위치 임베딩 (`position_embeddings`)
  3. 문장 구분 임베딩 (`token_type_embeddings`)
- 마지막에 LayerNorm과 Dropout을 적용해 안정화 및 정규화한다.

<br>

`BertSelfAttention`

- Multi-Head Attention 연산을 수행한다.
- 입력 벡터를 Query/Key/Value로 각각 선형 투영한 뒤,
  - `num_attention_heads` 개의 헤드로 나누어
  - 스케일드 점곱 어텐션을 계산하고
  - 다시 합쳐서 컨텍스트 벡터를 만든다.
- Dropout으로 어텐션 가중치를 일부 랜덤하게 제거해 학습을 안정화한다.

<br>

`BertLayer`

- 하나의 Transformer 인코더 블록을 구현한다.
- 내부 구성
  1. `BertSelfAttention` → Residual 연결 + LayerNorm
  2. Feed-Forward Network (`intermediate` → activation → `output`) → Residual + LayerNorm
- GELU 활성화를 사용해 비선형성을 추가한다.

<br>

`BertEncoder`

- `BertLayer` 블록을 `num_hidden_layers` 개만큼 순차적으로 쌓는다.
- 입력 임베딩이 첫 번째 블록에 들어가며, 각 블록의 출력을 다음 블록의 입력으로 사용하여 깊은 인코딩을 수행한다.

<br>

`BertPooler`

- 인코더 출력 가운데 `[CLS]` 위치(첫 번째 토큰)의 벡터만 추출한다.
- 이를 `nn.Linear(hidden_size, hidden_size)` 레이어와 `tanh` 활성화를 거쳐 “문장 전체 표현”으로 가공한다.
- 주로 문장 분류·관계 예측 등의 downstream 태스크에 사용된다.

<br>

`BertModel`

- 위의 모든 모듈을 하나로 묶은 최상위 클래스이다.
- 순서대로 실행
  1. `embeddings` → 입력 임베딩 생성
  2. `encoder` → 깊은 인코딩 수행
  3. `pooler` → 문장 표현(pooler_output) 생성
- `forward` 호출 시 `(encoder_output, pooled_output)` 두 가지를 반환한다.

<br>

## Dataset

```python
# Dummy dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, seq_len):
        self.input_ids = torch.randint(0, config.vocab_size, (num_samples, seq_len))
        self.token_type_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
        self.attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
        self.labels = torch.randint(0, num_labels, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'token_type_ids': self.token_type_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.labels[idx]
        }

dataset = RandomDataset(1000, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and head
model = BertModel(config)
classifier = nn.Linear(config.hidden_size, num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        encoder_output, pooled_output = model(input_ids, token_type_ids, attention_mask)
        logits = classifier(pooled_output)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
```
