- [\[코드 분석\]Self-Attention](#코드-분석self-attention)
  - [전체 흐름](#전체-흐름)
  - [입력 임베딩](#입력-임베딩)
  - [Q, K, V 생성](#q-k-v-생성)
    - [예시)](#예시)
  - [유사도 계산](#유사도-계산)
    - [Q, K 유사도 계산](#q-k-유사도-계산)
    - [Scaling(스케일링)](#scaling스케일링)
    - [softmax 적용](#softmax-적용)
    - [가중합(어텐션 가중치, V)](#가중합어텐션-가중치-v)
- [Mult-Head-Attention](#mult-head-attention)

# [코드 분석]Self-Attention

모든 입력 토큰들이 전체 입력 시퀀스(입력이 두 문장이 될 수 있음)의 토큰들을 모두 참조해서 문맥 기반 표현을 얻는다.

<br>

## 전체 흐름

1. 입력 임베딩
2. Q, K, V 생성
3. 어텐션 스코어 계산
4. 소프트맥스 적용
5. 어텐션 가중치 x V

<br>

## 입력 임베딩

## Q, K, V 생성

Q(Query, 질문): 내가 알고 싶은 기준

K(Key): 다른 토큰들의 특징

V(Value): 그 토큰이 실제로 갖고 있는 정보

<br>

Self-Attention에서 Q, K, V(Query, Keys, Value)는

어텐션이 “무엇을 얼마나 보고, 어떻게 반영할지”를 결정하는 데 꼭 필요한 역할이다.

<br>

### 예시)

문장: `"The dog chased the cat"`

→ 토큰: `[The, dog, chased, the, cat]`

입력 단어 임베딩 벡터

| 토큰   | 임베딩 벡터 (4차원)    |
| ------ | ---------------------- |
| The    | `[1.0, 0.0, 1.0, 0.0]` |
| dog    | `[0.0, 2.0, 0.0, 2.0]` |
| chased | `[1.0, 1.0, 1.0, 1.0]` |

<br>

입력 단어 임베딩을 통해 Q, K, V 생성한다.

| 토큰   | Q                 | K                | V                |
| ------ | ----------------- | ---------------- | ---------------- |
| The    | `[ 0.31, -0.22 ]` | `[-0.05, -1.34]` | `[ 0.52,  0.23]` |
| dog    | `[-0.86,  0.48 ]` | `[ 1.12, -0.26]` | `[-0.08, -1.35]` |
| chased | `[-0.28,  0.13 ]` | `[ 0.53, -0.80]` | `[ 0.22, -0.56]` |

<br>

각각 Q, K, V를 위해 선형변환으로 가중치 행렬을 만든다.

```python
W_Q = nn.Linear(input_embed_dim, input_embed_dim)
W_K = nn.Linear(input_embed_dim, input_embed_dim)
W_V = nn.Linear(input_embed_dim, input_embed_dim)

Q = W_Q(X)  # [B, T, D]
K = W_K(X)  # [B, T, D]
V = W_V(X)  # [B, T, D]
```

- `input_embed_dim`: 입력 임베딩 차원 (예: 768 for BERT-base)
- `qk_dim`: Query와 Key의 출력 차원 (보통 head당 차원 × head 개수)
- `v_dim`: Value의 출력 차원 (보통 qk_dim과 같음)

<br>

## 유사도 계산

### Q, K 유사도 계산

각 단어는 자신을 포함한 모든 토큰(단어)의 K와 내적합을 계산해 유사도(score)를 계산한다.

자기 자신을 포함하는 이유는 다른 단어만 보면 왜곡 가능성이 생기 때문이다.

```python
score₁ = Q · K₁ = (0.31)*(1.12) + (-0.22)*(-0.26)
       = 0.3472 + 0.0572 = 0.4044

score₂ = Q · K₂ = (0.31)*(0.53) + (-0.22)*(-0.80)
       = 0.1643 + 0.1760 = 0.3403
```

score_1이 score_2보다 유사도가 더 크다. 따라서 K1인 “dog” 쪽에 더 많은 attention을 준다.

<br>

### Scaling(스케일링)

dot product(내적)은 차원이 클수록 값이 커져서 softmax가 너무 작은 gradient를 생성한다.

이를 방지하기위해 각 헤드에서의 차원수를 root하여 나눠준다.

(여기서 dk는 각 헤드에서의 차원 수. BERT-base에선 64)

```python
scaled_scores = Q · Kᵀ / √dₖ # -> [score1, score2,...]
```

<br>

### softmax 적용

score들을 softmax에 넣어 어텐션 가중치로 바꿔준다.

```python
attention_weight = softmax([score₁, score₂, ...])
# attention_weight = [0.2, 0.5, 0.3]
```

각 토큰(단어)별로 확률 값이 구해진다.

즉, 현재 토큰이 다른 토큰에 얼마나 attention 하는지를 구한다.

<br>

### 가중합(어텐션 가중치, V)

```python
output = attention_weight₁ * V₁ + attention_weight₂ * V₂ + ...

# attention_weight = [0.2, 0.5, 0.3]
# output = 0.2*[0.1, 0.3] + 0.5*[0.2, 0.4] + 0.3*[0.0, 0.5]
#       = [0.02, 0.06] + [0.10, 0.20] + [0.00, 0.15]
#       = [0.12, 0.41]   ← 최종 output은 벡터 (2차원)
```

어텐션 가중치와 V와 가중합을 해준다.

가중합: 각 요소별로 곱한뒤 모든 요소를 더해준다.

<br>

스칼라 x 벡터 → 벡터

벡터들의 합 → 동일한 차원의 벡터

즉, V 차원과 같은 차원의 shape가 된다.

<br>

최종적으로 각 토큰(단어) 별로 현재 시퀀스의 문맥이 반영된 벡터 표현(output)을 얻는다.

<br>

# Mult-Head-Attention

![image.png](Self-Attention,%20Mult-Head-Attention%20222880186a9f80e28bdac28e1522ff8a/image.png)

Self-Attention을 병렬로 h번 학습시키는 것이 Multi-Head-Attention 이다.

- Query, Key, Value 생성부터 어텐션 가중치 계산, 가중합까지 전부 통째로 여러 번 반복

```python
입력 X →
      ┌──────────────────────────────┐
      │     Multi-Head Attention     │
      │ ┌─────┐  ┌─────┐  ┌─────┐     │
      │ │Head1│  │Head2│  │Head3│ ... │  ← 각각 독립된 Self-Attention
      │ └─────┘  └─────┘  └─────┘     │
      │     ↓ concat (출력 이어붙임) │
      │     ↓ Linear Layer           │
      └──────────────────────────────┘
```

한개의 어텐션 헤드는 단어 사이의 관계를 하나의 관점으로만 본다.

여러개의 헤드를 사용하여 다양한 관점에서 단어들 관계를 동시에 볼 수 있다.

그래서 문장의 이해가 훨씬 깊어지고, 복잡한 언어 구조를 더 잘 학습한다.

<br>

예시)

문장 - `"나는 학교에 갔다. 거기에서 친구를 만났다."`

- Head 1: `"나는"` ↔ `"갔다"` 를 강하게 연결 → 주어-동사 관계에 집중
- Head 2: `"학교"` ↔ `"거기"` 연결 → 장소 대명사 해석에 집중
- Head 3: `"친구"` ↔ `"만났다"` 연결 → 목적어-동사 관계
