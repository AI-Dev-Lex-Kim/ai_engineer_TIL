# [NLP 논문 리뷰] Attention is all you need(Transformer)

- [\[NLP 논문 리뷰\] Attention is all you need(Transformer)](#nlp-논문-리뷰-attention-is-all-you-needtransformer)
  - [Abstract](#abstract)
  - [1. Introduction](#1-introduction)
  - [2 Background](#2-background)
  - [3. Model Architecture](#3-model-architecture)
    - [3.1 Encoder and Decoder Stacks](#31-encoder-and-decoder-stacks)
    - [3.2 Attention](#32-attention)

---

논문 링크: https://arxiv.org/pdf/1706.03762

<br>

## Abstract

기존의 인코더와 디코더를 포함한 모델들은 순환 신경망(RNN)이나 합셩곱 신경망(CNN)에 기반해 있었다.

이 논문에서 <mark>**Tansformer라는 모델은 순환과 합성곱을 완전히 제거하고, 오직 어텐션 메커니즘에만 기반해 새롭고 간단한 네트워크 구조를 제안**</mark>한다.

Transformer 모델은 <mark>**품질면에서도 우수**</mark>하고 병렬화가 훨씬 용이해서 <mark>**학습 시간이 굉장히 짧다**</mark>고 한다.

<br>

## 1. Introduction

순환 신경망(RNN)과 장단기 메모리(LSTM) 그리고 게이티드 순환 신경망(GRU)는 언어 모델링과 기계 번역과 같은 시퀀스 모델링 및 변환 문제에서 SOTA를 기록해왔다.

이러한 모델들은 시퀀스 위치를 계산 시간 단계에 맞추어 정렬하고, 이전 은닉 상태(ht-1)과 현재 위치(t)의 입력을 은닉 상태(ht) 시퀀스를 생성한다.

이러한 <mark>**필연적인 순차적인 특성은 학습시 병렬화를 불가능**</mark>하게 하며 <mark>**시퀀스 길이가 길어 질수록 이 문제가 심각**</mark>해 진다. 메모리 제약으로 인해서 여러 <mark>**example간의 배치 처리가 제한**</mark>되기 때문이다.

Attention 메커니즘은 <mark>**이전에도 다양한 작업**</mark>에서 시퀀스 모델링 및 변환 모델의 핵심적인 구성요소로 사용했었다.

<mark>**입력 또는 출력 시퀀스 내에서의 거리와 관계없이 의존 관계를 모델링**</mark> 할수있었기 때문이다.

하지만 이러한 어텐션 메커니즘은 <mark>**대부분 순환신경망(RNN)과 함께 사용**</mark>되었다.

이 논문에서는 Transformer를 제안하며, <mark>**순환 구조를 완전히 배제하고 입력과 출력 사이의 global 적인 의존 관계를 학습하기 위해서 오직 어텐션 메커니즘에만 의존하는 모델구조를 설계**</mark>했다.

Transformer는 <mark>**훨씬 더 높은 수준의 병렬화를 가능**</mark>하게 해주며, 이 논문의 실험에서 단 8개의 P100 GPU에서 12시간 정도만 학습해도 번역 품질에서 SOTA에 도달 할 수 있었다고 한다.

<br>

## 2 Background

시퀀스 모델링에서 순차적 계산을 줄이려는 목표로 Extended Neural GPU, ByteNet, ConvS2S 모델링이 제안되어왔다.

이 모델들은 <mark>**합성곱 신경망(CNN)을 기본 구성 요소로 사용해, 모든 입력 및 출력 위치에 대해 은닉 표현을 병렬로 계산하여 해결**</mark>했다.

하지만, 두 임의의 입출력 위치간의 신호를 연결하는데 필요한 연산 수는 <mark>**위치 간의 거리와 함께 증가해 먼 위치간의 의존 관계를 학습하는 것이 어려웠다.**</mark>

Transformer는 이러한 <mark>**위치간의 연산 수가 일정한(constant) 수준으로 줄어든다.**</mark>

<br>

## 3. Model Architecture

![논문리뷰1](../images/논문리뷰/Attention%20is%20all%20you%20need/1.png)

Transformer는 인코더와 디코더 구조를 가지고 있다.

<mark>**인코더**</mark>는 <mark>**입력 시퀀스 토큰(논문에서는 symbol이란 표현을 사용한다.)들인 x1, …, xn 들이 각 토큰들의 은닉상태 값을 가지고 있는 (z = z1, …, zn)으로 출력**</mark>되어진다.

<mark>**디코더**</mark>는 인코더가 생성한 은닉상태 값들인 z가 주어지면, <mark>**다음 예측할 토큰을 한번에 하나의 토큰씩 생성한다.**</mark>(y1, …, ym)

이러한 인코더와 디코더의 과정전체를 한번의 스텝으로 치자면, <mark>**각각의 스텝별로 auto-regressive(자기 회귀) 방식으로 동작**</mark>한다.

<mark>**auto-regressive 방식이란 모델이 다음 출력을 생성할때, “이전 시점에 생성한 출력”을 입력으로 사용하는 방식을 의미**</mark>한다.

다음 출력을 생성할때, 이전에 생성된 토큰들을 추가 입력으로 사용한다.

Transformer는 이러한 메커니즘으로 구성되어 있으며, <mark>**인코더와 디코더 모두에 층층이 쌓인 Self-Attention과 Point-wise fully connected layer을 사용한다.**</mark>

여기서 <mark>**Point-wise fully connected layer란 각 토큰(위치)마다 독립적으로 fully connected layer가 적용**</mark>되다는 의미이다.

예를 들어 시퀀스 길이가 n일때, FFN(feed-forward network)층을 <mark>**각 토큰 벡터에 독립적으로 적용해 시퀀스 길이 n에는 영향을 주지 않고 토큰별로 동일한 FFN를 적용하는 구조**</mark>이다.

Fully connected layer = MLP (Multi-Layer Perceptron) = 선형 변환 + 비선형(ReLU)

<br>

### 3.1 Encoder and Decoder Stacks

> 인코더(Encoder)는 <mark>**N=6개의 동일한 층들로 구성된 스택(stack)이다.**</mark> 각 층은 <mark>**두 개의 서브 레이어**</mark>(sub-layer)으로 이루어져 있다. <mark>**첫 번째는 다중 헤드 셀프 어텐션(multi-head self-attention) 메커니즘**</mark>이고, 두 번째는 간단한 <mark>**위치별 완전연결 피드포워드(position-wise fully connected feed-forward) 네트워크**</mark>이다.
>
> <mark>**각 두 개의 서브 레이어 주변에는 잔차 연결(residual connection) 을 적용**</mark>하며, <mark>**그 뒤에 층 정규화(layer normalization) [1]를 수행**</mark>한다. 즉, <mark>**각 서브 레이어의 출력은 LayerNorm(x + Sublayer(x)) 형태**</mark>이며, 여기서 <mark>**Sublayer(x)는 해당 서브 레이어이 수행**</mark>하는 함수이다.
>
> 이러한 <mark>**잔차 연결을 용이하게 하기 위해, 모델의 모든 서브 레이어들과 임베딩 층(embedding layer)은 출력 차원 d_model = 512를 갖는다.**</mark>

<br>

> 디코더(Decoder) 또한 **N=6개의 동일한 층들로 구성**되어 있다. <mark>**각 인코더 층의 두 개의 서브 레이어에 더해, 디코더는 세 번째 서브 레이어을 추가**</mark>하는데, 이는 <mark>**인코더 스택의 출력을 대상으로 다중 헤드 어텐션(multi-head attention)을 수행**</mark>한다.
>
> 인코더와 마찬가지로, <mark>**각 서브 레이어 주위에 잔차 연결을 적용하고 그 후에 층 정규화를 수행**</mark>한다.
>
> 또한 디코더 스택내에 <mark>**마스크드 멀티 헤드 셀프 어텐션(Masked Mult-Head Self-attention)가 있다. 현재 위치가 이후 위치를 참조(attend)하지 못하도록 한다.**</mark>
>
> <mark>**마스킹을 해서 현재 시점보다 이전에 예측했던 토큰들만 의존하도록 보장**</mark>해준다. 즉, 이미 예측한 토큰들만 의존하고 이후 토큰들은 의존하지 못하게 마스킹한다.

<br>

더 자세히 설명하자면 아래와 같다.

<br>

트랜스포머는 <mark>**auto-regressive 구조 방식으로 동작**</mark>한다고 했다.

y1 → y2 → y3 → … 순서대로 하나씩 생성하며, y_t를 예측할 때는 t 이전의 출력들만 참조해야한다.

하지만 디코더에 입력할때 토큰들을 한번에 시퀀스로 넣어버리면 문제가 된다.

예를들어 정답 시퀀스가 [i, like, apples] 이라고한다.

훈련할때는 y1=i, y2=like, y3=apples 모두 가지고 있다.

그런데 디코더에 [i, like, apples]를 <mark>**한꺼번에 넣어버리면, 디코더의 self-attention은 기본적으로 모든 토큰끼리 attention을 계산**</mark>하려고한다.

<br>

즉 y1 → y3 정보를 볼수있고 y2 → y3, y3 → y1 등 미래 참조가 자유로워진다.

<mark>**그래버리면 auto-regressive 구조 방식이 이뤄지지 않는다.**</mark>

<br>

Transformer 에서는 훈련시 다음과 같이 처리한다.

입력 시퀀스:

y1 = i,

y2 = like

y3 = apples

<br>

하지만 디코더 입력으로는 다음처럼 왼쪽으로 한칸 이동시켜, 가장 앞 토큰인 i을 없에고, SOS 토큰을 넣어준다.

디코더 입력: [SOS, I, like]

디코더가 예측하는 정답: [I, like, apples]

<br>

결국 모델은

SOS → I

I → like

like → apples

를 학습한다.

이 상태에서 위치인 i는 이전 입력만 참조하게 된다.

<br>

shifted 입력이 준비되더라도 self-attention은 여전히 모든 토큰을 참조하려고 한다.

그래서 마스크를 적용한다.

<br>

self-attention은 다음 수식으로 계산된다.

Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) × V

<br>

시퀀스 길이가 3이고 토큰이 3개라면,

QK^T 3x3 행렬이 된다.

QK^T =

[ s00 s01 s02

s10 s11 s12

s20 s21 s22 ]

<br>

이 행렬의 의미는

s10 = 위치 1의 토큰이 위치 0 토큰을 얼마나 참고하는지 나타내는 점수

s12 = 위치 1의 토큰이 위치 2 토큰을 얼마나 참고하는지 나타내는지 점수

<br>

이때 self-attention은 모든 위치를 다 보게 된다.

<mark>**디코더는 auto-regressive 이끼 때문에 미래 토큰을 보면 안된다.**</mark>

<br>

그래서 mask 행렬을 사용한다.

mask =

[ 0 -inf -inf

0 0 -inf

0 0 0 ]

<br>

여기서

- <mark>**0은 “참조 가능”**</mark>
- <mark>**-inf는 “softmax에서 0으로 만들기 위해 완전히 막는 값”이다.**</mark>

<mark>**QK^T 행렬과 mask 행렬을 더한다.**</mark>

<br>

QK^T:

[ 1 2 3
4 5 6
7 8 9 ]

<br>

mask:

[ 0 -inf -inf

0 0 -inf

0 0 0 ]

<br>

S_masked =

[ 1 2+(-inf) 3+(-inf)

4 5 6+(-inf)

7 8 9 ]

<br>

S_masked =
[ 1 -inf -inf
4 5 -inf
7 8 9 ]

<br>

이제 softmax를 하게 되면 -inf는 0이 된다.

<mark>**마스킹된 미래 위치는 음의 무한대 -inf을 더해서 0이 되게 만든다.**</mark>

<br>

결과적으로:

<mark>**i 번째 위치는 절대로 i + 1 이상의 토큰을 참조할 수 없다.**</mark>

<br>

3줄 요약

1. 출력 임베딩을 왼쪽으로 한 칸 shifted 한다.
2. 미래 토큰 참조를 막는 마스크를 self-attention에 적용한다.
3. 그래서 디코더는 항상 이전 시점 토큰들만 보고 다음 토큰을 예측하게 된다.

<br>

<mark>**정리**</mark>

1. <mark>**인코더**</mark>
   - 입력 시퀀스의 모든 토큰이 임베딩 + 위치 임베딩을 거친다.
   - Self-Attention을 통해 <mark>**자신을 포함한 시퀀스 전체 토큰과의 관계**</mark>를 반영한 은닉벡터를 계산한다.
   - 각 토큰별로 Feed-Forward를 거쳐 최종 은닉상태(z₁, z₂, …, zₙ)를 갖게 된다.
   - 결과적으로 인코더 출력은 <mark>**토큰별 context-aware 은닉벡터 시퀀스**</mark>다.
2. <mark>**디코더**</mark>
   - 생성 시작 시점에는 `<SOS>` 토큰을 입력으로 사용하며, 이후에는 <mark>**이전 시점에 예측한 토큰**</mark>이 입력으로 들어간다.
   - 입력 토큰은 임베딩 + 위치 임베딩을 거쳐 디코더 입력 벡터가 된다.
   - <mark>**Masked Self-Attention**</mark>으로 지금까지 생성된 모든 이전 토큰 벡터를 참고한다.
   - <mark>**Encoder-Decoder Attention**</mark>으로 인코더의 모든 토큰 은닉상태(z₁,…,zₙ)를 참조하여 시퀀스 전체 맥락을 반영한다.
   - 디코더 마지막 은닉벡터는 선형변환 + 소프트맥스를 거쳐 다음 토큰 확률을 계산하고, 확률이 가장 높은 토큰을 최종 예측으로 생성한다.

<br>

### 3.2 Attention

![image.png](../images/논문리뷰/Attention%20is%20all%20you%20need/2.png)

> 어텐션(attention) 함수는 <mark>**쿼리(query, Q)와 키-값(key-value, K, V) 쌍들의 집합을 출력으로 매핑하는 함수**</mark>로 설명될 수 있다. 여기서 <mark>**쿼리, 키, 값, 출력은 모두 벡터(vector)**</mark>이다.
>
> 출력은 값(value)들의 <mark>**가중합(weighted sum)으로 계산**</mark>되며, 각 값에 할당되는 가중치는 <mark>**쿼리와 해당 키 간의 호환성 함수(compatibility function)에 의해 계산**</mark>된다.

논문에서는 위와같이 설명하고 있다.

멀티 헤드 어텐션은 오른쪽 사진처럼 스케일 닷 프러덕트 어텐션과 나머지 들로 이루어져있다.

그중 핵심인 스케일드 닷 프로덕트 어텐션은 좌측 사진처럼 구조로 계산된다.

<br>

스케일드 닷 프로덕트 어텐션은 다음 과 같은 구조 방식으로 계산된다.

1. Q 와 K를 곱한다.(MatMul, Matrix Multiplication)
2. 곱한 결과값이 너무 커지지 않게 각 값 ÷ $\sqrt{차원 수}$ 을 해준다(Scale)
3. 미래 토큰 위치의 값을 -inf로 바꾼다.(Mask)
4. 곱한 결과를 softmax 해주어 가중치를 만든다.(SoftMax)
5. 가중치를 V와 곱한다.(MatMul)
6. 곱해서 나온 벡터들끼리 더해서 하나의 벡터를 만든다.(Concat)

<br>

Q(쿼리) = 무엇을 찾고 싶은지

K(키) = 어떤 정보를 가진 토큰 인지

V(값) = 실제 정보

<br>

모든 토큰들은 각각의 Q, K, V가 벡터형태로 존재한다.

1. Q는 자기에게 중요한 값을 얻기 위해 키 벡터들과 비교하여 중요도를 계산한다.
2. 그 결과 Q와K를 통해 생성된 중요도 값과 V를 합하여 하나의 출력 벡터를 만든다.

<br>

예를 들어 입력 시퀀스가 다음 네 단어라고 하자.

The cat eat fish

<br>

지금 보고 있는 토큰이 cat이라고 하자.

다른 단어들 중에서 “어떤 토큰이 cat이라는 토큰을 설명하는데 얼마나 중요한지”

그 중요도를 계산하는데 쓰이는 Q이다.

예를들어 “뒤에 eats가 있으므로 주어 역활을 할 가능성이 높다.”

“앞에 The가 있으므로 명사임이 확실하다” 같은 문맥 정보를 반영하기 위해 찾는 Q이다.

<br>

모든 토큰은 K, V를 가지고 있다.

K는 “내가 어떤 정보인지 이름”을 나타내고

V는 “내가 가지고 있는 정보에 대한 실제 내용”을 의미한다.

<br>

<mark>**첫번째. Q 와 K를 곱한다.**</mark>

이제 cat의 Q를 다른 모든 키들과 비교해서 각 단어를 얼마나 중요한지 중요도 점수를 만든다.

q_cat · k_The = qk_The

q_cat · k_cat = qk_cat

q_cat · k_eat = qk_eat

q_cat · k_fish = qk_fish

QK = [qk_The, qk_cat, qk_eat, qk_fish]

<br>

<mark>**두번째. 곱한 결과값이 너무 커지지 않게 각 값 ÷ $\sqrt{차원 수}$ 을 해준다(Scale)**</mark>

Q 와 K를 곱하여 Softmax에 넣으면, 점수가 너무 커져서 softmax가 치우치게 된다.

그렇게 되면 큰 값에 치우쳐져서 기울기가 0이 되어 버리게 된다.

그래서 각 값을 벡터 차원 수에서 루트를 씌워서 나누어준다.

<br>

- d_k = 차원 수 = 4라고 가정
- Q·K 결과 = [qk_The, qk_cat, qk_eat, qk_fish] = [1.2, 2.5, 1.8, 0.9] (예시 값)
- Scale 적용: 각 값 ÷ $\sqrt{d_k}$ = 각 값 ÷ 2
- 결과: [0.6, 1.25, 0.9, 0.45]

<br>

<mark>**세번째. 미래 토큰 위치의 값을 -inf로 바꾼다.(Mask)**</mark>

디코더에서는 아직 예측하지 않은 미래 단어를 참조하지 못하게 특정 점수를 -inf로 바꾼다.

<br>

The cat eat fish

예를 들어 토큰이 eat 이후 단어를 참조하면 안된다.

fish 위치 값을 -inf로 바꿔준다.

[0.6, 1.25, 0.9, -inf]

<br>

softmax를 적용시 -inf는 0으로 수렴해 미래 단어는 가중치에 영향이 없어진다.

<br>

<mark>**네번째. 곱한 결과를 softmax 해주어 가중치를 만든다.**</mark>

이 점수들을 softmax해서 가중치로 만든다.

softmax(qk_The) = w_The = 0.6

softmax(qk_cat) = w_cat = 1.25

softmax(qk_eat) = w_eat = 0.9

softmax(qk_fish) = w_fish = 0

<br>

<mark>**다섯번째. 가중치를 V와 곱한다.**</mark>

그다음 각 토큰의 V(값) 벡터에 방금 계산한 가중치를 곱한다.

임베딩 차원이 d_model = 4라고 하자.

- v_The = [0.1, 0.2, 0.3, 0.4]
- v_cat = [0.5, 0.6, 0.7, 0.8]
- v_eat = [0.9, 1.0, 1.1, 1.2]
- v_fish = [1.3, 1.4, 1.5, 1.6]

<br>

그리고 softmax를 통해 얻은 가중치가 있다.

- w_The = 0.6
- w_cat = 1.25
- w_eat = 0.9
- w_fish = 0

<br>

output_cat =

- w*The * v*The = 0.6 * [0.1,0.2,0.3,0.4] = [0.06, 0.12, 0.18, 0.24]
- w*cat * v*cat = 1.25 * [0.5,0.6,0.7,0.8] = [0.625, 0.75, 0.875, 1.0]
- w*eat * v*eat = 0.9 * [0.9,1.0,1.1,1.2] = [0.81, 0.9, 0.99, 1.08]
- w*fish * v*fish = 0 * [1.3,1.4,1.5,1.6] = [0, 0, 0, 0]

<br>

<mark>**여섯번째. 곱해서 나온 벡터들끼리 더해서 하나의 벡터를 만든다.**</mark>

output_cat =[

0.06+0.625+0.81+0,

0.12+0.75+0.9+0,

0.18+0.875+0.99+0,

0.24+1.0+1.08+0

]

output_cat = [1.495,1.77,2.045,2.32]

<br>

<mark>**이렇게 만들어진 벡터가 Decoder의 이후 단계를 거쳐 다음 단어인 fish를 예측하게 된다.**</mark>

1. Attention → 컨텍스트 벡터(`output_cat`) 생성
2. Feed-Forward → 비선형 변환
3. Residual + LayerNorm → 안정화
4. Linear → 단어 사전 크기로 변환
5. Softmax → 단어 확률 계산 → 다음 단어 예측

<br>
