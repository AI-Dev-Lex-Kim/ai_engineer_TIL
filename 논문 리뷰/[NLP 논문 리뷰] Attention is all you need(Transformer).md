# [NLP 논문 리뷰] Attention is all you need(Transformer)

> 유명하고 자주 사용하는 개념이다 보니 이전 부터 이 논문을 굉장히 읽고 싶었었다.
> 논문을 읽을때, 아키텍처 부분이 어려운 내용이 많아서, 깊게 이해하는데 시간이 꽤 걸리고 정리하는데도 최대한 자세히 적었다.
> 역시 복잡하고 깊게 이해하는 만큼 성취감과 즐거움이 매우 커서 즐겁게 논문을 읽었다.

## Abstract

기존의 시퀀스 모델(인코더와 디코더)들은 순환 신경망(RNN)이나 합성곱 신경망(CNN)에 기반이었다.

<br>

이 논문에서 <mark>**Transformer라는 모델은**</mark>

- <mark>**RNN과 CNN을 전혀 쓰지않고**</mark>
- <mark>**오직 Attention 메커니즘에만 사용하는 새로운 구조다.**</mark>

<br>

Transformer 모델은

- <mark>**성능(번역 품질)도 기존모델 보다 좋고**</mark>
- 병렬화가 훨씬 쉬워서 <mark>**학습 시간이 굉장히 짧다.**</mark>

<br>

## <mark>**1. Introduction**</mark>

RNN, LSTM, GRU는 그동안 언어 모델링, 기계 번역 같은 시퀀스 문제에서 SOTA를 달성해왔다.

<br>

이런 기존의 모델들은 다음과 같은 특징들이 있다.

- 시퀀스의 순서를 시간(t)에 맞게 차례대로 정렬하고
- <mark>**이전 은닉 상태 $h_{t-1}$**</mark>와 <mark>**현재 위치(t)의 입력을 사용**</mark>해서 <mark>**은닉상태 $h_t$ 를 계산하는 구조**</mark>이다.
- 결국, <mark>**계산이**</mark> 시퀀스 순서에 맞춰서 <mark>**순차적으로 진행**</mark>된다

<br>

이런 순차적인 특성 때문에

- 학습시 <mark>**병렬화가 불가능**</mark>하고
- 시퀀스 <mark>**길이가 길어질수록 문제**</mark>가 더 심해지며
- <mark>**메모리 한계**</mark> 때문에 큰 배치로 학습하기도 어렵다.

<br>

사실 <mark>**Attention 메커니즘은 기존모델에도 다양한 작업에서 사용**</mark>되어 왔다.

- 입,출력 시퀀스에서 거리에 상관없이 모델링 할 수 있었기 때문이다.
- 하지만, <mark>**대부분 RNN과 Attention이 같이 사용**</mark>되었었다.

<br>

이 논문에서의 Transformer는

- <mark>**RNN 같은 순환 구조를 아예 제거**</mark>하고
- 입력과 출력 사이의 global한 의존관계를
  <mark>**오직 Attention 만으로 학습하는 구조를 제안**</mark>한다.

<br>

그 결과

<mark>**훨씬 높은 수준의 병렬화가 가능**</mark>하고

8개의 P100 GPU로 12시간만 학습해도 번역 품질에서 SOTA를 달성했다.

<br>

## 2 Background

> 결국 <mark>**기존의 시퀀스 모델들은 순차적으로 계산되는 시도를 줄이기**</mark> 위한 여러 방법을 찾아왔고 비슷한 방식으로 처리했다.<br>
> 하지만 <mark>**여전한 한계가 존재했다.**</mark>

<br>

이 모델들은

- CNN을 기본 블록으로 사용하고
- 모든 입력/출력 위치의 은닉 표현을 병렬로 계산함으로써
  순차 계산 문제를 어느 정도 해결했다.
  <br>

하지만 CNN 기반 구조에서는

- 임의의 두 위치 사이의 정보가 전달되기까지 필요한 <mark>**연산 수가**</mark>
  <mark>**위치 간 거리와 함께 증가**</mark>한다.
- 그래서 <mark>**멀리 떨어진 위치들 간의 의존 관계를 학습하는 것이 어렵다.**</mark>

<br>

Transformer는

- 이러한 위치 간 <mark>**연산 수를 거리에 상관없이 거의 일정한 수준**</mark>으로 줄인다
  (즉, 멀리 떨어진 토큰 간 의존 관계도 효율적으로 학습 가능하게 만든다).

<br>

## 3. Model Architecture

![](https://velog.velcdn.com/images/lexkim/post/1a7d8666-99d5-4e6f-9919-96e8dbdae05f/image.png)
<mark>**인코더-디코더 전체구조**</mark>

Transformer도 <mark>**기존의 seq2seq 모델들 처럼 인코더-디코더 구조**</mark>를 가진다.

인코더

- 입력 시퀀스 토큰들: $x_1$, …, $x_n$
- <mark>**각 토큰들을 은닉 상태 벡터로 변환**</mark>해, 은닉 상태 토큰 벡터들의 집합인 시퀀스 $z = z_1, …, z_n$ 을 출력한다.

<br>

디코더

- <mark>**인코더가 만든 은닉상태 $z$를 입력**</mark>으로 받아
- 출력 시퀀스 토큰들을 $y_1, …, y_m$ 형태로, <mark>**디코더 한번의 과정당 하나씩 생성**</mark>한다.

<br>

<mark>**Auto-regressive 방식**</mark>

인코더-디코더 전체 과정을 보면, 디코더가 <mark>**출력 토큰을 생성하는 시점이 하나의 스텝**</mark>이라고 부른다.

Transformer 디코더는 <mark>**각 스텝에서 auto-regressive 방식**</mark>으로 동작한다.

<br>

auto-regressive 방식이란

- <mark>**다음 출력을 생성**</mark>할때,
- 이전 시점들에서 <mark>**이미 생성된 출력(토큰)들을 추가 입력**</mark>으로 사용하는 방식이다.
- 즉, $y_k$를 만들때, 이전에 생성한 토큰들인 $y_1, …, y_{k-1}$을 사용한다.

<br>

<mark>**인코더-디코더 블록의 공통 구조**</mark>

Transformer의 인코더와 디코더는 모두

- Self-Attention 층
- Point-wise Fully Connected Layer(FFN)
  를 층층이 쌓은 형태로 구성된다.

<br>

<mark>**Self-Attention Layer**</mark>

시퀀스 내의 <mark>**모든 토큰들이 서로의 정보를 참조**</mark>하며 자신의 토큰 벡터에 반영하게 하는 층이다.

<br>

<mark>**Point-wise Fully Connected Layer(FFN)**</mark>

point-wise란

- <mark>**~wise: ~별로 따로따로 나눠서 수행한다.**</mark>
- <mark>**point: 지점**</mark>

즉, 각 토큰 별로 따로따로 나눠 수행한다는 뜻이다.

<br>

그래서 <mark>**모든 각각의 토큰**</mark>마다,

<mark>**Fully Connected Network를 독립적으로 적용**</mark>한다는 뜻이다.

<br>

시퀀스 길이가 n일때

- 각 토큰 벡터에 대해 같은 FFN을 수행하지만
- 토큰들 사이를 섞지는 않는다

즉, 길이 n은 그대로 유지되고, 각 위치의 벡터만 변환된다.

<br>

여기서 Fully Connected Layer = MLP = (선형 변환 → 비선형 활성함수(ReLU 등)) 구조를 의미한다.

<br>

### 3.1 Encoder and Decoder Stacks

<mark>**인코더 구조**</mark>

Transformer의 <mark>**인코더는 6개의 동일한 레이어(layer)들이 쌓인 구조**</mark>이다.

- Encoder Layer 1
- Encoder Layer 2
- …
- Encoder Layer 6

이런 식으로 같은 형태의 블록을 6번 반복.

<br>

<mark>**인코더 서브레이어**</mark>

이제 하나의 Encode Layer가 어떻게 구성되어있는지 알아보자.

각 인코더 레이어는 <mark>**두 개의 서브 레이어로 구성**</mark>된다.

1. <mark>**멀티 헤드 셀프 어텐션(Multi-Head Self-Attention)**</mark>

   - <mark>**토큰들끼리 서로의 정보를 보면서(Attention) 섞는 부분**</mark>이다.
   - Multi-Header 이라서 <mark>**여러개의 다른 관점(헤드)으로 Attention을 동시에 수행한뒤 합친다.**</mark>

   <br>

2. <mark>**포지션-와이즈 피드포워드 네트워크(Position-wise Fully Connected Feed-Forward Network)**</mark>
   - <mark>**각 토큰에 대해 동일한 MLP(fully connected network)를 독립적으로 적용**</mark>한다.
   - 시퀀스 길이는 바꾸지 않고, 각 토큰의 표현만 <mark>**비선형 변환으로 변환**</mark>시킨다.
   - Fully Connected Layer = MLP = (선형 변환 → 비선형 활성함수(ReLU 등))

<br>

이렇게 Encode Layer는 각 두 개의 서브레이어로 이루어져있다.

서브 레이어가 출력한 x는 다음 서브 레이어에 전달된다.

이때 출력 x가 다음 서브 레이어에 x의 정보가 보다 자연스럽게 전달되게 추가적인 레이어를 거친다.

<br>

<mark>**Residual Connection + Layer Normalization**</mark>

두 개의 서브레이어 주변에 잔차 연결과 정규화를 두어 수행한다.

<br>

<mark>**Residual Connection(잔차 연결)이란?**</mark>

- 레이어에 입력된 x를,
- 레이어를 거쳐 출력된 값과 더해주는것이다.

<br>

조금더 풀어서 쓰면 더 이해하기 쉽다.

일반 레이어 → `y = Sublayer(x)`

잔차 연결 레이어 → `y = x + Sublayer(x)`

<br>

잔차 연결을 쓰는 이유는

- 깊은 네트워크 학습이 더 안정적이고,
- 기울기 소실 문제를 줄여준다.

<br>

<mark>**Layer Normalization**</mark>

서브 레이어의 출력 결과를 정규화 해주는것이다.

각 토큰 별로 정해진 차원으로 <mark>**정규화해주어서 학습을 더 안정화**</mark> 시켜준다.

<br>

결국 Residual Connection + Layer Normalization는 다음과 같다.

- <mark>**출력 = LayerNorm(x + Sublayer(x))**</mark>

<br>

<mark>**모든 레이어의 차원은 d_model=512 차원**</mark>이다.

이렇게 맞춰야, x와 Sublayer(x)의 차원이 같아서

x + Sublayer(x) 형태의 <mark>**잔차 연결을 자연스럽게 적용**</mark>할 수 있다.

<br>

<mark>**한줄 정리**</mark>

인코더: 6개, [Self-Attention → Residual + LayerNorm → FFN → Residual + LayerNorm]

<br>

<mark>**디코더 구조**</mark>

Transformer의 디코더는 <mark>**6개의 동일한 레이어(layer)들이 쌓인 구조**</mark>이다.

- Decoder Layer 1
- Decoder Layer 2
- …
- Decoder Layer 6

이런 식으로 같은 형태의 블록을 6번 반복.

<br>

<mark>**디코더의 서브레이어**</mark>

인코더는 서브 레이어가 2개였지만, <mark>**디코더는 서브 레이어가 3개이다.**</mark>

1. <mark>**마스크드 멀티 헤드 셀프 어텐션(Masked Multi-Head Self-Attention)**</mark>

   디코더가 지금까지 예측한 토큰들에 대해 self-attention을 수행한다.

   여기서 중요한 포인트는 <mark>**마스킹(masking**</mark>)을 한다는 점이다.

   - <mark>**현재 위치가 미래 위치(아직 생성되지 않은 토큰들)를 참조하지 못하게 막는다.**</mark>
   - 현재 토큰 위치 t가 t 이후의 토큰들(t+1, t+2, …)을 보지 못하게 마스킹한다.
   - 결과적으로 모델이 훈련시에도 추론 할때와 동일하게 미래를 모르는 상태로 학습하게 된다.

   <br>

   이 덕분에 디코더는 <mark>**auto-regressive 특성을 가지게된다.**</mark>

   - $y_t$를 만들때 $y_1,\ …,\ y_{t-1}$ 이전 정보만 사용할 수 있다.

   <br>

2. <mark>**인코더-디코더 어텐션 (Encoder-Decoder Multi-Head Attention)**</mark>

   - <mark>**디코더가 현재까지 예측한 토큰들과**</mark>
   - <mark>**인코더의 출력**</mark>
     <mark>**을 입력으로 받아**</mark>
     <mark>**입력 시퀀스의 어떤 위치에 주목 해야할지 학습하게 된다.**</mark>

   <br>

   디코더가 현재까지 예측한 토큰들은 Q(query)로 들어가고

   인코더의 출력은 K, V(key, value)로 들어가서 attention을 수행한다.

   <br>

3. 위치별 피드포워드 네트워크 (Position-wise Feed-Forward Network)

   인코더와 동일하게, 각 토큰 벡터에 독립적으로 적용되는 MLP.

<br>

<mark>**Residual Connection + Layer Normalization**</mark>

- 디코더 역시 인코더와 마찬가지로 각 서브 레이어 마다
  잔차 연결 + 정규화를 수행한다.

<br>

디코더: 6개, [Masked Self-Attention → Residual Connection + Layer Normalization → Encoder-Decoder Attention → Residual Connection + Layer Normalization→ FFN]

<br>

<mark>**이번 섹션에서 내가 인코더-디코더를 딥하게 이해한 것을 아래와 같이 정리해봤다.**</mark>

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

![](https://velog.velcdn.com/images/lexkim/post/164801ac-3803-4716-bdf0-44f9806224bd/image.png)

> 어텐션(attention) 함수는 <mark>**쿼리(query, Q)와 키-값(key-value, K, V) 쌍들의 집합을 출력으로 매핑하는 함수**</mark>로 설명될 수 있다. 여기서 <mark>**쿼리, 키, 값, 출력은 모두 벡터(vector)**</mark>이다.
>
> 출력은 값(value)들의 <mark>**가중합(weighted sum)으로 계산**</mark>되며, 각 값에 할당되는 가중치는 <mark>**쿼리와 해당 키 간의 호환성 함수(compatibility function)에 의해 계산**</mark>된다.

논문에서는 위와같이 설명하고 있다.

멀티 헤드 어텐션은 오른쪽 사진처럼 스케일 닷 프러덕트 어텐션과 나머지 들로 이루어져있다.

<br>

### 3.2.1 Scaled Dot-Product Attention

<mark>**스케일드 닷 프로덕트 어텐션은 다음 과 같은 구조 방식으로 계산된다.**</mark>

1. Q 와 K를 곱한다.(MatMul, Matrix Multiplication)
2. 곱한 결과값이 너무 커지지 않게 각 값 ÷ $\sqrt{차원 수}$ 을 해준다(Scale)
3. 미래 토큰 위치의 값을 -inf로 바꾼다.(Mask)
4. 곱한 결과를 softmax 해주어 가중치를 만든다.(SoftMax)
5. 가중치를 V와 곱한다.(MatMul)
6. 곱해서 나온 벡터들끼리 더해서 하나의 벡터를 만든다.(Concat)

<br>

<mark>**Q(쿼리) = 무엇을 찾고 싶은지**</mark>

<mark>**K(키) = 어떤 정보를 가진 토큰 인지**</mark>

<mark>**V(값) = 실제 정보**</mark>

<br>

모든 토큰들은 각각의 Q, K, V가 벡터형태로 존재한다.

1. Q는 자기에게 중요한 값을 얻기 위해 <mark>**키 벡터들과 비교하여 중요도를 계산한다.**</mark>
2. 그 결과 Q와K를 통해 생성된 <mark>**중요도 값과 V를 합하여 하나의 출력 벡터를 만든다.**</mark>

<br>

예를 들어 입력 시퀀스가 다음 네 단어라고 하자.

The cat eat fish

<br>

지금 보고 있는 토큰이 cat이라고 하자.

다른 단어들 중에서 <mark>**“어떤 토큰이 cat이라는 토큰을 설명하는데 얼마나 중요한지”**</mark>

그 중요도를 계산하는데 쓰이는 Q이다.

예를들어 “뒤에 eats가 있으므로 주어 역활을 할 가능성이 높다.”

“앞에 The가 있으므로 명사임이 확실하다” 같은 문맥 정보를 반영하기 위해 찾는 Q이다.

<br>

모든 토큰은 K, V를 가지고 있다.

K는 “내가 어떤 정보인지 이름”을 나타내고

V는 “내가 가지고 있는 정보에 대한 실제 내용”을 의미한다.

![](https://velog.velcdn.com/images/lexkim/post/c85285e3-0fb5-4d51-b5ad-ade9f12815db/image.png)

<br>

<mark>**첫번째. Q 와 K를 곱한다.(MatMul)**</mark>

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

<mark>**네번째. 곱한 결과를 softmax 해주어 가중치를 만든다.(Softmax)**</mark>

이 점수들을 softmax해서 가중치로 만든다.

softmax(qk_The) = w_The = 0.6

softmax(qk_cat) = w_cat = 1.25

softmax(qk_eat) = w_eat = 0.9

softmax(qk_fish) = w_fish = 0

<br>

<mark>**다섯번째. 가중치를 V와 곱한다.(MatMul)**</mark>

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
- w*cat * v*cat = 1.25 * [0.5,0.6,0.7,0.8] = = [0.625, 0.75, 0.875, 1.0]
- w*eat * v*eat = 0.9 * [0.9,1.0,1.1,1.2] = = [0.81, 0.9, 0.99, 1.08]
- w*fish * v*fish = 0 * [1.3,1.4,1.5,1.6] = = [0, 0, 0, 0]

<br>

<mark>**여섯번째. 곱해서 나온 벡터들끼리 더해서 하나의 벡터를 만든다.(Concat)**</mark>

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

### 3.2.2 Multi-Head Attention

Q, K, V로 하나의 어텐션으로만 연산을 수행하는 대신에, 여러 개의 학습된 선형 변환(linear)을 적용한뒤, <mark>**여러개의 어텐션으로 병렬적으로 연산하는것이 성능이 더 좋다는 것을 발견**</mark>했다.

그렇게 병렬적으로 <mark>**생성된 여러개의 결과들을 모두 연결(concat, 합치는것이 아니라 연결임)한뒤, 다시 선형변환(linear) 하여 최종 출력**</mark>을 생성한다.

<br>

<mark>**왜 여러 개의 attention이 성능이 좋았을까?**</mark>

단일 어텐션은 하나의 관점으로만 토큰을 해석한뒤, 결과를 만들게된다.

<mark>**즉, 한 가지 필터만 가지고 모든 관계를 판단해야한다.**</mark>

하나의 head로 문장을 보면, 한번의 attention은 해석할 수 있는 정보의 다양성이 떨어진다.

- 문장을 볼때 문법적 관계만 볼 수도 있고
- 의미적 연결만을 볼 수도 있고
- 문장 구조적 패턴만 볼 수도 있다.

이것들을 <mark>**모두 하나의 attention으로만 처리하려고하면 여러 관점을 보지 못하고 한 가지 관점**</mark>으로 보게 된다.

이것이 “다양성이 제한된다” 라는 표현이다.

<br>

멀티 헤드 어텐션은 여러개의 attention을 만든다.

<mark>**각 head는 서로 다른 방식으로 Q, K, V를 바라보게 된다.**</mark>

<br>

예를들어

head1은 구문적 관계에 집중

head2는 대명사와 참조 관계에 집중

head3는 동사-목적어 관계에 집중

… 이런식으로 각각이 다른 패턴을 학습한다.

그리고 마지막에 여러개의 head 결과를 연결(concat)하여 하나의 벡터로 합친다.

합쳐져 고유의 특징이 유지되게 된다.

(concat은 섞는 것이 아니라 연결하는 것이다.)

<mark>**즉, 멀티헤드는 여러 관점에서 본 결과를 이어붙여, 최종적으로 훨씬 풍부한 벡터를 만들어 낸다.**</mark>

<br>

### 3.2.3 Applications of Attention in our Model

![](https://velog.velcdn.com/images/lexkim/post/c2b6e78f-457a-4ae0-a8ea-08e8ebc6c26f/image.png)

<br>

<mark>**트랜스포머는 멀티헤드 어텐션(Multi-Head Attention)을 세 가지 방식으로 사용한다.**</mark>

1. 디코더에서 “인코더-디코더 어텐션” 층에서는 Q가 이전 디코더 층에서 값이 오며, K와V는 인코더에서 생성된 출력에서 오게된다.

   <br>

   <mark>**왜 Q는 이전 디코더 층에서 값이 오며, K와 V는 인코더에서 출력된 값이 오게 되는걸까?**</mark>

   먼저 어텐션에서 Q, K, V의 개념부터 다시 정리해보자.

   - Query는 지금 내가 어떤 정보를 찾고 싶은가를 나타내는 질문 벡터이다.
   - Key는 내가 가지고 있는 정보가 어떤 질문(Q)에 잘 맞는지를 나타내는 특징 벡터이다.
   - Value는 그 Key가 실제로 품고 있는 내용, 즉 참조하려는 정보 자체이다.

   <br>

   즉 Q는 질문, K는 질문과 비교하기 위한 주소, V는 실제 가져올 값이다.

   Q와 K를 연산해서 얼마나 관련있는지 계산한뒤, 그 가중치로 V과 연산해서 결과를 만든다.

   <br>

   Transformer 디코더는 출력 문장을 한단어씩 생성한다.

   예를 들어서 영어를 한국어로 번역한다면,

   입력: I love cats

   출력: 나는 고양이를 좋아한다.

   <br>

   디코더는 다음 단어를 생성하기 위해 두가지 정보가 필요하다.

   1. 현재까지 생성된 한국어 문장의 정보
   2. 입력 문장(영어)의 정보

   <br>

   즉, 디코더는 항상 <mark>**“내가 지금까지 만든 출력 문맥(한국어)”과 “인코더가 제공한 입력 문장(영어)의 의미 정보” 이 두가지를 결합해서 다음 단어를 만든다.**</mark>

   <br>

   <mark>**그래서 디코더는 위에서 설명한 두 가지 정보를 취합하기 위한 어텐션이 필요하다.**</mark>

   1. 현재까지 예측한 <mark>**한국어 문장 정보**</mark>의 토큰이 서로 다른 토큰들을 참조하게 하는 Self-attention
   2. 한국어 번역을 위해 <mark>**영어 문장 정보**</mark>를 가진 벡터를 생성한 Self-attention

   <mark>**Q는 위에서 설명한 1번 Self-attention인 디코더에**</mark>서 오고

   <mark>**K, V는 위에서 설명한 2번 Self-attention인 인코더에서 온다**</mark>는 것이다.

   <br>

   이 두 개의 Self-attention이 생성한 정보 벡터를 합쳐야한다.

   이 <mark>**두 개 벡터를 Self-attention으로 합치며, 각각 인코더와 디코더 Self-attention에서 왔으므로 인코더-디코더 어텐션이라고 부른다.**</mark>

   <br>

   <mark>**왜 한국어 Self-attention이 Q에 올까?**</mark>

   번역예시를 계속 보자.

   디코더가 지금까지 생성한 단어가 “나는 고양이를” 까지 만들었다.

   이제 다음 단어인 “좋아한다”를 만들어야한다.

   <br>

   이때 디코더는 이렇게 말하고 있다.

   “다음으로 예측하려는 단어가 이전까지 만든 한국어 문장 토큰들 중에서 어떤 토큰들이 중요한 토큰인지 알고싶다.”

   <br>

   <mark>**즉, 질문(Query)는 현재 한국어 단어를 만들고 있는 디코더 이다.**</mark>

   <mark>**그래서 디코더는 “내가 지금 까지 예측해서 만든 어떠한 한국어 토큰 정보를 참고해야하나?” 라는 질문을 던지는 것이다.**</mark>

   만약 이전까지 생성한 한국어 문장을 참조하지 않는다면, 다음 단어는 이전까지 생성한 한국어 문장과 상관없이 원문인 영어 문장만을 보고 다음 토큰을 예측해야하는 것이다.

   <br>

   그래서 첫번째로 오는 서브레이어층인 Masked Multi Self-attention에서 다음으로 예측하려고 하는 토큰이 현재까지 만든 한국어 문장의 토큰들을 모두 참조한다.

   <br>

   <mark>**그럼 K와 V는 왜 인코더에서 오는것일까?**</mark>

   Masked Multi Self-attention 서브 레이어층에서 지금까지 예측한 한국어 문장의 토큰들을 모두 참조했다.

   <br>

   이제는 영어 문장 정보를 얻어서 다음 토큰을 예측해야한다.

   <mark>**만약, 원문인 영어 문장을 참조하지 않는다면, 이전까지 생성한 한국어 문장만을 보고 다음 단어를 생성해야한다. 그렇다면 영어 문장 번역이란 태스크가 아니게 된다.**</mark>

   <br>

   입력 문장인 영어 문장의 의미는 인코더가 이미 다 분석해서 하나의 의미 벡터들로 변환해둔 상태이다.

   <br>

   지금까지 생성한 한국어 문장에서 다음 한국어 토큰을 예측해야한다.

   다음 한국어 토큰이 영어 문장의 정보를 얻어야하므로 질문(Query)는 한국어가 하고 Key, Value를 영어 문장 정보 벡터로 정한것이다.

   <br>

   최종적으로

   - <mark>**Q는 Masked Multi Self-attention로 생성한 이전까지 예측한 한국어 문장 정보 벡터로 설정**</mark>
   - <mark>**K와V는 인코더의 Multi Self-attention로 생성한 영어 문장 정보 벡터로 설정**</mark>

   인코더-디코더 어텐션을 통해 2가지 정보를 반영한 벡터를 생성하는것이다.

   <br>

2. 인코더는 셀프 어텐션 층을 가지고 있다. 셀프 어텐션층에서 Q, K, V가 모두 “포지셔널 임베딩 + 인풋 임베딩” 층의 출력에서 오게 된다.

   > 인코더의 각 위치는 인코더의 이전 층에 있는 모든 토큰 위치를 참조할수 있다.

   논문에서는 위와 같은 문장이 있다.

   인코더는 여러 층(layer)로 쌓여 있는데, 예를들어 <mark>**인코더 2층에서 셀프 어텐션을 계산한다면, 셀프 어텐션으로 들어오는 입력 값은 인코더 1층에서 출력되어 생성된 값이다.**</mark>

   <br>

   즉, 인코더 2층의 각 토큰은 인코더의 1층의 모든 토큰들을 참고할 수 있다는 뜻이다.

   이 구조를 쓰게 되면, 2층의 입력으로 들어온 1층의 출력값 안에는, 문장 내에서 <mark>**멀리 떨어진 단어도 벡터형태로 존재하기 때문에 참조하여 정보를 반영**</mark>할 수 있다.

   <br>

3. 마찬가지로, <mark>**디코더의 셀프 어텐션 층**</mark>들은 디코더 내에서 현재 위치의 토큰을 포함하여 <mark>**그 이전의 모든 위치의 토큰들을 참조**</mark>할 수 있도록 한다.

   그러나 <mark>**셀프회귀(auto-regressive) 특성을 유지하기 위해 디코더 내에서의 미래 방향 정보 흐름을 차단**</mark>해야 한다.

   이를 위해 스케일드 닷-프로덕트 어텐션(scaled dot-product attention) 내부에서 <mark>**미래 토큰 위치의 모든 값을 마스킹(masking)하여 −∞로 설정**</mark>한다.

   이렇게 되면 Softmax에서 -inf는 0값으로 바뀌게되어 어떠한 영향도 주지 않는다.

### 3.3 Position-wise Feed-Forward Networks

Transformer는 크게 두 서브레이어로 구성된다.

1. Self-attention 서브 레이어
2. FFN(Position-wise Feed-Forward Networks) 서브 레이어

<br>

Self-attention은은 각 토큰이 문장의 모든 토큰 사이의 관계를 계산하는 역활이다.

그런데 그 뒤에는 하나하나 각 토큰별로 변환하는 FFN이 붙는다.

<mark>**이 FFN은 토큰 간의 관계를 보지 않는다.**</mark>

<mark>**오직 각 토큰 하나만 입력으로 받아, 그 토큰의 벡터를 더 풍부한 형태로 가공하는 역활을한다.**</mark>

<br>

즉, Self-attention이 “토큰 간 상호작용”이라면,

FFN은 “토큰 내 정보 확장”이다.

<br>

여기서 Position-wise라는 말은

FFN(x1)

FFN(x2)

FFN(x3)

…

이렇게 각각 토큰에게 별도의 계산을 한다는 뜻이다.

- position: 위치
- wise: ~마다

그리고 각 토큰에 FFN을 적용할때 사용하는 가중치가 모두 같다.

<br>

즉, FFN은 이렇게 동작한다.

x1 → 동일한 FFN → y1
x2 → 동일한 FFN → y2
x3 → 동일한 FFN → y3

<br>

서로에게 영향을 주지 않는다.

오로지 자기 자신의 벡터만 변화시킨다.

<br>

FFN은 다음과 같은 구조를 가진다.

$$
FFN(x)\ =\ W2(ReLU(W_1 * b_1)) + b_2
$$

여기서 w1는 512차원 → 2048 차원으로 확장하는 가중치이다.

W2는 2048차원 → 512차원으로 다시 줄여주는 가중치이다.

<br>

<mark>**왜 2048차원으로 확장하고 다시 512차원으로 줄일까?**</mark>

512 → 2048 → 512

<mark>**이렇게 확장하면, 비선형 변환인 ReLU을 통해 더 복잡한 패턴을 모델링 할 수 있다.**</mark>

<br>

<mark>**어텐션만 사용**</mark>하면, 토큰 간 관계는 잘 파악할 수 있지만

<mark>**각 토큰 내부의 표현을 더욱 복잡하게 비틀거나 확장하는 능력이 부족하다.**</mark>

<br>

<mark>**FFN은 각 토큰의 벡터 내부에서 비선형 조합을 만들어 더 강력한 표현을 구성한다.**</mark>

“attention의 토큰간 관계”와 “FFN의 토큰 내부 표현 강화” 이 둘을 합쳐 강력한 표현 학습이 가능해진다.

<br>

<mark>**왜 FFN은 각 토큰별로 적용하는 것일까?**</mark>

문장이 길든 짧든 FFN은 모든 토큰을 동시에 처리할 수 있어 매우 빠르다.

토큰 간의 의존성이 없기 때문이다.

<br>

토큰 간 상관을 고려하지 않기 때문에

문장의 구조 같은 것은 FFN이 전혀 신경쓰지 않는다.

문장의 구조를 고려하는것은 앞 단계인 attention에서 이미 처리된 일이다.

<br>

정리

1. 각 토큰을 독립적으로 처리한다
2. 두 개의 선형 변환(512→2048→512)과 ReLU로 구성된다
3. 모든 토큰에 동일한 W1, W2를 사용한다
4. 토큰 간 관계를 보지 않으며

   오직 토큰 내부 정보만 확장/변형해준다.

5. 결과적으로 각 토큰이 더 풍부한 특징 표현을 갖게 된다.

<br>

### 3.4 Embeddings and Softmax

1. <mark>**학습된 두 임베딩**</mark>

인코더와 디코더로 들어가기 전에 임베딩 레이어로 입력 토큰과 출력 토큰(디코더 입력)이 벡터로 변환한다.

<br>

<mark>**하나는 인코더 입력을 위한 학습된 임베딩**</mark>

<mark>**하나는 디코더 입력을 위한 학습된 임베딩이다.**</mark>

<br>

1. <mark>**디코더 출력 벡터 → 다음 토큰 확률로 변환하는 과정(선형 변환 & softmax)**</mark>

디코더는 문장을 생성할 때, 다음 토큰을 위한 출력 벡터를 만들어낸다.

이 출력 벡터의 차원 크기는 512차원의 벡터이다.

<br>

하지만 실제로 필요한 것은 토큰 확률이다.

즉, 단어사전(vocabulary) 크기 만큼의 확률 벡터이다.

<br>

예를 들어 단어 종류가 30000개면, 디코더 출력을 30000개 확률로 변환해야한다.

<br>

그때 한는 작업이 바로

1. 선형 변환(512 → 30000)
2. softmax 적용

이다.

```python
logit = W_softmax * y
prob = softmax(logit)
```

이러한 구조를 사용해서 각 단어가 다음에 올 확률을 만든다.

<br>

1. <mark>**임베딩 레이어 가중치와 softmax 직전 선형 변환 레이어 가중치는 서로 동일한 가중치이다.**</mark>

입력 임베딩을 만들때 쓰는 가중치 W_embed

출력 logit을 만들때 쓰는 가중치 W_softmax

이 둘은 서로 같은 가중치 행렬을 공유한다.

<br>

<mark>**이것을 weight tying(가중치 공유)라는 용어로 부른다.**</mark>

두 가중치 행렬을 사용하는 대신 하나의 가중치 행렬만 사용하는 방법이다.

<br>

왜 가중치를 공유할까?

1. <mark>**단어 임베딩을 할때의 의미 공간과 단어를 뽑아낼때의 의미 공간을 같은 맥락으로 유지해서 성능이 향상된다.**</mark>
2. <mark>**파라미터 수가 감소하게 되서 overtfitting에 빠질 위험이 감소한다.**</mark>
3. <mark>**학습해야할 가중치가 감소해서 학습 속도가 개선된다.**</mark>

<br>

결국, 성능도 좋아지고, 메모리도 절약되고, 훈련도 안정화된다.

<br>

1. <mark>**임베딩에서는 해당 가중치에 sqrt(d_model)을 곱한다.**</mark>

임베딩에서 가중치를 sqrt(512)로 스케일링 하는 이유는

임베딩 값의 분포가 너무 작아지는것을 방지하기 위함이다.

<br>

<mark>**임베딩 테이블의 초기값은 일반적으로 작은 값으로 초기화된다.**</mark>

그런데 위치 인코딩(position encoding)도 함께 더해지기 때문에

<mark>**입력 임베딩이 너무 작으면, position encoding에도 영향을 과도하게 주게된다.**</mark>

<br>

그래서 임베딩 벡터를

sqrt(d_model) = sqrt(512) = 약 22.6

<mark>**으로 스케일업해서 임베딩이 갖는 절대 크기를 어느정도 균형있게 맞춘다.**</mark>

<br>

<mark>**정리**</mark>

1. 임베딩은 토큰을 512차원 벡터로 변환하는 테이블 같은 역할이다.
2. 디코더 출력은 (512 → 단어 수) 선형 변환 후 softmax로 다음 단어 확률이 된다.
3. 임베딩 가중치와 softmax 가중치를 공유하는 이유는 같은 의미 공간을 일관되게 사용하고 파라미터 수를 줄이기 위해서이다.
4. 임베딩에 sqrt(d_model)을 곱하는 이유는 초기 임베딩 크기와 위치 인코딩의 크기를 균형 있게 맞추기 위해서이다.

<br>

### 3.5 Positional Encoding

![](https://velog.velcdn.com/images/lexkim/post/37498cac-c317-44bc-8db8-0bd35c3bf092/image.png)

Transformer 모델은 순환 구조(Recurrent)나 합성곱(convolution)을 포함하지 않기 때문에, <mark>**시퀀스의 순서 정보를 알기 위해 각 토큰의 상대적 또는 절대적 위치에 대한 정보를 주입**</mark>해야 한다.

<mark>**이것을 위해서 인코딩과 디코딩 스택 맨아래층에 위치 인코딩(positionnal encoding)을 입력 임베딩에 추가한다.**</mark>

<br>

<mark>**위치 인코딩의 차원은 임베딩과 동일하게 d_model이며, 이를 통해 두 값을 더할 수 있다.**</mark>

<mark>**위치 인코딩에는 학습 가능한 것과 고정된(fixed) 것 등 다양한 선택이 있다.**</mark>

<br>

> 이 논문에서는 사인과 코사인 함수를 사용했다.

<br>

<mark>**왜 사인과 코사인 함수를 사용했을까?**</mark>

위치 정보(pos)를 일종의 규칙있는 숫자 패턴으로 바꿔서 임베딩에 더해준다.

<br>

<mark>**이때 단순히 위치를 1,2,3,4로 주는게 아니라 다양한 주파수의 사인파, 코사인파를 이용해 더 풍부한 패턴을 만든다.**</mark>

<br>

<mark>**사인파를 토큰의 어떤 값에 적용할까?**</mark>

위치 인코딩(positional encoding)의 각 차원은 하나의 사인파(sinusoid)에 대응된다.

예를 들어 d_model = 512 라고 하자.

그러면 위치 인코딩 벡터는 각 토큰 별로 512개의 요소가 존재하는 512차원 벡터이다.

<br>

토큰 위치 5

위치 임베딩(5) = [값0, 값1, 값2, …, 값511]

여기서 총 512개의 값이 있다.

<br>

<mark>**이 512개의 차원 하나하나 마다, 서로 다른 파장, 서로 다른 주파수를 가진 사인파 또는 코사인파를 계산해 적용한다는 뜻이다.**</mark>

<br>

i = 차원

i = 0 → 첫번째 sin 계산

i = 1 → 두번째 sin 게산

…

i = 511 → 마지막 sin 계산

<br>

<mark>**어떤 수식으로 위치 인코딩을 할까?**</mark>

$$
PE(pos, 2i) = sin(pos / 10000^{2i / d_{model}})
$$

여기서

pos는 시퀀스 내에 토큰의 위치이다.

i는 각 토큰 내의 512차원중 차원의 인덱스이다.

d_model = 512 (모델 차원)

<br>

여기서 토큰의 차원이 증가할수록 차원의 위치 값이 커지게된다.

pos = 3 (시퀀스에서 3번째 토큰)

i = 0 → $sin(\frac{3}{10000^{\frac{0}{512}}})$

i = 1 → $sin(\frac{3}{10000^{\frac{2}{512}}})$

i = 2 → $sin(\frac{3}{10000^{\frac{4}{512}}})$

i가 커질수록 조금씩 분모가 커진다.

그러면 sin(pos / 매우 큰 숫자)는 변화 폭이 아주 작아져서 천천히 움직이는 패턴이된다.

<br>

즉,

차원 0
sin(pos / 1)
→ 빠르게 변하는 파형

차원 250
sin(pos / 10000^(중간값))
→ 중간 속도 파형

차원 511
sin(pos / 10000^(거의 1))
→ 매우 느리게 변하는 파형

<br>

이렇게 512개의 파형을 섞으면 각 토큰의 pos라는 위치 정보를 다양한 주파수 성분을 가진 벡터로 표현할 수 있다.

<br>

<mark>**왜 다양한 파장이 필요할까?**</mark>

문장에서는 단어 간의 관계가 근거리일수도 장거리일수도 있다.

<mark>**그래서 위치 인코딩은 다양한 거리에 대해 감도있게 반응하도록 짧은 파장부터 긴 파장까지 다 넣는것이다.**</mark>

<br>

짧은 파장: pos가 조금만 변해도 값이 크게 변화

→ 가까운 토큰 위치 차이에 민감

긴파장: pos가 많이 변해야 값이 변함

→ 먼 토큰 위치 차이에 민감

<br>

보통 가까운 위치에 있는 단어들 간의 관계는 매우 중요하다.

“나는 오늘 학교에 갔다”

여기서

오늘 ↔ 갔다

학교에 ↔ 갔다

이런 관계는 대체로 가까운 위치에서 발생한다.

<br>

즉, 근처에 있는 단어들이 서로 영향을 주는 경우가 많다.

그래서 모델은 단어 간 간격이 작은 경우를 민감하게 구분할 필요가 있다.

<br>

<mark>**왜 짧은 파장이 근거리 관계에 유리할까?**</mark>

근접 단어 관계는 보통 단어 간 거리가 1~3 정도 수준에서 많이 나타난다.

이런경우

pos=10과 pos=11 → pos 차이 1

pos=10과 pos=12 → pos 차이 2

pos=10과 pos=13 → pos 차이 3

<mark>**이런 작은 거리 차이를 Transformer가 서로 다른 관계로 명확히 인식해야한다.**</mark>

<br>

1칸 차이(근접)
2칸 차이(조금 멀리)
3칸 차이(더 멀리)

이런 거리 감각을 정확히 구분해야 한다는 말이다.

<br>

<mark>**이런 관계를 구분하기 위해서는**</mark>

<mark>**각 위치 pos가 1 또는 2만 달라져도 명확히 다른 패턴을 가져야한다.**</mark>

<br>

pos=10과 pos=11의 차이(거리1)

pos=10과 pos=12의 차이(거리2)

<br>

즉, 가까운 두 위치를 세밀하게 구분할 수 있다.

<br>

그래서 짧은 파장의 $sin(pos / 10000^{2i / d_{model}})$는 pos가 딱 1만 달라도 값이 크게 변한다.

<br>

<mark>**그래서 512개의 파장들중에서 짧은 파장들 중에서 짧은 파장은 가까운 위치 차이에 민감하다.**</mark>

<br>

<mark>**만약 파장이 매우 길기만 하다면?**</mark>

반대로 512개의 파장들이 짧은 파장이 없고 <mark>**긴 파장들만 있다고 해보자.**</mark>

pos가 1~5정도 변해도 값이 거의 안바뀌게 된다.

<br>

예를 들어
파장이 긴 사인파 sin(pos / 10000)

pos=10
sin(10/10000) ≈ sin(0.001)

pos=11
sin(11/10000) ≈ sin(0.0011)

<br>

두 값은 거의 차이가 없다.

<mark>**pos가 1, 2만 바뀌어도 거의 동일하게 보인다.**</mark>

그럼 모델은 pos=10과 pos=11이 얼마나 가까운 위치인지 구분하기가 매우 어렵다.

<br>

<mark>**그래서 Transformer는 두 가지 파장을 모두 쓴다.**</mark>

짧은 파장

pos가 조금만 달라도 값이 확 달라짐

→ 짧은 거리 표현에 유리

<br>

긴 파장

pos가 크게 변해야 값이 달라짐

→ 먼 거리 관게 표현에 유리

<br>

<mark>**Transformer는 문장의 짧은 거리, 긴거리 모두 다룰 수 있어야 하므로**</mark>

<mark>**512차원 전체를 다양한 파장인 짧은 파장~긴 파장 으로 채워 넣는다.**</mark>

<br>

<mark>**정리**</mark>

512개의 서로 다른 파장을
“512개의 감각 센서”라고 생각해보자.

짧은 센서 → 아주 작은 위치 변화도 감지
중간 센서 → 중간 변화 감지
긴 센서 → 큰 변화 감지

<br>

센서들이 내는 신호가 벡터 전체에 녹아있고

<mark>**그 벡터끼리 내적을 수행하면서 모든 센서의 신호가 결합된 결과를 얻는것이다.**</mark>

<br>

## 4 Why Self-Attention

![](https://velog.velcdn.com/images/lexkim/post/44f47a52-488d-4790-a20d-afac96d60ac7/image.png)

Table 1: 다양한 레이어 유형에 대해, 최대 경로 길이(maximum path lengths), 레이어당 복잡도(per-layer complexity), 그리고 최소 순차 연산 개수(minimum number of sequential operations)를 나타낸다. 여기서 n은 시퀀스 길이(sequence length), d는 표현 차원(representation dimension), k는 합성곱(convolution)의 커널 크기(kernel size), r은 제한적 셀프 어텐션(restricted self-attention)에서의 이웃(neighborhood)의 크기를 의미한다.

<br>

<mark>**이 챕터에서는 시퀀스 변환(sequence transduction) 모델에서 흔히 쓰이는 세가지 층을 비교한다.**</mark>

---

1. 순환층(RNN)
2. 합성곱층(CNN)
3. 셀프 어텐션(Self-Attention)

<br>

<mark>**셀프 어텐션을 사용하는 동기를 설명하기 위해 3가지 고려사항이 있다.**</mark>

---

1. 계산 복잡도(computational complexity)

   <mark>**층 하나를 실행하는데 필요한 총 연산량이다.**</mark>

2. 최소 순차 연산량(sequential operations)

   병렬 처리가 얼마나 가능한지 보여준다.

   <mark>**순차적으로 실행해야 할 연산이 적을수록 병렬화가 잘되고 빠른 모델이다.**</mark>

3. 장거리 의존 관계의 경로 길이(maximum path length)

   멀리 떨어진 단어가 서로 영향을 주기 위해

   <mark>**네트워크 내부에서 정보가 몇 개 층을 통과해야 하는지를 의미한다.**</mark>

   경로가 짧을수록 장거리 의존 관계를 더 학습하기 쉽다.

<br>

<mark>**표1: Self-Attention vs RNN**</mark>

---

1.  Self-Attention은 모든 위치를 연결하는데 상수 개(constant number)의 연산만 필요하다.
    <mark>**즉, 한번의 어텐션 계산으로 모든 토큰이 서로를 직접 바라볼수있다.**</mark>
    그래서 경로 길이는 1이다. $O(1)$
2.  RNN은 <mark>**토큰을 순서대로 하나씩 처리해야 해서 O(n)의 순차 연산이 필요**</mark>하다.

    경로 길이 = n

3.  <mark>**그래서 RNN은 병렬화가 어렵고, 장거리 관계를 잘 못잡는다.**</mark>

    반면, <mark>**Self-Attention은 병렬화가 매우 잘되고, 장거리 의존 관계를 훨씬 쉽게 학습**</mark>할 수 있다.

<br>

<mark>**계산 복잡도 비교: n이 작을 때 Self-Attention이 더 빠르다.**</mark>

---

Self-Attention 레이어에서의 계산량은 $O(n^2*d)$

RNN의 계산량은 $O(n*d^2)$

<br>

따라서 <mark>**문장 길이 n < 표현 차원 d**</mark>

<mark>**인 상황에서 Self-Attention이 더 빠르다.**</mark>

<br>

문장 표현이

- 워드피스(word-piece)
- BPE(Byte-Pair Encoding)

<mark>**을 사용하면, 토큰 수(n)가 상대적으로 작기 때문에, 대부분 Self-Attention이 RNN 계산량 보다 유리하다.**</mark>

<br>

<mark>**긴 시퀀스를 다룰때: 제한된 셀프 어텐션(restriceted Self-Attention)**</mark>

만약 시퀀스가 아주 길면 Self-Attention의 계산 복잡도 $O(n^2)$의 계산량이 부담이 된다.

이를 해결하기 위한 방법 중 하나가 “제한된 어텐션(restricted attention)”이다.

<br>

각 위치에서 전체 시퀀스를 보지 않고 그 토큰 위치 주변 r개의 이웃만 보는 방식이다.

복잡도는 줄어들지만, 경로 길이는 $O(1)$에서 $O(\frac{n}{r})$로 증가한다.

Transformer 논문에서도 향후 이 방법을 더 연구할 것이라고 언급했다.

<br>

<mark>**CNN과의 비교**</mark>

CNN층은 kerner size = k일때, 한번에 k 범위의 토큰만 연결할 수 있다.

따라서 전체 시퀀스를 연결하려면 계속 층을 쌓아야한다.

필요한 층수는 O(log_k(n))으로 층을 여러개 쌓는 만큼 경로 길이도 증가한다.

게다가 CNN은 계산비용이 k배 더 높다.

<br>

이렇게 RNN과 CNN을 비교할때 성능과 계산 효율성이 좋기때문에 Transformer는 Self-Attention + FNN 조합을 채택했다.

<br>

<mark>**추가적 장점으로 해석 가능한 모델이라는 점도 있다.**</mark>

Self-Attention은 각 토믄이 어떤 토큰을 바라보는지 어텐션 분포를 시각화 할 수 있어서 모델 내부 동작을 해석 가능하게 만든다.

Transformer 논문에서는 부록에서 예시를 보여주면서 설명한다.

다양한 헤드들은

- 문법적 구조(주어-동사 연결 등)
- 의미적 구조(대명사 참조 등)

을 포착하도록 학습된다.

이것은 RNN이나 CNN에서는 가능하지 않은 장점이다.

![](https://velog.velcdn.com/images/lexkim/post/419a1017-1326-48b3-8444-20b1cd628c8c/image.png)

위의 두 그림은 각각 헤드 1개씩을 의미한다. 2개의 헤드가 서로 다른 task를 담당하는것을 시각적으로 보여준다.

## Conclusion

이 논문에서는 Transformer를 제안했다. <mark>**순전히 어텐션에 기반한 최초의 시퀀스 변환(sequence transduction) 모델**</mark>이다.

인코더-디코더 구조에서 가장 일반적으로 사용되는 <mark>**순환층을 다중 헤드 셀프 어텐션으로 대체**</mark>한 모델이다.

번역 작업에서는 트랜스포머가 RNN이나 CNN 층에 기반한 아키텍처보다 <mark>**훨씬 빠르게 학습**</mark>할 수 있었다.

영어-독일어와 영어-프랑스어 번역과제에서도 <mark>**state-of-the-art을 달성**</mark>했다.

기존에 보고된 <mark>**모든 앙상블 모델보다도 우수한 성능**</mark>을 보여줬다.

이 논문에서는 어텐션 기반 모델을 텍스트 외의 입력 및 출력 모달리티를 포함하는 문제로 확장하고, <mark>**이미지, 오디오, 비디오 같은 대규모 입력과 출력을 효율적으로 처리**</mark>할 수 있도록 <mark>**국소적 제한(local restricted) 어텐션 메커니즘을 연구**</mark>할 것이라고 했다.

<mark>**생성 과정에서 덜 순차적(sequential)으로 만드는것도 또 다른 연구**</mark>의 목표라고 했다.

<br>
