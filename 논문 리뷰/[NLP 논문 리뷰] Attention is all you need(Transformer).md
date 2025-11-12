# [NLP 논문 리뷰] Attention is all you need(Transformer)

- [\[NLP 논문 리뷰\] Attention is all you need(Transformer)](#nlp-논문-리뷰-attention-is-all-you-needtransformer)
  - [Abstract](#abstract)
  - [1. Introduction](#1-introduction)
  - [2 Background](#2-background)
  - [3. Model Architecture](#3-model-architecture)
    - [3.1 Encoder and Decoder Stacks](#31-encoder-and-decoder-stacks)

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

![논문리뷰1](../images/논문리뷰/1.png)

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

인코더(Encoder)는 N=6개의 동일한 층들로 구성된 스택(stack)이다. 각 층은 두 개의 하위 층(sub-layer)으로 이루어져 있다. 첫 번째는 다중 헤드 자기 어텐션(multi-head self-attention) 메커니즘이고, 두 번째는 간단한 위치별 완전연결 피드포워드(position-wise fully connected feed-forward) 네트워크이다.

각 두 개의 하위 층 주변에는 잔차 연결(residual connection) [11]을 적용하며, 그 뒤에 층 정규화(layer normalization) [1]를 수행한다. 즉, 각 하위 층의 출력은 LayerNorm(x + Sublayer(x)) 형태이며, 여기서 Sublayer(x)는 해당 하위 층이 수행하는 함수이다.

이러한 잔차 연결을 용이하게 하기 위해, 모델의 모든 하위 층들과 임베딩 층(embedding layer)은 출력 차원 d_model = 512를 갖는다.

디코더(Decoder) 또한 N=6개의 동일한 층들로 구성되어 있다. 각 인코더 층의 두 개의 하위 층에 더해, 디코더는 세 번째 하위 층을 추가하는데, 이는 인코더 스택의 출력을 대상으로 다중 헤드 어텐션(multi-head attention)을 수행한다.

인코더와 마찬가지로, 각 하위 층 주위에 잔차 연결을 적용하고 그 후에 층 정규화를 수행한다.

또한 디코더 스택 내의 자기 어텐션 하위 층을 수정하여, 현재 위치가 이후 위치를 참조(attend)하지 못하도록 한다.

이러한 마스킹(masking)은 출력 임베딩(output embedding)이 한 위치(offset)만큼 이동되어 있다는 사실과 결합되어, 위치 i의 예측이 i보다 작은 위치들의 이미 알려진 출력에만 의존하도록 보장한다.
