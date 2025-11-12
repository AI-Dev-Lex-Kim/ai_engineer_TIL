# [NLP 논문 리뷰] Attention is all you need(Transformer)

- [\[NLP 논문 리뷰\] Attention is all you need(Transformer)](#nlp-논문-리뷰-attention-is-all-you-needtransformer)
  - [Abstract](#abstract)
  - [1. Introduction](#1-introduction)

---

논문 링크: https://arxiv.org/pdf/1706.03762

## Abstract

기존의 인코더와 디코더를 포함한 모델들은 순환 신경망(RNN)이나 합셩곱 신경망(CNN)에 기반해 있었다.

이 논문에서 <mark>**Tansformer라는 모델은 순환과 합성곱을 완전히 제거하고, 오직 어텐션 메커니즘에만 기반해 새롭고 간단한 네트워크 구조를 제안**</mark>한다.

Transformer 모델은 <mark>**품질면에서도 우수**</mark>하고 병렬화가 훨씬 용이해서 <mark>**학습 시간이 굉장히 짧다**</mark>고 한다.

## 1. Introduction

순환 신경망(RNN)과 장단기 메모리(LSTM) 그리고 게이티드 순환 신경망(GRU)는 언어 모델링과 기계 번역과 같은 시퀀스 모델링 및 변환 문제에서 SOTA를 기록해왔다.

이러한 모델들은 시퀀스 위치를 계산 시간 단계에 맞추어 정렬하고, 이전 은닉 상태(ht-1)과 현재 위치(t)의 입력을 은닉 상태(ht) 시퀀스를 생성한다.

이러한 <mark>**필연적인 순차적인 특성은 학습시 병렬화를 불가능**</mark>하게 하며 <mark>**시퀀스 길이가 길어 질수록 이 문제가 심각**</mark>해 진다. 메모리 제약으로 인해서 여러 <mark>**example간의 배치 처리가 제한**</mark>되기 때문이다.

Attention 메커니즘은 <mark>**이전에도 다양한 작업**</mark>에서 시퀀스 모델링 및 변환 모델의 핵심적인 구성요소로 사용했었다.

<mark>**입력 또는 출력 시퀀스 내에서의 거리와 관계없이 의존 관계를 모델링**</mark> 할수있었기 때문이다.

하지만 이러한 어텐션 메커니즘은 <mark>**대부분 순환신경망(RNN)과 함께 사용**</mark>되었다.

이 논문에서는 Transformer를 제안하며, <mark>**순환 구조를 완전히 배제하고 입력과 출력 사이의 global 적인 의존 관계를 학습하기 위해서 오직 어텐션 메커니즘에만 의존하는 모델구조를 설계**</mark>했다.

Transformer는 <mark>**훨씬 더 높은 수준의 병렬화를 가능**</mark>하게 해주며, 이 논문의 실험에서 단 8개의 P100 GPU에서 12시간 정도만 학습해도 번역 품질에서 SOTA에 도달 할 수 있었다고 한다.
