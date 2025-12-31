# [CV 논문 리뷰] You Only Look Once: Unified, Real-Time Object Detection

논문 링크: https://arxiv.org/pdf/1506.02640

> YOLO는 CV에 큰 영향을 주었기 때문에, 이전부터 YOLO 첫번째 논문을 읽고 싶었다.
> 논문에서도 그 당시에는 <span style='background-color: #fff5b1'>**혁신적으로 새로운 구조를 적용**</span>해, 빠르고 뛰어난 성능을 보여줬다고 말하고 있다.
> 읽으면서 <span style='background-color: #fff5b1'>**Transformer랑 비슷하게 기존과 아예 다른 구조로 접근**</span>했다는것이 인상깊었다.
> 논문에서 <span style='background-color: #fff5b1'>**반복적 강조하는 부분이 YOLO라는 이름 그대로 “단 한번의 네트워크로 예측한다”라는 점을 반복하며 강조**</span>한다.
> 학습 뿐만 아니라 추론도 한번의 네트워크로 진행하며, 그에 따라 <span style='background-color: #fff5b1'>**속도가 매우 빠르지만, 성능도 뒤쳐지지 않는다는 저자의 자부심(?)이 느껴지는것 같았다.**</span>

<br>

## Abstract

YOLO는 객체 탐지를 전혀 다른 방식으로 접근한 모델이다.

기존의 객체 탐지 방식은 분류 모델을 여러번 사용하느라 구조가 느리고 복잡했다.

<br>

<span style='background-color: #fff5b1'>**분류 모델이란?**</span>

분류 모델이란 이미지 속에 무엇이 있는지만 판단하는 모델이다.

분류 모델은 물체가 어디 있는지 모르고 오로지 클래스만을 분류한다.

<br>

<span style='background-color: #fff5b1'>**과거의 객체 탐지는 이미지에서 물체가 있을만한 위치를 후보로 많이 만들어 놓고**</span>,

<span style='background-color: #fff5b1'>**그 후보 하나하나를 분류모델에 넣어서**</span> 물체가 있는지, 없는지를 판단하는 방식이였다.

<br>

YOLO는 이런 과정 없이 <span style='background-color: #fff5b1'>**이미지 전체를 한번에 보고 바운딩 박스(위치)와 클래스 확률을 예측**</span>한다.

덕분에 <span style='background-color: #fff5b1'>**탐지 과정 전체를 하나의 신경망으로 구성**</span>할 수 있고, 학습 또한 처음 부터 끝까지 통합적으로 최적화할 수 있다.

<br>

<span style='background-color: #fff5b1'>**하나의 신경망이란?**</span>

객체 탐지는 2가지 작업을 모두 해야한다.

1. 어디에 있는지
2. 무엇인지

이 두단계를 서로 다른 알고리즘이나 서로 다른 모델이 담당 했다.

<br>

예를 들어서 R-CNN은 다음과 같다.

1. 영역 후보 생성
2. 각 후보를 CNN으로 물체 분류
3. 겹치는 박스를 정리하는 후처리(NMS)

<br>

YOLO는 하나의 네트워크로 바운딩 박스와 클래스를 둘다 한번에 예측해서 탐지 과정 전체가 하나의 신경망으로 이루어진 구조이다.

<br>

이런 통합된 구조 덕분에 <span style='background-color: #fff5b1'>**YOLO는 매우 빨라 실시간 처리에 탁월하며 성능도 매우 좋다.**</span>

다만, YOLO는 속도가 빠른 만큼 <span style='background-color: #fff5b1'>**위치 예측에서 실수**</span>할 수 있지만, <span style='background-color: #fff5b1'>**배경을 물체로 잘못 인식하는 경우는 훨씬 적다.**</span>

<br>

기존의 DPM, R-CNN 보다 더 나은 성능을 보여주며, YOLO는 단순히 빠른 모델을 넘어서, <span style='background-color: #fff5b1'>**객체 탐지의 새로운 방향을 제시하는 접근법**</span>이라고 한다.

<br>

## Introduction

![](https://velog.velcdn.com/images/lexkim/post/dca682cc-91d2-410d-b012-f4a1a99850f9/image.png)

Figure 1: The YOLO Detection System. Processing images
with YOLO is simple and straightforward. Our system (1) resizes
the input image to 448 × 448, (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by
the model’s confidence.

---

기존의 객체 탐지 방식은 기본적으로 <span style='background-color: #fff5b1'>**분류기(classifier)를 탐지 용도로 사용**</span>한다.

이미지에 어떤 물체가 있는지 알기 위해, 한장의 이미지 크기와 위치를 바꿔가며 분류기에 넣어 계속 탐지하게 한다.

예를 들어서 슬라이딩 윈도우 방식이 있다.

<span style='background-color: #fff5b1'>**이미지 전체에 일정 간격으로 작은 필터를 움직여가며, 각 위치마다 분류기를 실행해서 물체를 탐지**</span>한다.

이미지 한장을 처리할때 분류기가 <span style='background-color: #fff5b1'>**수천번 반복 실행해서 속도가 매우 느리고 복잡**</span>해진다.

<br>

<span style='background-color: #fff5b1'>**R-CNN 동작 방식**</span>

1. R-CNN은 이미지에서 물체가 있을 만한 <span style='background-color: #fff5b1'>**후보 박스들을 뽑는 과정(Region Proposal)을 거친다.**</span>
2. 이 후보 박스들을 <span style='background-color: #fff5b1'>**하나씩 분류기에 넣어서 어떤 물체인지 판단**</span>한다.
3. 박스 <span style='background-color: #fff5b1'>**위치를 조금씩 조정**</span>하고,
4. 여러번 잡힌 <span style='background-color: #fff5b1'>**중복 박스를 제거**</span>하고,
5. 주변 물체들을 참고해 문맥을 보고 점수를 다시 매긴다.(사람 옆에 강아지 ok, 하늘 한가운데 강아지 x)

<br>

이렇게 여러 단계를 거쳐야 하기에 속도도 느리며, 각 단계별로 따로따로 학습되어야 하기 때문에 전체 시스템을 최적화 하기도 어렵다.

<br>

<span style='background-color: #fff5b1'>**YOLO 장점**</span>

첫번째로 <span style='background-color: #fff5b1'>**YOLO의 구조는 매우 단순**</span>하다.

<span style='background-color: #fff5b1'>**단일 CNN 하나가 여러개의 바운딩 박스와 각 박스의 클래스 확률을 동시에 예측**</span>한다.

이런 통합된 <span style='background-color: #fff5b1'>**단일 네트워크 구조**</span> 덕분에 속도도 빠르며, 학습도 최적화 할수있다.

YOLO는 속도 뿐만 아니라 기존 실시간 탐지 시스템보다 평균 정확도(mAP)가 두배 이상 높다.

<br>

두번째로 <span style='background-color: #fff5b1'>**이미지를 전체적으로 본다는 점**</span>이다.

기존의 방식인 슬라이딩 윈도우나 Region Proposal 방법은 작은 영역 단위로 물체를 판단하기 때문에 주변 상황이나 장면 전체 정보를 알지 못한다.

<br>

YOLO는 항상 전체이미지를 봐서, <span style='background-color: #fff5b1'>**각 물체가 나타날 법한 위치나 주변 관계 같은 문맥 정보(context)를 자연스럽게 학습**</span>하고 활용한다.

이 때문에 Fast R-CNN 같은 모델이 배경을 물체로 착각하는 일이 많지만, <span style='background-color: #fff5b1'>**YOLO는 배경을 오인하는 오류가 절반 이하로 줄어든다.**</span>

<br>

세번째로 물체를 <span style='background-color: #fff5b1'>**일반화해서 이해하는 능력**</span>이다.

예를 들어서 자연환경 이미지를 학습한 모델이 그림이나 예술 작품 같은 <span style='background-color: #fff5b1'>**전혀 다른 도메인에서도 R-CNN 같은 기존 모델보다 훨씬 잘 동작**</span>한다.

<br>

<span style='background-color: #fff5b1'>**왜 일반화가 잘될까?**</span>

YOLO는 이미지 전체를 한번에 보고, <span style='background-color: #fff5b1'>**물체의 전반적인 형태와 장면 문맥까지 학습**</span>하기 때문에, 배경이나 패턴을 과도하게 의존하지 않고 <span style='background-color: #fff5b1'>**본질적인 특징을 배우기 때문이다.**</span>

그래서 예상치 못한 입력에도 성능이 크게 떨어지지 않는다.

<br>

하지만 단점으로는

- 정확도 면에서는 최첨단 모델에 <span style='background-color: #fff5b1'>**조금 뒤처진다.**</span>
- 특히 <span style='background-color: #fff5b1'>**작거나 복잡한 물체**</span>의 위치를 정확히 잡는데 <span style='background-color: #fff5b1'>**어려움**</span>이 있다.

<br>

<span style='background-color: #fff5b1'>**즉, YOLO는 빠르고 일반화가 잘되는 모델이지만, 작은 물체 위치 정밀도에서는 약간의 한계가 있는 트레이드오프가 존재한다.**</span>

<br>

## Unifed Detection

![](https://velog.velcdn.com/images/lexkim/post/0632c642-6b44-4ef3-9e88-842addb1e9bb/image.png)

Figure 2: The Model. Our system models detection as a regression problem. It divides the image into an S × S grid and for each
grid cell predicts B bounding boxes, confidence for those boxes,
and C class probabilities. These predictions are encoded as an
S × S × (B ∗ 5 + C) tensor.

<br>

YOLO의 핵심은 기존 처럼 후보 박스를 뽑고, 분류기를 여러번 돌리고, 후처리를 따로하는 복잡한 구조가 아니라, <span style='background-color: #fff5b1'>**하나의 네트워크(신경망)로 통합**</span>했다는 점이다.

<br>

<span style='background-color: #fff5b1'>**S x S 그리드 방식**</span>

YOLO는 이미지를 한번의 CNN으로 나온 feature map을 S x S 그리드 셀 나눈다.

예를 들어 7 x 7 feature map이 나온다면, 각 셀별로 다음과 같은 예측을한다.

1. <span style='background-color: #fff5b1'>**x, y**</span>
   - <span style='background-color: #fff5b1'>**바운딩 박스 중심 좌표**</span>
   - 셀 내부에서 <span style='background-color: #fff5b1'>**상대 좌표로 표현됨(0~1)**</span>
   - x=0.5이면 셀 중앙
   - <span style='background-color: #fff5b1'>**바운딩 박스 개수만큼 예측**</span>
2. <span style='background-color: #fff5b1'>**w, h**</span>
   - <span style='background-color: #fff5b1'>**바운딩 박스 너비와 높이**</span>
   - 이미지 전체 크기에 대해 <span style='background-color: #fff5b1'>**상대적으로 표현됨(0~1)**</span>
   - 이미지가 400x400이면 0.5는 200
   - <span style='background-color: #fff5b1'>**바운딩 박스 개수만큼 예측**</span>
3. <span style='background-color: #fff5b1'>**confidence**</span>

   - Pr(Object) x IOU(pred, ground-truth)
   - Pr(Object)는 객체가 셀안에 <span style='background-color: #fff5b1'>**있다면 1, 없으면 0**</span>이다.
   - IOU는 <span style='background-color: #fff5b1'>**예측한 객체 좌표(x, y, w, h)와 실제 객체의 위치 좌표가 얼마나 겹치는 지에 대한 수치**</span>로 나타낸것이다.
   - <span style='background-color: #fff5b1'>**바운딩 박스 개수만큼 예측**</span>

   $$
   IOU = \frac{예측\ 박스와\ 정답\ 박스의\ 겹치는\ 영역}{예측\ 박스와\ 정답\ 박스를\ 합친\ 전체\ 영역}
   $$

4. <span style='background-color: #fff5b1'>**클래스 확률**</span>
   - Pr(Class_i | Object)
   - <span style='background-color: #fff5b1'>**클래스 확률은 각 셀마다 한번만 예측**</span>하고 <span style='background-color: #fff5b1'>**바운딩 박스 개수와는 상관없다.**</span>

<br>

<span style='background-color: #fff5b1'>**바운딩 박스**</span>

- <span style='background-color: #fff5b1'>**각 셀당 고정된 개수의 바운딩 박스를 가진다.**</span>
- <span style='background-color: #fff5b1'>**바운딩 박스의 크기**</span>는 물체의 크기를 예측한것이니, <span style='background-color: #fff5b1'>**셀의 크기보다 클 수 있다.**</span>
- 이미지를 한번에 보기때문에 여러 CNN을 거치면서 <span style='background-color: #fff5b1'>**전체 이미지의 정보를 각 셀마다 어느정도 반영하고 있다.**</span>
- 셀 안에 물체가 없어도, x, y, w, h를 예측하지만, confidence는 0에 가까워서 물체가 없음을 의미하게 된다.
- 셀 안에서 <span style='background-color: #fff5b1'>**IoU가 가장 높은 박스가 그 셀의 대표 바운딩 박스**</span>가 된다.

<br>

<span style='background-color: #fff5b1'>**후보 박스 생성**</span>

결국 모든 셀들의 대표 바운딩 박스를 남기면, <span style='background-color: #fff5b1'>**총 S x S x B 개수의 바운딩 박스가 후보 박스로 생성**</span>된다.

<br>

<span style='background-color: #fff5b1'>**클래스별 신뢰도(class-specific confidence score)**</span>

이 점수는 <span style='background-color: #fff5b1'>**“클래스가 박스안에 나타날 확률”과 “예측한 박스가 실제 객체와 얼마나 잘맞는지”를 동시에 반영**</span>한다.

<br>

$\text{class-specifi\ confidence\ score}
\\ = 클래스\ 확률 ×\ confidence
\\ = Pr(Class_i​∣Object)× \text{confidence}
\\=Pr(Class_i∣Object)×Pr(Object)×IOU(pred,\ truth)$

<span style='background-color: #fff5b1'>**추론 및 테스트시에만 사용**</span>되는 점수이다.

- “클래스 확률”과
- “각 바운딩 박스의 confidence”을 곱해서 구하게 된다.

<br>

보통 confidence score라고 줄여서 부른다.

정리하자면, <span style='background-color: #fff5b1'>**“클래스 확률”과 “객체 위치 정보”를 반영한 신뢰도**</span> 이다.

<br>

<span style='background-color: #fff5b1'>**Non-Maximum Suppression(NMS)**</span>

만약 <span style='background-color: #fff5b1'>**셀의 경계선에 걸치는 큰 물체**</span>는 여러 바운딩 박스에 존재하게 된다.

<span style='background-color: #fff5b1'>**NMS(Non-maximum suppression)이 이런 겹치는 바운딩 박스를**</span> 처리해준다.

<br>

<span style='background-color: #fff5b1'>**NMS 과정**</span>

1. 가장 높은 score를 가진 박스 선택
2. 겹치는 박스(IOU > threshold)는 제거
3. 남은 박스 중 다시 최고 score 선택
4. 반복

결과: 각 물체마다 최종적으로 하나의 바운딩 박스가 생긴다.

<br>

요약

1. CNN → feature map → S×S 셀별 B개의 박스 예측
2. class-specific confidence 계산
3. Non-Max Suppression으로 중복 박스 제거
4. 최종 물체 위치와 클래스 출력

<br>

<span style='background-color: #fff5b1'>**S x S vs 슬라이딩 윈도우**</span>

이 논문에서 “S x S 그리드” 라는 개념을 설명할때 슬라이딩 윈도우와 비슷한것 아닌가? 라고 생각이 들었다.

<br>

슬라이딩 윈도우는 다음과 같이 동작한다.

- 작은 n x n을 일정 간격으로 이미지 전체를 훑으며, <span style='background-color: #fff5b1'>**각 위치마다 분류 모델을 반복적으로 실행한다.**</span>
- 즉, 윈도우를 한칸 이동하면, 분류 모델을 사용하고 이 과정을 수백 ~ 수천 번 반복한다.
- <span style='background-color: #fff5b1'>**N개의 위치마다 CNN 반복 실행**</span>
- 연산량 = CNN 연산량 x 위치 수

<br>

YOLO

- <span style='background-color: #fff5b1'>**CNN에 이미지 전체를 넣음**</span>
- 헷갈리면 안되는 점이, CNN 레이어는 여러개이고 각 CNN 레이어에 이미지 전체를 한번에 넣어서 처리한다는 점이다.
  슬라이딩 윈도우는 필터로 이미지를 훑으며 여러번 진행한다는 점과 다르다.
- 최종 출력 S x S feature map의 <span style='background-color: #fff5b1'>**각 위치(셀)당(i, j) x, y, w, hconfidence score 만 계산**</span>한다.
- <span style='background-color: #fff5b1'>**연산량 = CNN 한번의 연산량**</span>
- 각 셀당 confidence score를 연산하지만, <span style='background-color: #fff5b1'>**CNN 연산량에 비하면 극히 작다.**</span>
- 예시
  S(feature 크기) = 7, B(바운딩 박스) = 2, C(채널) = 20
  한 셀당 연산량 = B x 5 + C = 2 x 5 + 20 = 30
  전체 예측 차원 = 7 x 7 x 30 = 1470
  최종 레이어가 <span style='background-color: #fff5b1'>**계산해야하는 숫자는 1470개 뿐**</span>이다.
  <span style='background-color: #fff5b1'>**CNN이 이미지에서 수백만개의 연산을 수행**</span>하는것을 보면 무시해도 될 정도이다.
  여기서 “숫자 5”는 정보(x, y, w, h, confidence)개수 이다.

<br>

### Network Design

![](https://velog.velcdn.com/images/lexkim/post/9e0232c2-0eb1-485f-8e20-64453bc115d6/image.png)

Figure 3: Dectection 네트워크는 24개의 CNN 레이어와 2개의 fully connect layer로 구성되어 있다.

1 x 1 CNN 층은 이전 층에서 나온 feature map의 차원을 줄이는 역활을 한다.

이 논문에서는 224 x 224 해상도의 입력 이미지를 사용해서 ImageNet 분류 작업으로 CNN층을 사전 학습한뒤, 탐지 단계에서 해상도를 두 배로 늘려서 사용했다.

<br>

YOLO 모델은 전체 구조가 CNN으로 구성되어 있다.

<br>

<span style='background-color: #fff5b1'>**24CNN + 2FC**</span>

<span style='background-color: #fff5b1'>**앞쪽의 24단계 레이어**</span>들은 <span style='background-color: #fff5b1'>**CNN 레이어로 구성**</span>되고 이미지에서 물체의 형태, 색, 윤곽선 같은 <span style='background-color: #fff5b1'>**특징을 추출**</span>한다.

<br>

마지막 2개 단계는 <span style='background-color: #fff5b1'>**Fully Connected 레이어**</span>는 이 <span style='background-color: #fff5b1'>**특징(feature map)을 보고**</span>

- 바운딩 박스 위치(x, y, w, h)
- 각 클래스의 확률
  <span style='background-color: #fff5b1'>**을 최종적으로 예측한다.**</span>

<br>

<span style='background-color: #fff5b1'>**즉, 24개의 CNN 레이어는 이미지 해석, 2개의 FC 레이어는 답을 내리는 역활로 상당히 깊고 큰 네트워크 CNN 모델이다.**</span>

<br>

<span style='background-color: #fff5b1'>**GoogleNet → YOLO**</span>

YOLO의 구조는 기본적으로 <span style='background-color: #fff5b1'>**GoogleNet처럼 깊은 CNN 구조를 참고**</span>했다.

다른점은 GoogleNet에는 복잡한 <span style='background-color: #fff5b1'>**Inception 블록**</span>이 있는데 YOLO는 그 대신

- <span style='background-color: #fff5b1'>**1 x 1 합성곱(채널 축소)**</span>
- <span style='background-color: #fff5b1'>**3 x 3 합성곱을 반복해서 더 단순하게 구성했다.**</span>

<br>

<span style='background-color: #fff5b1'>**Fast YOLO**</span>

기존 YOLO는 깊은 네트워크로 더 빠른 속도를 위해 Fast YOLO를 따로 만들었다.

<br>

Fast YOLO는

- CNN 층을 24개 → <span style='background-color: #fff5b1'>**9개로 크게 줄이고**</span>
- 각 층의 <span style='background-color: #fff5b1'>**필터 수도 줄여서**</span>
  <span style='background-color: #fff5b1'>**연산량을 대폭 줄였다.**</span>

<br>

<span style='background-color: #fff5b1'>**구조만 작게 바뀌고 학습 방식, 출력방식, 파라미터는 기본 YOLO와 동일하다.**</span>

<br>

<span style='background-color: #fff5b1'>**최종 출력은 7x7x30 텐서(행렬)**</span>

YOLO는 이미지 전체를 7x7 그리드로 나누고,

각 셀마다 30개의 값을 예측한다.

그래서 최종 출력은 7x7x30 형태이다.

- “2개의 바운딩 박스” x “5개의 값(x, y, w, h, confidence)” = 10개
- 20가지 종류의 클래스 확률 = 20개
- B x 5 + C = 2 x 5 x 20 = 총 30개

<br>

### Training

YOLO는 처음부터 객체 탐지 학습을 시키지 않는다.

먼저 이미지 분류용 CNN으로 ImageNet의 1000개 클래스를 학습 시킨다.

<br>

<span style='background-color: #fff5b1'>**왜 사전학습?**</span>

- 사전학습시 이미지의 기본적인 패턴, 엣지, 색, 형태 등을 잘배운다.
- <span style='background-color: #fff5b1'>**기초 지식을 가진 상태에서 객체 탐지 학습으로 전환하면 훨씬 빠르고 안정적으로 학습하게 된다.**</span>

1주 정도 학습후 top-5 정확도는 88%라고 한다.(당시 GooleNet 정도 성능)

<br>

<span style='background-color: #fff5b1'>**Object detection Task 변환**</span>

Pretrained YOLO 모델은 아직 이미지의 1000개 클래스를 분류하는 task만 가능하다.

<span style='background-color: #fff5b1'>**Object detection task를 수행**</span>하기 위해서는 추가적인 레이어와 리사이징이 필요하다.

1. 기존 CNN 뒤에
   - <span style='background-color: #fff5b1'>**4개의 Conv 레이어**</span>
   - <span style='background-color: #fff5b1'>**2개의 FC 레이어를**</span>
     <span style='background-color: #fff5b1'>**새로 추가**</span>한다.(이 부분은 랜덤 초기화)
2. 입력 이미지 크기를 <span style='background-color: #fff5b1'>**224x224 → 448x448로 키운다.**</span>

   이유: 객체 탐지는 <span style='background-color: #fff5b1'>**더 세밀한 정보가 필요**</span>

<br>

<span style='background-color: #fff5b1'>**최종 레이어 FC**</span>

마지막 레이어인 2개의 FC 레이어는

- 바운딩 박스 좌표(x, y, w, h)
- 클래스 확률
  을 모두 예측하게 된다.

<br>

<span style='background-color: #fff5b1'>**w, h는 전체 이미지 크기 기준으로 0~1로 정규화한다.**</span>

<span style='background-color: #fff5b1'>**x, y는 해당 셀 안에서의 상대적 위치인 0~1로 정규화한다.**</span>

즉, x, y, w, h 모두 정규화로 0~1 범위가 된다.

<br>

<span style='background-color: #fff5b1'>**leaky ReLU**</span>

$$
\phi(x) = \begin{cases}x, & \tt{if}\ \ x>0\\
0.1x, & \tt{otherwise}\end{cases}
$$

YOLO는 총 24개의 CNN 레이어와 2개의 FC 레이어로 구성된다.

<span style='background-color: #fff5b1'>**모든 CNN 레이어 뒤에는 Leaky ReLU 활성화 함수가 사용된다.**</span>

- x가 0이상이면, 그대로
- x가 음수면 0.1을 곱해준다.

하지만, FC 레이어는 2개중에 앞의 <span style='background-color: #fff5b1'>**FC 1 레이어 에만 Leaky ReLU가 사용**</span>되고 FC 2 레이어는 사용하지 않는다.

<br>

<span style='background-color: #fff5b1'>**SSE(sum-squared error)**</span>

YOLO는 <span style='background-color: #fff5b1'>**출력 전체(x, y, w, h, confidence, class) 오차를 SSE(오차 제곱의 합)으로 최적화**</span> 한다.

<br>

SSE는 구현이 매우 단순하고 계산하기도 쉽다.

하지만, 객체 탐지 성능을 높이는데 중요한 <span style='background-color: #fff5b1'>**mAP(maximizing average precision)와는 완전히 일치하지 않는 문제**</span>가 있다.

<br>

<span style='background-color: #fff5b1'>**SSE 문제1**</span>

SSE 오차함수는 위치오차(localization)와 분류 오차(classification)를 그냥 동일한 가중치로 부여한다.

<br>

예를 들어

- 바운딩 박스 위치가 10픽셀 틀린 오차
- 클래스 확률이 0.1 틀린 오차

이 두가지를 동일한 비중으로 오차를 계산하게 된다.

<br>

하지만 <span style='background-color: #fff5b1'>**실제 객체 탐지에서 중요한 것**</span>은

<span style='background-color: #fff5b1'>**위치 오차(localization error)가 훨씬 치명적이다.**</span>

클래스 확률은 조금 틀려도 덜 중요하다.

<br>

<span style='background-color: #fff5b1'>**SSE 문제2**</span>

YOLO는 7 x 7 = 49개의 셀을 예측한다.

하지만 <span style='background-color: #fff5b1'>**대부분의 셀에는 물체가 없다.**</span>

그렇게 된다면, confidence 점수가 0에 점점 가깝게 되어,

물체가 있는 셀에서 나오는 <span style='background-color: #fff5b1'>**중요한 gradient가 묻히게 되어**</span> 학습이 올바르지 않게된다.

<br>

<span style='background-color: #fff5b1'>**SSE 문제3**</span>

큰 박스와 작은 박스의 오차를 동일하게 가중한다는 문제가 있다.

예를 들어서

크기가 100인 큰 박스의 오차가 10이라고 해보자.

크기가 30인 작은 박스의 오차도 10이다.

<br>

큰 박스의 경우

정답 w = 200

예측 w = 210

오차 = 10 → Loss = 100

<br>

작은 박스의 경우

정답 w = 20

예측 w = 30

오차 10 → loss = 100

<br>

같은 오차 10이더라도 <span style='background-color: #fff5b1'>**작은 박스에서는 훨씬 치명적인 위치 오차인데, 두 경우를 동일하게 취급**</span>한다.

작은 박스에서의 미세한 오차를 제대로 반영하지 못하게된다.

<br>

이러한 문제들 때문에 <span style='background-color: #fff5b1'>**모델이 불안정**</span>해지고, <span style='background-color: #fff5b1'>**학습 초기에 발산**</span>할 수 있다.

<br>

<span style='background-color: #fff5b1'>**문제 해결**</span>

SSE 문제1 에서 <span style='background-color: #fff5b1'>**위치 오차와 클래스 오차의 가중이 똑같은 문제**</span>가 있었다.

이런 문제를 해결하기 위해, <span style='background-color: #fff5b1'>**바운딩 박스 좌표 예측에 대한 손실을 증가**</span> 시켰다.

<br>

SSE 문제2는 객체가 없는 셀들이 다수여서, <span style='background-color: #fff5b1'>**객체가 있는 셀에서 나오는 중요한 gradient가 묻히게 되는 문제**</span>가 있었다.

이를 해결하기 위해서 <span style='background-color: #fff5b1'>**객체를 포함하지 않는 박스에 대한 confidence 예측 손실은 줄였다.**</span>

<br>

SSE 문제3는 <span style='background-color: #fff5b1'>**큰박스와 작은 박스에서 나오는 오차의 비중이 같다**</span>는 문제가 있었다.

이를 해결하기 위해서 폭과 높이를 실제 크기와 똑같이 예측하는 대신에, <span style='background-color: #fff5b1'>**제급근을 해준 수치로 예측**</span>하게 했다.

<br>

예를 들어

sqrt(200) = 14.14

sqrt(210) = 14.49

변화 = 0.35

<br>

하지만 작은 값에서는 변화가 더 크게 나타남

sqrt(20) = 4.47

sqrt(30) = 5.47

변화 = 1.00

<br>

덕분에, 큰 박스의 오차는 덜 중요하게 반영되고, 작은 박스의 오차는 더 크게 반영하게 되어 성능이 개선된다.

<br>

<span style='background-color: #fff5b1'>**추가적인 성능 개선1**</span>

<span style='background-color: #fff5b1'>**특성화된 바운딩 박스 예측기(Specialization bounding-box predictor)를 사용해 성능을 향상**</span>시켰다.

<br>

셀마다 여러개의 바운딩 박스를 예측한다.

<span style='background-color: #fff5b1'>**하지만, 각 객체는 단 하나의 bounding-box가 책임지도록 한다.**</span>

이것이 Specialization bounding-box predictor이다.

Specialization bounding-box predictor는 <span style='background-color: #fff5b1'>**현재 예측된 박스들 중 IOU가 가장 높은 박스로 선택**</span>되게 된다.

<br>

이렇게 하면 Predictor 마다 특정 크기, 종횡비, 클래스에 특화되게 되어 전체 성능이 좋아진다.

<br>

<span style='background-color: #fff5b1'>**추가적인 성능 개선2**</span>

손실함수를 최적화해 성능을 향상 시켰다.

<br>

클래스

- <span style='background-color: #fff5b1'>**객체가 있는 셀만 클래스 오차를 계산**</span>하게 했다.

<br>

바운딩 박스 좌표

- 그 셀에서 IOU가 가장 높은 Predictor인 <span style='background-color: #fff5b1'>**Specialization bounding-box predictor 에게만 좌표 오차를 계산**</span>하게 했다.
- 한 객체에 대해 여러 박스가 동시에 학습되지 않고 <span style='background-color: #fff5b1'>**특정 클래스를 담당하는 Specialization bounding-box predictor만 학습하게 했다.**</span>

<br>

### Inference

추론시 네트쿼크가 단 한번만 실행된다.

1. 입력 이미지를 448x448로 리사이즈 한다.
2. 이 이미지를 한번 네트워크에 통과시킨다.
3. 네트워크는
   - 여러개의 바운딩 박스 좌표 & confidence
   - 각 셀의 클래스 확률
     을 조합해서 최종 결과를 만든다.

<br>

다른 R-CNN 처럼

- region proposal 네트워크 한 번,
- 그 후에 각 proposal마다 CNN 여러 번,
  이런 식으로 동작하지 않는다.

<br>

<span style='background-color: #fff5b1'>**중복된 바운딩 박스 제거 NMS**</span>

큰 객체나 셀 경계에 걸친 객체들은 여러 박스에서 예측되게 된다.

- 예를들어 트럭이 화면 절반을 차지하고 있다면,
- 트럭 중심은 하나의 셀에 있지만,
- 주변 셀들에서도 트럭의 일부가 보인다.
- 여러 셀이 비슷한 위치, 크기의 박스를 동시에 예측하게 된다.

<br>

이전 설명에서 <span style='background-color: #fff5b1'>**Specialization bounding-box predictor라는 개념으로 성능을 향상시켰다고 했다.**</span>

다시 간단하게 설명하면, <span style='background-color: #fff5b1'>**Specialization bounding-box predictor는 한 클래스를 하나의 바운딩 박스로 전문화해서 예측하는것이다.**</span>

<br>

하지만 여러 셀에 중복된느 객체는 여러 박스가 예측한다.

<span style='background-color: #fff5b1'>**그래서 객체당 1박스를 위해 후처리인 NVM를 한다.**</span>

<br>

<span style='background-color: #fff5b1'>**NMS(Non-Maximum Suppression)**</span>

1. 특정 클래스를 예측한 박스들을 모두 모은다.
2. 각 박스의 score를 기준으로 내림차순 정렬한다.
3. 가장 score가 높은 박스를 하나 뽑는다.
4. 이 박스와 IOU가 높은 나머지 박스들은
   - 같은 객체를 중복으로 가리키고 있다고 보고 제거(Suppress)한다.
5. 남은 박스들에 대해 3~4 단계를 반복한다.

<br>

결과적으로 하나의 score가 가장 높은박스 하나만 남고, 나머지의 겹치는 박스들은 지워지게 된다.

논문에서는 <span style='background-color: #fff5b1'>**NMS가 성능에 필수적이지는 않지만, NMS는 mAP를 2~3% 정도 향상**</span>시킨다.

<br>

### Limitation of YOLO

YOLO도 이런 획기적인 아키텍처로 성능과 속도가 매우 빠르지만, 단점도 존재한다.

<br>

1. 그리드 기반의 공간적 제약

   YOLO는 그리드 기반으로 셀을 나눈다.

   하나의 셀은 두 개의 박스와 하나의 클래스만 예측할 수 있다.

   그래서 <span style='background-color: #fff5b1'>**서로 가까이 붙어 있는 여러개의 작은 객체인 새 떼 같은 것들은 잘 잡지 못하는 공간적 제약이 존재한다.**</span>

   <br>

2. 다양한 형태의 종횡비(aspect ratio)

   YOLO는 <span style='background-color: #fff5b1'>**학습 데이터로 학습해서 바운딩 박스를 예측**</span>한다.

   때문에 <span style='background-color: #fff5b1'>**새로운 형태나 특이한 모양, 객체는 잘 예측하지 못한다.**</span>

   <br>

3. 특징 맵이 너무 거칠다(coarse feature)

   YOLO는 입력 이미지를 여러 다운샘플링 계층을 거쳐 피처맵을 만든다.

   원본 이미지의 세밀한 정보는 점점 줄어들고, <span style='background-color: #fff5b1'>**대략적인 정보만 남게 된다.**</span>

   <br>

   바운딩 박스 좌표는 <span style='background-color: #fff5b1'>**미세한 위치 정보가 필요**</span>하지만,

   <span style='background-color: #fff5b1'>**YOLO의 마지막 특징 맵이 7x7처럼 너무 작아 정확히 예측하지 못한다.**</span>

<br>

## Comparison to Other Detection Systems

R-CNN 계열

- <span style='background-color: #fff5b1'>**Selective Search**</span>로 후보 영역 박스(region proposal)를 약 2000개 뽑는다.
- 각 박스마다
  - CNN으로 특징 추출
  - SVM으로 분류
  - 선형 회귀로 박스 조정
  - NMS로 중복 제거
    이런 <span style='background-color: #fff5b1'>**복잡한 다단계 파이프라인**</span>을 거친다.
- 각 단계는 따로 튜닝해야 하고,
  CNN을 영역마다 계속 돌려야 해서 <span style='background-color: #fff5b1'>**매우 느리다**</span>
  (논문 기준 이미지 한 장당 40초 이상).
  <br>

YOLO와의 공통점, 차이

공통점

- YOLO도 “후보 박스 제안 + CNN 특징으로 점수화”라는 구조를 갖는다.

차이점

- YOLO는 <span style='background-color: #fff5b1'>**region proposal이 아니라**</span> 이미지를 <span style='background-color: #fff5b1'>**그리드로 나누고**</span>, 각 셀이 일정 개수의 박스를 예측하므로
  같은 객체를 여러 번 잡는 현상을 어느 정도 구조적으로 줄인다.
- R-CNN: 이미지당 약 2000개 박스
  YOLO: 이미지당 98개 박스
- R-CNN은 여러 모듈이 따로 학습/동작하지만,
  YOLO는 하나의 통합된 네트워크를 end-to-end로 한 번에 학습한다.

<br>

## Conclusion

1. 통합 객체 검출 모델

   - YOLO는 하나의 네트워크로 특징 추출부터 박스·클래스 예측까지 모두 처리하는 통합 모델이다.
   - 이미지 전체를 입력으로 받아 직접 검출을 학습하며, 따로 분류기나 파이프라인을 쌓을 필요가 없다.

   <br>

2. 학습 방식의 차별점

   - 기존 분류기 기반 방식은 분류 성능 위주로 학습한 뒤 검출에 재활용하는 구조였다.
   - YOLO는 <span style='background-color: #fff5b1'>**검출 성능(mAP 등)과 직접적으로 연관된 손실 함수**</span>를 사용해,
     <span style='background-color: #fff5b1'>**모든 파라미터를 한 번에 joint training 하는 end-to-end 모델**</span>이다.
     - <span style='background-color: #fff5b1'>**joint training란?**</span>
       각 단계별로 학습하지 않고, YOLO는 하나의 네트워크안에 있어서 하나의 손실함수로 역전파를 할때 모든 레이어로 흘러가며 한번에 업데이트 하는 방식

   <br>

3. 속도 측면

   - Fast YOLO는 논문 기준으로 가장 빠른 범용 객체 검출기이다.
   - YOLO는 실시간(real-time)으로 동작하면서도 <span style='background-color: #fff5b1'>**현재 기준 실시간 검출 중에서는 매우 높은 성능**</span>을 낸다.

   <br>

4. 일반화 능력
   - 자연 이미지뿐 아니라 <span style='background-color: #fff5b1'>**그림, 예술작품 등**</span> 새로운 도메인에서도 비교적 잘 동작한다.
   - 그래서 <span style='background-color: #fff5b1'>**빠르고 안정적인 객체 검출이 필요한 실제 응용(로봇, 자율주행, 실시간 모니터링 등)에 적합한 모델**</span>이다.
