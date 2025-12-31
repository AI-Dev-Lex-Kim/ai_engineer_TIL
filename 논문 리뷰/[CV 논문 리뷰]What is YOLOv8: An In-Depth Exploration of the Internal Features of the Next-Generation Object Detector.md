# [CV 논문 리뷰] What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector

논문 링크: https://arxiv.org/html/2408.15857v1

<br>

> YOLOv8 논문이 나오지 않아 YOLOv8을 연구한 논문을 읽었다.
> YOLOv8의 깊은 아키텍처 흐름 이나 layer 구조들을 자세히 분석할줄 알았지만, YOLOv8의 특징들과 장점들을 큰 특징들만 소개해주었다.
> 논문에서 있지 않은 자세한 내용들은 따로 조사한뒤 추가했다.

## Abstract

이 논문은 YOLOv8 이라는 객체 탐지 모델을 자세히 분석한 내용이다.

YOLOv8은 이전 버전인 YOLOv5보다 구조도 좋아지고 속도와 정확도도 크게 향상되었다.

<br>

이 논문에서는 YOLOv8의 핵심적인 변화들을 몇 가지로 나누어 설명한다.

<br>

첫 번쨰는 모델 구조(architecture)이다.

YOLOv8은 <mark>**CSPNet이라는 백본을 사용**</mark>해 이미지 특징을 더 효율적으로 뽑아낸다.

또 <mark>**FPN + PAN이라는 구조로 여러 크기의 물체를 더 잘 찾을 수 있도록 여러 단계의 특징을 합쳐준다.**</mark>

이런 구조 덕분에 크고 작은 물체를 모두 균형있게 탐지할 수 있다.

<br>

두번째는 <mark>**앵커(anchor)를 쓰지 않는 방식(anchor-free)**</mark>이다.

이전 YOLO들은 물체가 있을 법한 위치 후보(앵커 박스)를 미리 정해두고 학습했지만, YOLOv8은 이런 과정 없이도 바로 박스를 예측한다.

이 방식은 설정할 파라미터가 줄고, 다양한 형태의 물체를 더 자연스럽게 처리할 수 있다.

<br>

YOLOv8은 높은 정확도와 빠른 속도, 다양한 하드웨어(PC, 모바일 등)에서 실시간으로 동작할 수 있다.

추가적으로 <mark>**YOLOv8은 개발자 친화적인 기능**</mark>을 많이 포함하고있다.

파이썬 패키지와 CLI가 잘 정리되어 있어서 학습, 검증, 추론을 쉽게 진행할수있다.

<br>

## Introduction

객체 탐지 작업을 잘하기 위해 많은 알고리즘이 개발되어 왔고, 시간이 흐르면서 계속 새로운 기술들이 등장해 성능이 좋아져왔다.

당시에는 대부분의 모델들은 여러 단계로 나누어서(예: 먼저 후보 영역을 찾고, 그 다음에 그 영역을 분류) 처리했지만, YOLO는 이 과정을 한번에 해결하려고 했다.

<mark>**입력 이미지를 단 한번만 신경망에 넣어서 바로 박스 좌표와 클래스 확률을 모두 예측하게 만들었다.**</mark>

<br>

시간이 지나면서 YOLO 시리즈는 v2, v3, v4, v5 등 계속 발전해 왔다. YOLOv8은 이전보다 더 세련된 구조와 학습 방식을 사용한다.

그 덕분에 정확도, 속도, 실시간 사용성 모두를 크게 개선했다.

<br>

### <mark>**Survey Objective**</mark>

이 논문에서 가장 중요한 목표는 YOLOv8 객체 탐지 모델의 성능을 완전히 분석하고 평가하는 것이다.

특히 두가지를 집중적으로 본다.

1. YOLOv8이 <mark>**다른 최신 객체 탐지 모델들과 비교할때 얼마나 정확**</mark>한가?
2. YOLOv8이 <mark>**얼마나 빠른 속도**</mark>로 추론을 수행하는가?

YOLOv8은 크기에 따라 tiny, small, medium, large 같은 다양한 버전이 있는데, 각 버전은 속도와 정확도의 균형이 다르다.

이 논문은 <mark>**어떤 상황에서 어떤 크기의 YOLOv8이 가장 잘 맞는가를 분석하는 것도 중요한 목표**</mark>로 삼았다.

<br>

논문에서는 4가지를 깊게 살펴보았다.

1. CSPNet backbone + FPN + PAN neck 구조가 얼마나 중요한지?

   YOLOv8 구조는 크게 백본과 넥으로 나뉜다.

   - <mark>**CSPNet backbone은 이미지 특징을 얼마나 잘 뽑아내는지 영향을 준다.**</mark>
   - <mark>**FPN + PAN nect은 크거나 작은 물체를 다양한 크기에서 얼마나 잘 탐지하는지를 영향을 준다.**</mark>

   이 두가지 구조가 YOLOv8 성능에 어떤 영향을 끼치는지 분석한다.

   <br>

2. 앵커(anchor)를 쓰지 않는 방식의 장점은 무엇인지?

   <mark>**YOLOv8은 anchor-free 방식을 사용한다.**</mark>

   이 방식은 모델을 더 단순하게 만들고, 다양한 크기와 비율의 물체를 더 잘잡아 낼 수있게한다.

   이 방식이 정확도 향상에 어떤 도움을 주는지 분석한다.

   <br>

3. YOLOv8 파이썬 패키지와 CLI가 왜 중요한지?

   <mark>**YOLOv8은 개발자들이 모델을 학습,평가,배포하는 과정을 훨씬 쉽게 만들어주는 도구들을 제공한다.**</mark>

   이 편의성 자체도 논문에서 중요한 분석 대상으로 삼았다.

   <br>

4. COCO, Roboflow 100같은 데이터셋에서 얼마나 잘 나오는지?

   <mark>**YOLOv8이 실제 벤치마크에서 얼마나 잘 작동하는지 측정한다.**</mark>

   YOLOv5 등 <mark>**이전의 YOLO들과 비교**</mark>해서 얼마나 잘 발전했는지를 평가한다.

<br>

<mark>**YOLOv8은 YOLOv5를 기반으로 더 빠르고 정확하게 만들고 개발자가 쓰기 편해지도록 개선한 모델이다.**</mark>

특히 YOLOv8은 <mark>**실시간 객체 탐지(real-time object dectection)에 더 잘 맞도록 구조와 학습 방식이 크게 개선**</mark>되었다.

<br>

## Architectural Footprint of YOLOv8

![](https://velog.velcdn.com/images/lexkim/post/11199ff8-95f5-476c-b72a-e0b18a642c35/image.png)

YOLOv8은 이전 YOLO 모델들(v3, v4, v5 등)이 만들어 놓은 기반 위에서 최신 기술들을 더 얹어서 발전시킨 모델이다.

<br>

YOLO의 기본 아이디어는 다음과 같다.

- <mark>**이미지 속에서 물체가 어디 있는지(위치)**</mark>
- <mark>**그 물체가 무엇인지(클래스)**</mark>

<mark>**이 두가지를 한번의 신경망으로 동시에 해결한다.**</mark>

YOLOv8도 이 방식을 그대로 유지하면서 내부 구조를 개선해왔다.

<br>

YOLOv8은 세가지 핵심 모듈로 구성된다.

1. Backbone(특징 추출)
2. Neck(특징 결합)
3. Head(최정 예측)

<br>

### Backbone

<mark>**Backbone은 이미지에서 중요한 정보를 뽑아내는 역활이다.**</mark>

YOLOv8 backbone은 CNN으로 이루어져 있다.

이 CNN은 이미지의 다양한 패턴을 찾아낸다.

- <mark>**낮은 단계(feature map 초기)는 선, 모서리, 색 패턴 같은 단순한 형태를 찾는다.**</mark>
- <mark>**깊은 단계는 사람 얼굴의 형태, 자동차 윤곽 같은 더 의미있는 정보를 찾는다.**</mark>

이렇게 단순 → 복잡한 특징을 점점 쌓아 가는 것을 계층적 특징(hierarchical feature)이라고 한다.

<mark>**단계가 낮아질수록 feature map의 크기는 점점 작아지게 된다.**</mark>

<br>

이런 CNN 구조를 더 효율적으로 만들기 위해 <mark>**CSPNet 계열의 구조를 사용**</mark>한다.

CSPNet은

- 연산량을 줄인다.
- 메모리 사용을 줄인다.
- 정확도는 유지하거나 더 높인다.

여기에 더해서 <mark>**Depthwise Separable Convolution**</mark> 이라는 기법을 사용한다.

<mark>**기존의 CNN의 무거운 계산 구조를 더 가볍게 나눠 수행해서 같은 정확도를 유지하면서 계산량을 전반으로 줄일 수 있게 해준다.**</mark>

<br>

정리하면

<mark>**CSPNet + Depthwise Spearable Convolution을 사용해 최적의 성능을 내도록 설계했다.**</mark>

<br>

CSPNet 이란 무엇일까?

<br>

Depthwise Spearable Convolution 이 뭘까?

<br>

### Neck

![](https://velog.velcdn.com/images/lexkim/post/050fed84-535f-4774-a033-a4179b72179d/image.png)

Neck은 <mark>**Backbone에서 얻은 여러 단계의 특징을 결합하고 재구성하는 과정**</mark>이다.

Backbone은 단순한 특징에서 복잡한 특징을 추출할때, 점점 크기가 작은 feature map을 만들게된다.

- 얕은 층 → 크기가 큰 feature map(80x80) 선, 모서리, 색
- 중간 층 → 중간 크기 feature map(40x40)
- 깊은 층 → 크기가 작은 feature map(20x20) 얼굴 형태, 자동차 윤곽

<br>

feature map이 최종적으로 20x20 까지 작아진후

다음 단계는 Neck 단계로 여러 단계에서 생성된 feature map을 결합한다.

<br>

<mark>**FPN(Top-down Path)**</mark>

마지막 단계에서 생성한 20x20 feature map을 점점 키우며,

Backbone에서 같은 크기 였던 featurem map과 결합해준다.

<br>

이렇게 깊은 층의 feature map과 얕은 층의 feature map을 결합해서 서로 다른 특징 정보를 얻게된다.

<mark>**“큰 물체를 잘잡는 정보”를 “작은 물체를 찾는 층”에게도 나눠준다는 것이다.**</mark>

<br>

이 과정을 업샘플링(upsampling) 레이어에서 이루어진다.

Backbone에서 가장 깊은 단계에서 생성한 20x20 feature map을 40x40 feature map으로 확대해준다.

- “Backbone 단계중에서 생성한 40x40 feature map” 과
- “20x20 feature map을 40x40 feature map으로 업샘플링한 feauture map”을
- concat해준다.

<br>

concat한 40x40 feature map을 80x80으로 업샘플링한뒤, 똑같이 Backbone 에서 생성한 80x80 feature map과 concat 해준다.

이런 과정을 통해 단순한 특징과 뚜렷한 특징들을 결합해 풍부한 특징을 가지게 해준다.

<br>

<mark>**PAN(Bottom-up Path)**</mark>

80x80 feautre map 까지 업샘플링 하며 FPN을 해주었다.

다시 20x20까지 작게 만드는 과정에서 FPN에서 생성한 같은 크기의 feature map을 결합해준다.

<mark>**“작은 물체를 찾아내는 세밀한 정보”를 “큰 물체용 깊은 층”에게도 나눠준다.**</mark>

이 과정을 다운 샘플링(downsampling)과 합성곱으로 진행된다.

<br>

<mark>**얕은 층의 큰 feature map을 더 작게 줄여서, 깊은 층의 작은 feautre map과 크기를 맞춘다.**</mark>

stride=2 convolution

(3x3 conv, stride=2)

FPN 단계에서 최종적으로 생성한 80x80 feature map을 40x40 feature map으로 크기를 작게만든다.

- “FPN 단계에서 생성한 40x40 feature map”과
- “80x80 feature map의 크기를 작게만든 40x40 feature map”을
- concat해준다.

<br>

20x20까지 같은 작업을 반복한다.

<br>

<mark>**Concat 레이어**</mark>

<mark>**크기가 동일해진 두 feautre map을 이어서 붙여준다.(더해주는것 아님)**</mark>

feautre map A: (40x40, 128채널)

feature map B: (40x40, 192채널)

→ concat → (40x40, 320채널)

<br>

<mark>**C2f Block**</mark>

<mark>**concat된 feature map은 noise도 많아서, C2f Block 레이어로 다시 정제한다.**</mark>

YOLOv8의 대표적인 여러 conv가 연결된 특수 블록이다.

<br>

C2f Block은 Conv와 Bottleneck 2개씩으로 구성되어있다.

C뒤에 2라는 숫자도 Conv가 2개라는 뜻이다.

Bottleneck에는 2개의 Conv layer가 존재한다.

<mark>**Bottleneck에 들어오는 input과 2개의 Conv layer을 통과한 output을 concat하는 과정을 거친다.**</mark>

이것은 Conv layer을 거치면서 input의 <mark>**데이터 손실 발생을 우려해서 보완**</mark>한 것이다.

<br>

### Head

<mark>**Head 모듈은 Neck에서 생성된 정제된 특징 정보를 바탕으로 최종 예측 값을 생성하는 역활을 담당한다.**</mark>

여기에서

- 바운딩 박스 좌표
- 객체 신뢰도 점수
- 클래스 레이블

들을 같이 예측한다.

<br>

YOLOv8은 이전 YOLO 버전에서 사용되던 앵커 기반 방식에서 앵커가 필요없는 방식(achor-free)으로 바운딩 박스를 예측한다.

- 예측 과정을 단순화
- 하이퍼파라미터수 감소
- 다양한 비율과 크기 객체를 더 유연하게 적응

이런 구조로 정확도와 속도, 유연성 측면에서 더 뛰어난 결과를 보여준다.

<br>

Neck의 FPN + PAN 과정에서 각각의 크기(20x20, 40x40, 80x80) 별로 feature map을 concat해서 풍부한 정보를 만들어줬다.

<mark>**이때 생성된 3개의 Small, Middle, Large 크기의 feature map을 이용해서 바운딩 박스와 클래스 레이블을 예측한다**</mark>

<br>

## <mark>**YOLOv8 Training Methodologies**</mark>

YOLOv8은 모델 구조의 발전 뿐만아니라 학습 방식도 발전해서 높은 성능을 보여주었다.

그중에서 핵심적인 두가지 방법이 있다.

<br>

### <mark>**Advanced Data Augmentation**</mark>

더 다양한 상황을 학습할 수 있도록 <mark>**데이터 증강 방식을 사용**</mark>한다.

<mark>**모자이크 증강과 믹스업**</mark> 같은 기법을 강화했다.

<mark>**여러 이미지를 하나로 합치면서 다양한 크기, 방향, 배치를 가진 객체를 한번에 학습하도록 했다.**</mark>

덕분에 모델은 실제 환경에서 마주칠 다양한 장면에서 더 잘 적응하게 된다.

<br>

### Loss Caculation

YOLOv8의 손실함수는 모델이 더 정확하게 예측하도록 3가지의 Loss로 구성된다.

1. Focal Loss
2. IoU Loss
3. Objectness Loss

<br>

<mark>**Focal Loss Function**</mark>

YOLOv8은 분류 단계에서 <mark>**포컬 로스(Focal loss)라는 손실 함수를 사용**</mark>한다.

이 손실 함수는 <mark>**쉽게 맞추는 예시보다 어려운 예시(작은 객체, 가려진 객체, 드믈게 등장하는 객체)에 더 높은 비중을 둔다.**</mark>

데이터셋에서 자주 등장하지 않는 객체나 크기가 작아 잘 보이지 않는 객체도 더 잘 검출한다.

<br>

<mark>**IoU Loss**</mark>

IoU(Intersection over Union) 기반 손실은 바운딩 박스의 위치 정확도를 높인다.

예측된 박스와 실제 박스가 얼마나 잘 겹치는지를 계산해서 객체의 정확한 위치를 정밀하게 찾아내게 해준다.

<br>

<mark>**Objectness Loss**</mark>

이미지의 어떤 부분이 <mark>**실제로 객체를 포함하고 있을 가능성이 높은지를 판단**</mark>하도록 한다.

관심 없는 배경 역역에 시간낭비를 하지 않고, 실제 객체가 있을 만한 부분에 집중하도록 한다.

<br>

### <mark>**Mixed Precision Training**</mark>

<mark>**YOLOv8은 16비트 부동소수점 연산을 사용해서 혼합 정밀도 학습 방식을 적용했다.**</mark>

덕분에 YOLOv8은 빠르면서 효율적인 학습이 가능해지고, 리소스가 제한적인 환경에서도 안정적으로 사용할 수 있게 되었다.

<br>

### <mark>**CSP Backbone and Efficient Layer Aggregation**</mark>

YOLOv8은 <mark>**CSP(Cross Stage Partial) 보틀넥 모듈의 향상된 버전으로 불필요한 연산을 줄이고 특징 재사용을 높였다.**</mark>

FPN(Feature Pyramid Network)과 결합되서 여러 해상도의 특징을 더 효율적으로 통합하게 되어 추론속도와 전체적인 객체 탐지 성능이 개선 되었다.

<br>

### <mark>**Enhanced PANet Neck**</mark>

YOLOv5에서 사용되던 PANet 구조를 기반으로, <mark>**YOLOv8에서 PANet neck을 적용**</mark>했다.

개선된 neck 구조로 Backbone에서 Head로 전달되는 특징 정보 흐름을 최적화 하고 다양한 크기와 상황의 객체를 더 정교하게 탐지하게 되었다.

특히 작은 개체, 밀집된 환경, 복잡한 객체 탐지 문제에서 최고 수준의 성능을 가지게 되었다.

<br>

## YOLOv8 Models

YOLOv8은 가장 가벼운 YOLOv8n 부터 가장 강력한 YOLOv8x까지 구성되어 있다.

<br>

<mark>**YOLOv8n**</mark>

YOLOv8 시리즈 중에서 <mark>**가장 가볍고 빠른 모델**</mark>이다.

크기도 3.8MB로 매우 작아서 연산 자원이 제한된 환경에서 유리하다.

최적화된 CNN 레이어와 적은 파라미터 수로 경량화를 했다.

IoT 기기, 모바일 같이 효율과 속도가 중요한 디바이스에 적합하다.

<br>

<mark>**YOLOv8s**</mark>

YOLOv8의 기본 모델이며 9M 파라미터를 가진다.

속도와 정확도의 균형이 좋고 CPU와 GPU에서 모두 <mark>**안정적으로 추론 성능**</mark>을 보여준다.

SPP와 PANet이 적용되어서 특징들이 결합이 더 강화되고 작은 객체 탐지 성능도 높아졌다.

<br>

<mark>**YOLOv8m**</mark>

25M 파라미터를 가진 중간급 모델로, 효율성과 정확도의 균형을 가졌다.

<mark>**Backbone과 Neck이 더 깊고 넓어져서 다양한 데이터셋과 객체 탐지 상황에서 뛰어난 성능**</mark>을 가진다.

정확도가 중요한 실시간 어플리케이션에 적합하며 계산 자원 부담을 고려한 효율성도 유지하고 있다.

<br>

<mark>**YOLOv8l**</mark>

55M 파라미터를 가진 대형 모델이다.

정밀도가 필요한 작업을 위해서 만들어졌으며, 더 복잡한 특징 추출 구조와 정교해진 어텐션 메커니즘이 적용되서 <mark>**고해상도 이미지에서 작은 객체나 세밀한 구조를 더 정확하게 감지**</mark>할 수 있다.

의료 영상 분석, 자율 주행처럼 높은 정밀도가 필수인 분야에 적합하다.

<br>

<mark>**YOLOv8x**</mark>

YOLOv8 시리즈 중에서 가장 크고 강력한 모델로 90M 파라미터를 가진다.

가장 높은 mAP를 보여주며, <mark>**정확도가 절대적으로 중요한 상황에서 사용**</mark>한다.

감시 시스템, 정밀 산업 검사 등에서 좋은 성능을 보이지만, 높은 연산 자원이 필요하며 실시간 처리를 위해서는 고성능 GPU가 필요하다.

<br>

## YOLOv8 Annotation Format

어노테이션은 텍스트 파일에 저장해야한다.

<mark>**이미지에 포함된 객체 마다 한줄씩 기록**</mark>한다.

각 줄은 클래스 번호와 바운딩 박스 좌표 정보를 포함하며,

바운딩 박스는 이미지 크기를 기준으로 정규화된 <mark>**중심 좌표(center_x, center_y)와 너비(width), 높이(height) 값으로 표현**</mark> 해야한다.

<br>

형식:

<class><center_x><center_y><width><height>

예시:

`0 0.492 0.403 0.212 0.315`

<br>

위의 예시는 클래스 0번에 해당하는 객체가 중심좌표(0.492, 0.403), 너비 0.212, 높이 0.315를 가진 바운딩 박스로 표시된다는 의미이다.

<br>

추가적으로 <mark>**YAML 파일을 같이 사용**</mark>해야한다.

이 파일에 <mark>**모델 아키텍처와 클래스 레이블 정보를 정의**</mark>해야한다.

Rofoflow, VOTT, LabelImg, CVAT같은 어노테이션 도구를 사용하면 YOLOv8 형식에 맞추기 위해 변환이 필요할 수 있다.

대부분 YOLO 형식으로 직접 내보내기 기능을 제공하거나, 간단한 변환 도구가 포함되어 있어서 쉽게 처리할수있다.

<br>

## YOLOv8 Labelling

YOLOv8 개발사인 Ultralytics는 Roboflow를 권장한다.

아래 표에서 YOLOv8과 호환되는 플랫폼이다.

| <mark>**Integration Platform**</mark> | <mark>**Functionality**</mark>                                                                                                      |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <mark>**Deci**</mark>                 | YOLOv8 모델의 자동 최적화 및 양자화(quantization)를 지원하여 추론속도를 높이고 모델의 크기를 줄인다.                                                                  |
| <mark>**ClearML**</mark>              | YOLOv8 모델의 학습 실험 관리, 추적, 원격 학습 기능 제공. 협업 및 확장 가능한 머신러닝 운영 환경을 지원.                                                               |
| <mark>**Roboflow**</mark>             | 데이터셋 라벨링, 증강, 내보내기를 한 번에 처리할 수 있는 솔루션. <mark>**YOLOv8과 직접 호환**</mark>되어 데이터 준비 과정을 단순화. |
| <mark>**Weights and Biases**</mark>   | 클라우드 기반 학습 추적, 하이퍼파라미터 튜닝, 시각화 기능 제공. 실험 관리와 모델 성능 모니터링을 효율적으로 수행 가능.                                                |

> 이전에 YOLOv8을 사용한 프로젝트에서 Roboflow를 사용 해본적이 있다.
> Roboflow의 인터페이스와 데이터 증강 같은 기능들이 좋았다.

<br>

## Conclusion

YOLOv8은 <mark>**CSPNet 백본과 개선된 (FPN + PAN) Neck을 통합**</mark>해서 특징 추출과 다중 스케일 객체 탐지 성능을 크게 향상 시켰다.

<mark>**앵커 프리(anchor-free)방식**</mark>으로 전환하고 <mark>**모자이크 및 믹스업**</mark>과 같은 고급 데이터 증강 기법을 도입해서 다양한 데이터셋에서 정확도와 일반화를 높였다.
