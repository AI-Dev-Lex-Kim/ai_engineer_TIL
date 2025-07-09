- [LoRA(Low Rank Adaptation of LLM)](#loralow-rank-adaptation-of-llm)
  - [LoRA 등장 배경](#lora-등장-배경)
    - [저장 공간](#저장-공간)
    - [배포](#배포)
    - [학습](#학습)
  - [LoRA 개념](#lora-개념)
    - [왜 이렇게 적은 연산으로도 성능이 나올 수 있을까?](#왜-이렇게-적은-연산으로도-성능이-나올-수-있을까)
    - [다운 프로젝션 → 업 프로젝션 하는 이유는?](#다운-프로젝션--업-프로젝션-하는-이유는)
    - [d가 뭐야?](#d가-뭐야)
    - [r이 뭐야?](#r이-뭐야)
  - [Pytroch 코드](#pytroch-코드)
  - [정리](#정리)

---

# LoRA(Low Rank Adaptation of LLM)

## LoRA 등장 배경

대형 언어 모델(GPT-3, BERT 등)은 수십억 개 이상의 파라미터를 가지고 있다.

그걸 사전 학습으로 먼저 훈련 시켜 놓은후 각각의 downstream task에 맞게 다시 fine-tuning 한다.

<br>

전체 모델을 학습시키면, 매 task 마다 수시 GB 짜리 전체 모델을 복사해야한다.

그렇게 되면 학습, 저장, 배포, 메모리, 시간 , 돈 다 많이 드는 단점이 있다.

<br>

### 저장 공간

예를 들면

풀 파인튜닝 방식에서 여러 task 마다 각각 복사해 저장한다.

- 감정 분석용 모델: W_task1(6GB)
- 요약용 모델: W_task2 (6GB)
- 질의응답용 모델: W_task3 (6GB)

→ 3개 task만 해도 18GB 저장 공간이 필요하다.

<br>

LoRA 방식은 딱 한번만 사전 학습 모델을 가져오면 된다.

- 공통 W₀: 6GB (단 1번 저장)
- 감정 분석용 LoRA adapter: ΔW_task1(10MB)
- 요약용 LoRA adapter: ΔW_task2(10MB)
- 질의응답용 LoRA adapter: ΔW_task3(10MB)

→ 총 저장량은 6GB + 30MB = 약 6.03GB이다.

<br>

### 배포

두번째 장점으로 배포 측면도 있다.

LoRA는 서버에 Wo 하나만 불러온다.

각 요청 task에 따라 필요한 LoRA adapter만 불러와서 조립한다.

```python
model.load_adapter("lora-emotion.pt")  # 감정 분석
model.load_adapter("lora-summary.pt")  # 요약
model.load_adapter("lora-qa.pt")       # 질의응답
```

<br>

### 학습

세번째 장점으로 학습 측면도 있다.

Full Fine-Tuning은 W을 전체 학습해야한다.

LoRa는 B, A 두 행렬만 학습해서 메모리를 적게 쓰고 빠르다.

<br>

## LoRA 개념

> 기존 모델의 가중치 W는 그대로 두고(freeze),
> 추가로 아주 작은(low rank) 두 개의 행렬(B, A)을 만들어서
> task에 알맞은 W의 변화량만 학습하자.

<br>

LoRA는 궁극적으로 원하는 가중치 W를 만들기 위해서,

기존의 사전 학습된 가중치 $W_o$에 두 개의 학습 가능한 저차원 임베딩 벡터 B, A를 행렬곱한 결과를 더하는 방식이다.

```python
W = W₀ + ΔW
ΔW = B @ A
```

A → 다운 프로젝션(고차원 → 저차원)

B → 업 프로젝션(저차원 → 고차원)

<br>

이때 두 행렬(B, A)는 학습을 통해 점점 업데이트 되면,

최종적으로 `ΔW = B @ A` 형태로 표현되며 `W = W₀ + ΔW` 가 되어 모델이 task에 맞게 적응 된다.

ΔW는 downstream task에 알맞은 원하는 가중치 행렬이 되기 위해 $W_o$에 더해주는 행렬

점점 ΔW이 되어가는것이라고 생각하면됨. 학습중에는 ΔW이 아직 아님.

<br>

정리하자면, 사전 학습된 가중치 행렬을 하나도 안건드림.

Full Fine Tuning이 아님. `ΔW = B @ A` 만 학습하면서 `W` 전체를 바꾸는것 처럼 효과를 낸다.

여기서 B, A는 저차원 행렬이라 계산량이 작음.

<br>

연산량을 예시로 비교해보자.

입력 x가 d차원일때

Full Fine-Tuning은 `$W_o$ @ x = 4096 × 4096 = 1,600만` 이다.

<br>

LoRA 연산은

- A @ x → `r × d = 8 × 4096 = 32,768`
- B @ (...) → `d × r = 4096 × 8 = 32,768`
- 합계는 약 6.5만 정도 된다.

즉 원래 연산 대비 0.4% 수준 밖에 안된다.

<br>

### 왜 이렇게 적은 연산으로도 성능이 나올 수 있을까?

대부분 파라미터($W_o$)는 이미 사전 학습된 가중치이므로 잘 학습되어 있다.

특정 task에 필요한 “미세 조정”만 해주면 된다.

그래서 저랭크 행렬을 태스크에 알맞게 학습만 시켜줘도 충분한 성능이 나온다.

즉, d x d 사이즈의 가중치 행렬을 가진 ΔW이 아니어도,

작은 r x d, d x r 행렬로도 충분히 근사할 수 있다는 것이 실험적으로 증명된 것이다.

<br>

### 다운 프로젝션 → 업 프로젝션 하는 이유는?

표현력을 유지하면서 구조를 단순화 하기 위해서이다.

단순히 ΔW를 작게 만들면 표현력이 부족해질 수 있다.

하지만 ΔW를 저랭크(low-rank) 구조로 만들면, 정보 압축 + 의미 있는 학습이 가능하다.

저차원 표현을 다시 원래 차원으로 확장해서 기존 모델과 결합도 가능하다.

| 단계          | 목적                                |
| ------------- | ----------------------------------- |
| 다운 프로젝션 | 차원을 줄여 파라미터/계산량 절감    |
| 업 프로젝션   | 다시 원래 차원으로 복원해 ΔW로 사용 |

<br>

### d가 뭐야?

d는 기존 모델에서 사용하던 원래 차원, 즉 embedding size or hidden size를 의미한다.

예를들어 GPT-3에서의 hidden size = 4096이다. 그러면 d = 4096이 된다.

원래 모델의 가중치는 d x d 행렬이다.

<br>

### r이 뭐야?

r은 “low-rank”차원으로 LoRa에서 새로 추가한 임베딩 크기이다.

얼마나 **“압축된 정보로 W를 표현할지”** 조절하는 크기이다.

<br>

r이 작을수록 계산량과 파라미터 수는 준다.

r이 클수록 표현력은 높아진다.

<br>

예시를 들어보자.

GPT-3급 차원 d = 4096이다.

r = 8이다.

- A: `8 × 4096` → 32,768개 파라미터
- B: `4096 × 8` → 32,768개 파라미터

Full Fine-Tuning 했을때 더해지는 가중치 행렬인 ΔW는 `4096 × 4096` 차원 이였다.

LoRA는 r=8 짜리 행렬 3개로 ΔW에 최대한 근사치하게 표현한다.

<br>

정리하자면, r은 LoRA에서 ΔW를 표현하기 위해 저차원(hidden) 공간의 크기이며,

W 변화량을 얼마나 정교하게 표현할 것인지 결정하는 하이퍼파라미터 이다.

<br>

## Pytroch 코드

```python
import torch
import torch.nn as nn

d = 4096
r = 8

class LoRALayer(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)  # 다운 프로젝션
        self.B = nn.Linear(r, d, bias=False)  # 업 프로젝션

    def forward(self, x):
        return self.B(self.A(x))  # B @ (A @ x)

# 입력
x = torch.randn(1, d)

# 기존 W₀는 고정된 채로 사용한다고 가정
W0 = torch.randn(d, d)
baseline = x @ W0.T

# LoRA output
lora_layer = LoRALayer(d, r)
delta = lora_layer(x)
output = baseline + delta  # W₀ x + ΔW x
```

<br>

## 정리

B, A의 저차원 행렬로 인해 계산량과 파라미터 수가 매우 작다.

하지만 학습 성능은 거의 Full Fine-Tuning에 가깝다.

LoRA가 가볍지만 효과가 좋은 이유가 이런 구조 이기 때문이다.
