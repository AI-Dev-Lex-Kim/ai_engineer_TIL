# cGAN

# cGAN이란?

cGAN은 판별자(Discriminator)와 생성자(Generator)을 사용해서 이미지를 생성한다.

생성자는 가짜 이미지를 생성한다. 가짜 이미지로 판별자가 진짜 이미지라고 판별하게 속이는 것이며 학습한다.

판별자는 가짜 이미지를 가짜 이미지라고 제대로 판별하는것을 학습한다.

이렇게 서로 경쟁 하며 학습한다.

<br>

랜덤한 이미지가 생성하는것이 아니라 생성자에 조건(Conditional)을 넣어주어 조건에 맞는 이미지를 생성하게끔 학습한다.

예를들어 y=3(dress)라는 클래스 조건을 넣어준다면, dress에 맞는 이미지를 생성하게 학습한다.

판별자도 y=3이라는 조건을 주어 입력 이미지가 dress인 진짜 이미지 인지 판별하며 학습한다.

<br>

## 노이즈 벡터

노이즈 벡터는 정규분포에 따른 실제 데이터와 무관한 랜덤 값을 바탕으로 생성된 벡터이다.

잠재 공간이라고 부른다.

노이즈 벡터의 값을 바탕으로 이미지를 생성한다.

따라서 이미지는 다양하게 생성된다.

<br>

### 노이즈 벡터의 구성은?

정규분포를 따르는 무작위한 값이 있는 벡터로 이루어져있다.

값들은 정규 분포 또는 균등 분포를 사용한 값이다.

`[  0.32, -0.71,  1.11,  0.05, -1.22, ..., 0.84 ]  ← 총 100개 실수`

<br>

클래스가 10개라면 보통 10개의 차원이 있다.

너무 많은 잠재공간이 있다면, 이미지의 특징을 더 정밀하게 표현하겠지만, 과도한 정밀함으로 생성된 이미지에 노이즈가 발생할 수 있다.

<br>

### 랜덤한 값인 이유는?

- 다양성 확보 → z이 배치마다 매번 다르다. 그렇게 다르기에 다양한 이미지를 생성할 수 있다.
- 분포 학습 → 분포를 이미지 공간으로 매핑하는 함수가 된다. 결과적으로 latent(노이즈 벡터) → image 변환을 학습

<br>

### 노이즈 벡터 값을 정규분포 vs 균등분포 중 어떤것을 써야할까?

정규분포 → 평균 0, 표준편차1 `torch.randn(batch_size,100)`

균등분포 → -1 ~ 1 사이 균일 `torch.rand(batch_size, 100) * 2 - 1`

<br>

둘다 가능하지만 정규분포가 더 자주 사용됨.

중심 집중, smooth한 분포, 미분 가능성 측면에서 유리하기 때문이다.

<br>

\*균등분포: 정의된 구간에서 모든 구간이 동일한 분포

\*정규분포: 평균과 분산 만으로 특성을 모두 설명, 평균을 그래프의 중심으로 좌우대칭인 종 모양

<br>

## 학습

### “노이즈 벡터를 바탕으로 이미지를 만드는 함수를 학습” 한다.

Generator의 파라미터(Conv 가중치)를 학습하면서 인풋을 이미지로 변환 해주는 함수로 점점 학습해간다는 뜻이다.

```python
z → Linear → ReLU → reshape → ConvTranspose2d → ... → output image
```

최종적으로 (1, 28, 28) 형태의 이미지가 나온다.(이상하게 나올지언정, 점점 나아짐)

<br>

### 어떻게 학습하는 것인지?

인풋 노이즈 벡터(z)와 조건(y = 3 → dress)를 넣어서 가짜 이미지 G(z, y) 생성한다.

Discriminator가 가짜인지 진짜인지 판별한후 손실을 계산한다.

이 손실을 바탕으로 Generator의 모든 가중치를 역전파로 업데이트 한다.

결국 Generator의 가중치가 학습을 통해 z가 최대한 진짜 이미지로 속일 수 있도록 학습한다.

<br>

## 판별자, 생성자 모델 학습하는 방법은?

### 판별자 학습

판별자 모델은 처음부터 진짜 이미지를 인풋으로 받았을때, 진짜 이미지라고 판별하지 못한다.

왜냐면 처음에는 파라미터들이 랜덤값이 들어가 있기 때문이다.

따라서 판별자 모델이 진짜 이미지를 진짜 이미지라고 판별하게 하는 학습을 시켜야한다.

<br>

거기서 더 나아가서 판별자 모델은 처음에 가짜 이미지를 가짜 이미지라고 판별하지 못한다.

왜내햐면 처음에 파라미터들이 랜덤값이 들어가 있기 때문이다.

따라서 판별자 모델이 가짜 이미지를 가짜 이미지라고 판별하게 학습을 시켜야한다.

<br>

즉,

가짜 이미지는 가짜 이미지로,

진짜 이미지는 진짜 이미지로 판별 할 수 있게 학습 시켜야한다.

<br>

그럴려면 손실 값이 2개가 나와야한다.

1. 가짜 이미지를 가짜 이미지로 판별한뒤 그 차이가 얼마나 나는지에 대한 손실
2. 진짜 이미지를 진짜 이미지로 판별한뒤 그 차이가 얼마나 나는지에 대한 손실

<br>

그 두 개를 더한 손실로 역전파 수행한 뒤, 옵티마이저로 가중치를 조절한다.

<br>

### 생성자 학습

판별자가 가짜 이미지를 진짜 이미지로 판별한뒤 그 차이가 얼마나 나는지에 대한 손실

이 손실로 생성자에 역전파 수행한 뒤, 옵티마이저로 가중치를 조절한다.

<br>

### 만약 모든 학습이 끝난후 z의 모든 원소가 같은 값인 2개의 z를 각각 넣는다면 똑같은 이미지가 나오는지?

훈련된 모델은 가중치가 고정되어 같은 이미지가 나온다.

<br>

### 만약 동일한 z를 넣었는데 결과가 바뀐다면?

1. 모델이 학습이 안됐다.
2. 내부에서 랜덤 요소가 고정되지 않았을 가능성

   1. torch.manual_seed()
   2. torch.Generator()

   를 통해 seed를 고정해서 실험 재현성 보장

<br>

## 코드 분석

```python
import torch
import torch.nn as nn

image_size = 28      # Fashion MNIST 이미지 크기
num_classes = 10     # 10개 클래스
latent_dim = 100     # 잠재 공간 차원

# 생성자 (Generator)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 레이블 임베딩
        self.label_emb = nn.Embedding(num_classes, num_classes)

				# 이미지 사이즈에 4를 나누어준다.
				# nn.Upsample을 2번 거치기 때문이다.
				# 28 ÷ 2 ÷ 2 = 7
        self.init_size = image_size // 4  # 7

        self.l1 = nn.Sequential(
		        # 완전 연결층(FC) 채널 128, 7 x 7 사이즈인 벡터로 변환해준다.
		        # (잠재공간 100 + 클래스 수 10), (128 * 7 * 7)
		        # 128개의 채널을 해주었다. 애초에 저해상도 였다보니 큰수를 하지 않았다.
            nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size),

            # inplace=True를 통해 값을 바로 덮어써 메모리를 절약한다.
            nn.ReLU(inplace=True)
        )
				# Conv와 Upsampling을 반복하여 28 x 28 이미지를 생성한다.
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), # 채널수 128에 대해 배치 정규화 해준다.
            nn.Upsample(scale_factor=2),  # 7 → 14
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 채널을 64로 줄인다.
            nn.BatchNorm2d(64, 0.8), # 채널 64에 대해 정규화 해준다(momentum=0.8)
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 14 → 28
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1), # 채널 1(흑백)으로 변환해준다.
            nn.Tanh()  # 출력 범위 [-1, 1]
        )

    def forward(self, noise, labels):
		    # 라벨을 임베딩 해준다.
		    # 라벨링으로 인풋값의 변경이 있어도 보다 부드럽게 이미지 적용이 가능하다.
        label_input = self.label_emb(labels)

        # 노이스 벡터와 라벨 임베딩을 합쳐준다.
        # 그로 인해 아웃풋 이미지가 라벨 클래스에 맞게 생성된다.
        # 라벨 임베딩은 파라미터가 있어 매번 가중치가 조정이 된다.
        gen_input = torch.cat((noise, label_input), -1)

        # FC를 통해 feature map 사이즈의 벡터로 변형시켜준다.
        out = self.l1(gen_input)

        # (b, 128, 7, 7) shape로 변경해준다.
        # 기본적으로 PyTorch의 텐서구조 기본은 앞에 batch size가 온다
        # nn.Linear로 나온 out.size(0)으로 shape의 첫번째 차원인 배치크기를 가져온다.
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 판별자 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 레이블을 단일 채널 값으로 임베딩
        self.label_emb = nn.Embedding(num_classes, 1)

        self.model = nn.Sequential(
            # kernel_size=3, stride=2, padding=1을 통해 28x28 -> 14x14로 다운 샘플링
            # feature map 채널도 2 -> 64개
            nn.Conv2d(1 + 1, 64, kernel_size=3, stride=2, padding=1),  # 28 → 14
            nn.LeakyReLU(0.2, inplace=True),

						# kernel_size=3, stride=2, padding=1 14x14 -> 7x7 다운 샘플링
						# 64 -> 128 채널
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),      # 14 → 7
						# 128채널에 대해 배치정규화로 수행해 학습 안정성을 높인다.
						# Conv2 레이어를 거치며 feature map의 값의 크기가 커질수있다. 정규화를 통해 평균0, 분산1로 정규화해준다.
						# 학습이 가능한 파라미터가 존재하는 스케일, 쉬프트 파라미터를 적용해주기 때문에, 숫자를 단지 작게 해주는것 뿐만아니라
						# 학습 안정화, 그레디언트 개선..등등 해준다.
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

						# 1차원으로 펼쳐준다.
            nn.Flatten(),

            # (128 x 7 x 7)의 벡터를 하나의 출력으로 나오게 해준다.
            nn.Linear(128 * (image_size // 4) * (image_size // 4), 1),   # 128*7*7

						# 출력 값을 0 ~ 1 범위로 압축해서 진짜 이미지일 확률을 반환한다.
            nn.Sigmoid()
        )

    def forward(self, img, labels):
		    # fake img를 배치 사이즈로 저장
        batch_size = img.size(0)

        # 정수라벨 임베딩하여 (batch_size, 1) 형태 벡터로 변환
        label = self.label_emb(labels)

        # batch_size을 앞으로 b라고 부르겠습니다.
        # (batch_size, 1) 벡터를 4차원으로 변환 (b, 1, 1, 1)
        label = label.view(batch_size, 1, 1, 1)

        # (b, 1, 1, 1) -> (b, 1, 28, 28)로 확장
        # torch.expand으로 하나의 숫자 레이블을 28x28 크기만큼 같은 숫자로 모두 채워서 확장시킨다.
        label = label.expand(batch_size, 1, image_size, image_size)
        # 이미지와 레이블을 채널 차원에서 연결
        # (b, 1, 28, 28)인 이미지와 (b, 1, 28, 28)인 라벨 채널을 합쳐 (b, 2, 28, 28)로 만들어준다.
        d_in = torch.cat((img, label), 1)

        # nn.Sequential 를 통과 시켜 가짜/진짜 확률을 계산한다. -> 결과 (b, 1) 형태
        # 결과 0 ~ 1 사이 실수를 반환해준다.
        validity = self.model(d_in)
        return validity
```

### `self.init_size = image_size // 4` 인풋 사이즈를 4로 나눈 이유는?

`nn.Upsample(scale_factor=2)`을 cGAN에서 2번을 한다. 각각 2배씩 보간을 한다.

따라서 최초 인풋 사이즈가 28 ÷ 2 ÷ 2 = 7이 되어야한다.

만약 업샘플 레이어가 3개 있다면, 28 ÷ 2 ÷ 2 ÷ 2 = 28 // 8 = 3 이되어야한다.

<br>

### `nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size)` 코드 분석

노이즈 벡터를 이미지 feature map의 형태로 확장시켜준다.

`nn.Linear(노이즈 벡터 개수 + 원소 개수, 채널 개수 * H * W)`

ex) `nn.Linear(100 + 10, 128 * 7 * 7)`

이후 나온 아웃풋을 `output.view(batch_size, 128, 7, 7)` shape 형태로 CNN block에 넘긴다.

<br>

### Generator의 노이즈 벡터 크기인 `(256, 7, 7)`이 `(128, 7, 7)`에 비해 뭐가 달라질까? 풍부한 피처맵으로 더 정확한 이미지가 추출되지 않을까?

채널이 많아지면 피처맵의 표현력이 늘어난다.

하지만 파라미터 개수도 엄청 늘어난다.

(256 _ 7 _ 7 = 12544개)로 많아진다. 학습이 어려워지고 과적합 가능성이 높다.

128 보다 질감, 섀도우, 곡선, 버튼 디테일까지 더 자세해 진다.

하지만 그만큼 모델이 무거워진다.

그래서 단순한 데이터셋은 128이면 충분하고 복잡한 고해상도 데이터셋은 256, 512, 1024까지도 쓴다.

<br>

## 파라미터 가중치 조절

### Generator의 어떤 것들이 가중치 조정돼?

- `nn.Linear`:
  - 업데이트: W, b
  - 역활: z + y를 CNN이 처리 가능한 feature map으로 변환)
- `nn.Conv2d` or `nn.ConvTranspose2d`
  - 업데이트: W, b
  - 역활: 이미지 해상도 키우면서 구조를 생성
- `nn.BatchNorm2d`
  - 업데이트: : scale, shift 파라미터
  - 역활: 학습 안정화 및 속도 향상
- 비학습 계층
  - nn.ReLU() → 파라미터 없음
  - nn.Upsample(scale_factor=2) → 단순 업샘플링, 학습 안함
  - nn.Tanh() → 비선형 활성화 함수, 파라미터 없음

<br>

### 파라미터수가 2배 많아지는데 왜 학습이 어려워지고 과적합 가능성이 높아질까?

파라미터가 많으면 피처맵이 많다는 뜻.

특징을 많이 표현할 수 있다는 것인데, 그렇다면 인풋을 학습할때 인풋의 특징들을 굉장히 많이 가지고 있게 된다는 것이다.

따라서 모델 훈련시 파라미터 가중치를 조정을 하면서, 파라미터가 많으니(인풋에 대해 굉장히 특징이 많아, 사실적으로 표현) 아웃풋도 굉장히 정답과 가까워진다.

그렇다면 테스트 데이터셋을 추론할때는 세세한 부분까지도 표현하는 그 많은 파라미터가 훈련 인풋을 훈련했으니, 테스트 데이터셋의 인풋은 잘 추측하지 못하고 훈련 인풋에 과적합할 수 있다.

<br>

## 조건 y

### y인 조건 레이블이라는 것은? 어떤 형태인지?

조건 레이블 y 라는것은 클래스가 있다면, 그 클래스 중 하나의 라벨을 의미한다.

```python
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
...
9: Ankle boot
```

y = 3 이면 Dress

<br>

### y의 형태는?

1. 정수 클래스 인덱스(LongTensor int64)

   ```python
   y = torch.tensor([3])  # 클래스 3 = Dress
   ```

   이 상태는 Generator가 직접 사용할 수 없다. → Embedding이 필요하다

2. one-hot 벡터(FloatTensor)

   ```python
   y = 3  →  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

   y_onehot = F.one_hot(y, num_classes=10).float() # one-hot 벡터 형태 변환
   input = torch.cat([z, y_onehot], dim=1) # z와 y를 연결해주어 input으로 넣어준다.
   ```

3. 임베딩 벡터(nn.Embedding)

   ```python
   self.label_emb = nn.Embedding(10, 10)
   y_emb = self.label_emb(y)  # (batch_size, 10)
   ```

   임베딩은 One-hot보다 더 부드럽고,

   학습을 통해 각 클래스 간의 유사성을 파악할 수 있다.

   (ex: 드레스랑 셔츠가 비슷한 방향일수있음)

<br>

### y를 왜 넣어야하는지?

y가 없으면 Generator은 무작위한 이미지를 생성한다.

y가 있으면 Generator은 특정 클래스의 이미지를 만들도록 유도해 학습한다.

<br>

### y 클래스 레이블에 대해 손실을 계산 한다는것인지?

Decriminator는 y(클래스 레이블)에 대해 얼마나 잘 맞는지를 기준으로 손실을 계산한다.

1. 진짜/가짜 판단
2. `D(G(z, y), y)` 을 보고 조건 y에 맞는 이미지 인지 판별

Generator은 y에 맞게 이미지를 생성한다.

<br>

### z와 y를 합치는 이유

Generator이 이미지가 어떤 클래스에 속해야 하는지를 알게하기 위해서이다.

<br>

비유

z → 백지에 무작위 낙서 하는 손

y → 낙서의 주제(예: 강아지)

z + y → 강아지 느낌의 무작위 낙서

G(z, y) → 강아지 느낌의 무작위 낙서 이미지를 생성

<br>

## 다른 이미지 나오는 이유

### z1, z2, z3가 다른데도 비슷한 이미지가 나오는 이유는?

모두 정규 분포에서 뽑혔기 때문에 너무 극단적으로 다르지 않다.

Generator가 그 공통 구조를 보고 학습한다.

<br>

### 클래스가 10개면, 각각의 클래스에 해당하는 파라미터가 따로 있어야 하지 않는지? 그렇지 않고 공통된 파라미터로 가중치를 조정하면 모든 이미지가 비슷하게 나오는것이 아닌지?

클래스 별로 각각의 파라미터를 가지기 위해 nn.Embedding을 한다.

클래스 별로 고유한 파라미터(벡터)를 하나씩 만들어서 학습하게 하는 구조이다.

<br>

중요한 포인트

> Generator 전체는 모든 클래스에 대해 동일한 가중치를 공유한다.
> 조건 y의 파라미터는 해당하는 y의 임베딩 벡터만 업데이트한다.

<br>

코드

```python
nn.Embedding(num_classes=10, embedding_dim=10)
y_emb = label_emb(torch.tensor([3]))  # → 클래스 3에 해당하는 임베딩 벡터 하나 선택
```

이 벡터만 Generator에 들어가고 해당 파라미터만 업데이트 된다.

<br>

## 임베딩

### 임베딩(Embedding) 이란? 임베딩 하면 어떻게 되는지?

임베딩이란?

정수형을 실수 벡터로 변환

정수형 클래스 인덱스 → 실수 벡터

y = 3 (Dress) → `[0.12, -0.83, 0.45, ..., 0.01]` 이런 **실수 벡터**

<br>

### 사용하는 이유?

하나의 정수가 여러개의 실수로 구성된다.

클래스 중에 비슷한 클래스가 존재할 수 있다. 이때 임베딩으로 인한 실수 값으로 유사성을 학습할 수 있다.

임베딩 된 클래스 인덱스는 학습 가능한 파라미터로, 훈련 도중 계속 바뀌고 업데이트 된다.

<br>

장점1

유사한 클래스간 관계 학습으로 유사한 클래스들 끼리는 점점 임베딩 벡터가 가까워 질수있다.

결과적으로 더 부드럽곡 자연스러운 생성이 가능하다.

<br>

장점2

one-hot은 메모리에 저장하지 않고 매번 생성한다.

embedding은 파라미터로 가중치를 조정하기에 메모리에 저장하고 가져온다.

<br>

메모리 점유율은 embedding이 높아 무겁지만, 속도는 훨씬 빠르다.(인덱스에서 가져오므로)

<br>

코드

```python
self.label_emb = nn.Embedding(num_classes, num_classes)

def forward(self, z, y):
        y_emb = self.label_emb(y)                  # (batch_size, num_classes)
        input = torch.cat([z, y_emb], dim=1)       # (batch_size, latent + num_classes)
        x = self.l1(input)
        ...
```

<br>

### 임베딩을 하면 정수 하나가 실수형태로 여러개 나오는지? 그 여러개의 실수들이 하나의 정수를 표현하는지?

그 실수 전체가 그 정수의 의미를 담고 있다.

이 벡터는 학습을 통해 계속 바뀌고, 그 클래스 레이블에 맞게 의미있게 변하게 된다.

이 실수들은 학습 가능한 파라미터로 초기에는 랜덤하게 나온 값이다.

```python
label_emb = nn.Embedding(10, 4)  # 10개 클래스, 4차원 임베딩
```

<br>

### 임베딩이 업데이트가 되므로써 “이미지가 부드럽게 전환된다”는 말이 무슨 뜻인지? 왜 임베딩을 써야하는지?

입력이 조금 바뀌었을때 출력 이미지가 급격히 깨지지 않고, 자연스럽게 변화한다는 뜻이다.

<br>

> z나 y가 조금 바뀌면 이미지도 조금 바뀌고
> z나 y가 크게 바뀌면 이미지도 크게 바뀐다

이게 잘된다면 모델이 잘 일반화 되고, 조건을 잘 반영하고 있다는 뜻이다.

<br>

만약 Generator가 정수로만 학습한다면?

- y = 3 (Dress) → 제대로 생성
- y = 4 (Coat) → y의 값이 3이 아니라 4이므로 이미지가 완전 깨짐 / 완전 다른 결과가 나옴
- y = 3.01 → 의미 없는 이상한 중간 이미지가 나온다.

→ 모델이 y의 작은 변화에 민감하게 반응하고 전혀 일반화가 안됨

→ 이런 모델은 조건에 약하고 불안정하며 생성 품질이 낮다.

<br>

```python
y = 3 → Dress
y = 3.2 → Dress에 가까운 Coat
y = 3.8 → Coat 느낌이 섞인 Dress
y = 4 → Coat
```

→ 이렇게 y의 연속적인 변화에 따라 이미지가 점진적으로 바뀐다면,

→ Generator는 y의 벡터를 잘 이해하고 있구나 볼수있다.

<br>

### 부드럽게 전환되는 모델의 장점은?

1. 일반화가 잘됨

   → 작은 변화에 부드럽게 반응하는 모델은 새로운 z, y 조합에도 안정적이다.

2. latent editing(잠재 공간 수정)

   → z나 y를 조절하면서 속성을 조절할 수 있다.(예: 신발 높이 점점 키우기)

3. interpolation(전이)

   → 이미지간 전이 가능, 다른 클래스가 추가되도, 전이 학습이 가능하다.

4. 의미 학습

   → 모델이 벡터 공간의 의미를 파악하고 있다는 증거

<br>

### 결론

부드러운 전환이란, 입력 값(z, y)이 바뀌어도 출력 이미지도 매끄럽게 반응한다는것!

마치 조이스틱을 살짝 밀면 살짝움직이고 세게 밀면 확 움직이는것 처럼.
