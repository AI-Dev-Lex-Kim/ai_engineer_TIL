- [Pytorch DataLoader DeepDive](#pytorch-dataloader-deepdive)
  - [DataLoader 파라미터](#dataloader-파라미터)
    - [dataset](#dataset)
    - [batch\_size](#batch_size)
    - [shuffle](#shuffle)
    - [drop\_last](#drop_last)
    - [num\_workers](#num_workers)
    - [collate\_fn](#collate_fn)
    - [sampler](#sampler)
    - [batch\_sampler](#batch_sampler)
    - [pin\_memory](#pin_memory)
    - [num\_worker](#num_worker)

# Pytorch DataLoader DeepDive

DataLoader는 전체 데이터 셋을 미니배치 단위로 나누어서 모델에 전달하는 역활을 해준다.

이 과정에서 데이터를 랜덤으로 섞거나, 여러개의 프로세스를 사용해서 데이터를 더 빠르게 불러오는 작업이 가능하다.

<br>

파이토치 데이터셋은 torch.uitils.data.Dataset 클래스를 상속 받아 만든다.

`__len__`과 `__getitem__` 두 개의 메소드를 반드시 포함해야한다.

<br>

Dataset는 `__getitem__`을 통해서 인덱스로 샘플의 데이터와 레이블을 반환한다.

<br>

## DataLoader 파라미터

```python
DataLoader(
		dataset,
		batch_size=1,
		shuffle=None,
		sampler=None,
		batch_sampler=None,
		num_workers=0,
		collate_fn=None,
		pin_memory=False,
		drop_last=False,
		timeout=0,
		worker_init_fn=None,
		multiprocessing_context=None,
		generator=None,
		*,
		prefetch_factor=None,
		persistent_workers=False,
		pin_memory_device='',
		in_order=True
)
```

<br>

### dataset

데이터를 불러올 데이터셋 객체이다.

`__len__`과 `__getitem__`이 존재해야한다.

<br>

### batch_size

한번에 불러올 샘플의 개수, 즉 미니배치의 크기를 지정한다.

<br>

### shuffle

`True`로 설정하면, 매 에폭 마다 데이터를 랜덤으로 섞는다.

defualt는 `False`이다. 모델 학습할때, 데이터의 순서에 모델이 과적합 되는것을

`shuffle=False`인 경우에는 몇 번을 반복해도 항상 동일한 순서의 배치가 생성된다.

<br>

`sampler` 파라미터가 지정된 경우에는 `shuffle`을 사용할 수 없다.

<br>

### drop_last

`True`로 설정하면, 데이터셋 크기가 배치 크기로 나누어 떨어지지 않을때, 마지막의 불완전한 배치를 버린다.

기본값은 `False` 이다. `False`인 경우 마지막 배치는 더 작은 크기를 갖게된다.

배치 크기가 일정해야하는 모델 구조(배치 정규화)에 유용할 수 있다.

<br>

### num_workers

데이터 로딩에 사용할 서브 프로세스의 개수를 지정한다.

기본값은 `0`이며 메인 프로세스에서 데이터를 로드한다.

`1`이상을 설정하면, 여러 개의 CPU 코어를 사용해서 데이터를 병렬적으로 불러오기 때문에, 데이터 로딩 병목현상을 해결해서 학습 속도를 크게 향상 시킨다.

예를들어 `4`를 설정하면, 배치를 위한 4개의 데이터를 4개의 워커가 하나씩 로드한다. 그 다음 배치를 위한 4개의 데이터를 또 동시에 로드하는 방식으로 로딩 시간이 크게 단축된다.

<br>

데이터 전처리 과정이 복잡하고 클수록 `num_workers`의 효과는 더욱 크다.

<br>

### collate_fn

`Dataset`에서 뽑힌 샘플 리스트를 받아서, 미니배치를 만들기 위한 전처리 함수를 직접 넣어줄 수 있다.

기본 `collate_fn`은 숫자나 텐서들을 모아 하나의 배치 텐서로 만든다.

하지만 자연어 처리 문장같이 데이터의 길이가 제각각인 경우나 복잡한 구조의 데이터를 배치로 만들고 싶을때 유용하다.

<br>

예를 들어 데이터셋이 각기 다른 길이의 시퀀스를 반환한다고 가정해보자.

```python
class VariableLengthDataset(Dataset):
    def __init__(self):
        # 데이터가 (길이, 피처) 형태의 텐서 리스트라고 가정한다.
        self.data = [torch.randn(3, 2), torch.randn(5, 2), torch.randn(2, 2)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 샘플들의 리스트를 입력으로 받는다.
def custom_collate_fn(batch_samples):
    # batch_samples =[(데이터1), (데이터2), (데이터3)]
    # 여기서 데이터는 각기 다른 길이의 텐서이다.
    # 배치 내의 텐서들을 길이를 기준으로 내림차순 정렬한다. (pack_padded_sequence를 위해)
    batch_samples.sort(key=lambda x: len(x), reverse=True)

    # 각 텐서의 길이를 저장한다.
    sequences, lengths = zip(*[(s, len(s)) for s in batch_samples])

    # torch.nn.utils.rnn.pad_sequence를 사용하여 패딩을 추가한다.
    # batch_first=True는 결과 텐서의 차원을 (배치 크기, 최대 길이, 피처) 순서로 만든다.
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    # 패딩된 시퀀스 텐서와 원래 길이를 담은 텐서를 반환한다.
    return padded_sequences, torch.tensor(lengths)

var_len_dataset = VariableLengthDataset()

# custom_collate_fn을 사용하는 DataLoader
loader_with_collate = DataLoader(
    dataset=var_len_dataset,
    batch_size=3,
    collate_fn=custom_collate_fn # 직접 정의한 함수를 지정
)

```

`collate_fn`이 없으면 `tensor`은 배치로 묶을때, 길이가 다르면 `RuntimeError`가 발생한다.

`custom_collate_fn`을 정의해서 배치 내에서 가장 긴 시퀀스 길이를 바탕으로 길이를 맞춘다.

현재 → Size(3, 2), Size(5, 2), Size(2, 2) 텐서 3개

길이가 부족한 시퀀스는 뒤에 패딩인 `0`으로 채운다. 그래서 모든 시퀀스의 길이가 가장 긴 `(5, 2)`로 통일 된다.

배치가 3개이니, `(3, 5, 2)` 형태의 최종 배치 텐서가 만들어진다.

이렇게 `collate_fn`는 크기가 정해지지 않은 ‘비정형 데이터’를 다룰 때 필수이다.

<br>

### sampler

데이터셋에서 데이터를 어떤 순서로, 어떤 방식으로 가져올지에 대한 전략을 정하는 객체이다.

`DataLoader`는 이 샘플러가 알려주는 인덱스 순서에 따라서 데이터셋에서 데이터를 가져온다.

단순히 데이터를 무작위로 섞는 `shuffle=True` 이상으로 정교한 샘플링을 할 수 있다.

예를 들어서 특정 데이터만 더 많이 더 적게 샘플링이 가능하다.

`sampler`를 직접 정의한다면 `shuffle`은 `False`여야하고 따로 `True`로 해도 무시되어 `False`가 된다.

파이토치는 몇가지 기본적인 샘플러를 제공한다.

- `SequentialSampler`: 항상 같은 순서(0, 1, 2, ...)로 인덱스를 반환한다. `shuffle=False`일 때의 기본 동작이다.
- `RandomSampler`: 인덱스를 무작위로 섞어서 반환한다. `shuffle=True`일 때의 기본 동작이다.
- `WeightedRandomSampler`: 각 샘플에 가중치를 부여하여, 특정 샘플이 더 자주 뽑히도록 할 수 있다. 데이터 불균형 문제를 다룰 때 매우 유용하다.

```python
sequential_sampler = SequentialSampler(custom_dataset)
loader_sequential = DataLoader(
    dataset=custom_dataset,
    batch_size=2,
    sampler=sequential_sampler
    # sampler를 지정했으므로 shuffle은 사용하지 않는다.
)

random_sampler = RandomSampler(custom_dataset)
loader_random = DataLoader(
    dataset=custom_dataset,
    batch_size=2,
    sampler=random_sampler
)

```

`SequentialSampler`는 에폭이 반복되어도 항상 `[0, 1]`, `[2, 3]`, ... 순서로 동일한 배치를 생성한다.

반면 `RandomSampler`는 에폭마다 완전히 다른 순서로 데이터를 섞어 배치를 만들어낸다.

<br>

`WeightedRandomSampler` 예제

클래스 불균현(class imbalance)이라고 가정해보자. 클래스 ‘A’에 속하는 데이터가 8개, B는 2개 밖에 없느 ㄴ데이터셋이 있다고 해보자.

일반적인 RandomSampler를 사용하면, 한 에폭에 클래스 B 데이터는 거의 학습하지 못한다.

`WeightedRandomSampler`을 사용하면 소수 클래스인 ‘B’의 데이터가 더 자주 샘플링 되도록 가중치를 조정할 수 있다.

```python
from torch.utils.data import WeightedRandomSampler

# 불균형한 데이터셋을 생성한다.
# 레이블 0 (클래스 A)이 8개, 레이블 1 (클래스 B)이 2개 있다.
imbalanced_data = torch.arange(10, dtype=torch.float32)
imbalanced_labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

# 각 샘플에 대한 가중치를 계산한다.
# 가중치는 (1.0 / 해당 클래스의 샘플 수)로 설정하는 것이 일반적이다.
# 이렇게 하면 소수 클래스의 가중치가 다수 클래스의 가중치보다 높아진다.
sample_weights = torch.zeros(len(imbalanced_dataset))

for i, label in enumerate(imbalanced_labels):
    class_weight = 1.0 / class_counts[label]
    sample_weights[i] = class_weight

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 생성한 샘플러를 DataLoader에 전달한다.
loader_weighted = DataLoader(
    dataset=imbalanced_dataset,
    batch_size=4, # 배치 크기는 4
    sampler=weighted_sampler
)

클래스별 데이터 수: [8 2]
각 샘플에 부여된 가중치:
tensor([0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250,
        0.5000, 0.5000])

--- WeightedRandomSampler 사용 결과 ---
Epoch 1:
  Data: [9. 8. 4. 2.], Label: [1 1 0 0]
  Data: [0. 9. 1. 8.], Label: [0 1 0 1]
  Data: [6. 5.], Label: [0 0]
  -> 샘플링된 레이블 분포: [6 4]
Epoch 2:
  Data: [8. 9. 3. 9.], Label: [1 1 0 1]
  Data: [1. 8. 5. 0.], Label: [0 1 0 0]
  Data: [8. 2.], Label: [1 0]
  -> 샘플링된 레이블 분포: [5 5]
```

원래 데이터셋에는 클래스 0이 80%, 클래스1이 20% 존재한다.

`WeightedRandomSampler` 을 사용한 결과를 보면 50:50 비율로 샘플링되는 것을 볼 수 있다.

클래스 1에 속한 데이터(인덱스 8, 9)는 0.5라는 가중치가 부여되었고

클래스 0에 속한 데이터들에는 0.125라는 낮은 가중치가 부여되었기 때문이다.

<br>

### batch_sampler

먼저 헷갈리지 않게 `sampler`와 `batch_sampler`의 차이점에 대해 알아보자.

<br>

**sampler**

다음에 어떤 데이터를 가져올지 하나씩 정한다.

“다음에 3번 인덱스 가져와”, “그 다음에는 8번 인덱스”, “그다음에는 1번 …” 이렇게 하나씩 인덱스 순서만 알려준다.

`DataLoader`는 샘플러가 알려준 인덱스 데이터를 하나씩 모으다가 `batch_size` 만큼 모이면 그것을 하나의 배치로 만든다.

즉, 랜덤으로 뽑을지, 더 자주뽑고 싶은지, 인덱스 순서대로 뽑을지 같이 인덱스 순서를 정해준다.

<br>

**batch_sampler**

다음에 만들 배치는 어떤 인덱스 들로 구성되는지 통째로 알려준다.

`DataLoader`에게 `[3, 8, 1, 5]` 이게 첫번째 배치야. `[9, 2, 0, 4`] 이게 두번째 배치야. 이런식으로 인덱스 리스트를 전달한다.

`DataLoader`는 배치 샘플러가 전달해준 인덱스 리스트를 받아 인덱스에 해당하는 샘플 데이터를 가져온다.

인덱스 요소개 4개있는 리스트를 전달받으면 4개를 배치크기로 정한다. 즉, `batch_size` 파라미터는 무시되고 `batch_sampler`가 반환한 인덱스 리스트의 크기만큼 배치 크기가 정해진다.

<br>

보통 어떤 데이터들끼리 한 배치에 묶을지에 대해 아주 구체적인 규칙이 필요할때 사용한다.

예를 들어서 ‘길이가 비슷한 문장끼리 한 배치로 묶기’, ‘동일한 사람의 얼굴 사진끼리 한 배치로 묶기’ 등 배치의 내용물 구성이 중요할때 사용한다.

<br>

`batch_sampler`를 사용하면 `batch_size`, `shuffle`, `sampler`, `drop_last` 파라미터는 모두 무시된다.

<br>

### pin_memory

GPU를 사용한 딥러닝 학습 속도를 최적화 하기위한 기능이다.

`True`로 설정하면, `DataLoader`는 GPU로 보내기전에 핀 메모리에 데이터를 저장한다.

일반 메모리 vs 핀메모리

- 일반 메모리: 운영체제는 효율적인 메모리 관리를 위해서 데이터를 물리 메모리인 RAM과 가상 메모리인 디스크 사이에서 자유롭게 이동 시킨다.
- 핀 메모리: 핀 메모리에 저장된 데이터는 운영체제가 마음대로 디스크가 옮길 수 없도록 고정(pin) 시킨다.

<br>

False

1. CPU가 하드디스크에서 데이터를 가져와서 메모리(페이지어블)에 저장한다.
   - 이 작업은 `num_worker=0`인경우 메인 프로세스가 한다. 0이 아니라면 워커프로세스가 한다.
2. 이후 GPU가 메모리에서 바로 못 가져오므로, 임시 핀 메모리 공간을 만들어 메모리에 있는 데이터를 복사한다.
   - 이 작업은 워커프로세스가 아닌 메인 프로세스가 작업한다.
3. 이후 핀 메모리에서 GPU로 옮긴다. 이 과정 전체가 동기적으로 일어나 병목이 된다.
   - 이 작업은 워커프로세스가 아닌 메인 프로세스가 작업한다.

<br>

True

1. 하드디스크에서 데이터를 가져와 핀 메모리에 바로 저장한다.
   - 기존에는 페이지어블 메모리에 옮겼다가 GPU에서 임시 핀메모리 저장함. 그 과정이 사라짐.
   - 메인 또는 워커 프로세스가 작업한다.
2. 핀 메모리의 데이터를 GPU로 보내라고 메인프로세스가 명령을 내린다.
   - `non_blocking=True` 을 하면 명령만 내리고 실제 전송이 끝날때까지 기다리지 않는다.
3. 이제 병렬 처리가 시작된다.
   - GPU: 백그라운드에서 핀 메모리에 있는 첫번째 배치 데이터를 GPU로 가져오는 작업을 한다.
   - CPU(메인 프로세스): GPU를 기다리지 않고 바름 다음 코드, 예를 들어 이전 스텝에서 전송된 데이터로 모델의 순전파를 계산한다.
   - CPU(워커 프로세스): 메인프로세스 나 GPU와 상관없이 계속해서 다음, 다다음 배치를 하드디스크에서 핀 메모리로 옮겨준다.

<br>

즉, `pin_memory=True`는 핀 메모리에 데이터를 저장하는 덕분에 병렬처리 작업으로 학습 속도를 높인다.

<br>

### num_worker

데이터를 불러오는 작업을 몇개의 CPU 프로세스를 병렬로 처리할지 결정한다.

기본 값인 0은 데이터 로딩의 모든작업을 메인 프로세스가 작업한다.

1. 메인 프로세스가 학습을 위해 GPU에서 이전 배치의 순전파/역전파를 수행한다.
2. 연산이 끝나면, 다음 배치를 가져오기 위해 **모든 연산을 멈춘다.**
3. 메인 프로세스가 직접 디스크에서 데이터를 읽고, 변환(transform)을 적용하여 배치를 만든다.
4. 데이터 로딩이 완료되면, 그제야 다음 학습 스텝을 진행한다.

<br>

1 이상인 경우

해당 개수 만큼의 백그라운드 워커프로세스를 생성한다.

이 워커들은 오로지 데이터를 미리 불러와서 준비하는것만 한다.

1. **메인 프로세스:** GPU에서 현재 배치(N번째 배치)에 대한 연산을 수행한다.
2. **워커 프로세스들:** 바로 그 시간 동안, 여러 워커 프로세스들이 CPU 코어를 나누어 사용하며 **동시에** 다음 배치(N+1, N+2번째 배치 등)를 디스크에서 읽고 변환하는 작업을 **미리 수행한다.**
3. 메인 프로세스가 N번째 배치의 연산을 끝내면, 기다릴 필요 없이 워커들이 미리 준비해 둔 N+1번째 배치를 바로 가져와 GPU로 보낼 수 있다.

<br>

참고

- [Pytorch DataLoader Docs](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
