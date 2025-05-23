# nn.BatchNorm2d

입력 값의 평균0, 표준편차1 이 되므로 파라미터 업데이트가 안정적이다.

너무 큰 값이나, 너무 작은 값으로 인해 발생하는 gradient vanishing/exlpoding 문제를 해결할 수 있다.

<br>

```python
classtorch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

| 파라미터              | 설명                                                    |
| --------------------- | ------------------------------------------------------- |
| `num_features`        | 입력 채널 수 (예: `Conv2d(3, 32, ...)`라면 여기선 `32`) |
| `eps`                 | 분모가 0이 되는 것을 막기 위한 작은 수 (안정성 목적)    |
| `momentum`            | running mean/var 업데이트에 사용되는 모멘텀 값          |
| `affine`              | `gamma`, `beta` 파라미터를 학습할지 여부                |
| `track_running_stats` | `True`면 inference할 때 running mean/var 사용           |

<br>

인풋으로 `[N, C, H, W]`가 올것으로 기대한다.

mean, std를 사용해 정규화를 해준다.

<br>

nn.BatchNorm2d는 `running_mean`, `running_var`을 사용한다.

`model.train()` 상태일때, running을 foward가 실행될때마다 지속적으로 업데이트 해준다.

`model.eval()` 상태에서는 train mode의 running_mean, running_var을 사용한다.

배치가 진행될때마다 running/mean,std는 점점 전체 데이터셋의 분포와 거의 유사하게 된다.

<br>

기본 `momentum=0.1` → 새 배치의 통계 10% 반영, 이전 것 90% 유지

```python
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var  = (1 - momentum) * running_var  + momentum * batch_var
```

$$
y=\frac{x−E[x]}{\sqrt{Var[x]+ϵ}}∗γ+β
$$

eps를 주어서 분모가 0이 되는것을 방지한다.

평균을 0, 분산을 1로 만들어서 정규화해준다.

```python
x_hat = (x - μ) / sqrt(σ² + ε)     ← 평균 0, 분산 1로 만듦
y = γ * x_hat + β                  ← 학습 가능한 스케일, 이동
```

- `μ`는 배치 평균
- `σ²`는 배치 분산
- `γ`는 scale (학습 파라미터)
- `β`는 shift (학습 파라미터)

<br>

| 이미지 타입                  | 채널 수(C) | num_features |
| ---------------------------- | ---------- | ------------ |
| RGB                          | 3          | 3            |
| 흑백(Grayscale)              | 1          | 1            |
| 16채널 피처맵 (예: CNN 출력) | 16         | 16           |

<br>

```python
# RGB 이미지 feature map → [N, 3, H, W]
bn_rgb = nn.BatchNorm2d(num_features=3)

# 흑백 이미지 → [N, 1, H, W]
bn_gray = nn.BatchNorm2d(num_features=1)
```

<br>

> BatchNorm2d를 두 번 썼다면, 각각의 BatchNorm 레이어는 자신의 running_mean, running_var를 독립적으로 가지고, 독립적으로 업데이트한다.
