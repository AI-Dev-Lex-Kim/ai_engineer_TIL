# OpenAI LLM Fine-Tuning을 위한 데이터 준비 및 분석

아래와 같은 데이터 준비 및 분석을 설명한다.

1. 형식 유효성 검사
2. 토큰 수 계산
   - 메시지 리스트로부터 토큰 수
   - 어시스턴스의 메시지로부터 토큰 수
   - 값들의 분포
3. 데이터 잠재적 문제 검사
4. API 비용 토큰 수 추정

<br>

필요 라이브러리

```python
import json
import tiktoken # 토큰(token) 계산용
import numpy as np
from collections import defaultdict
```

<br>

### 데이터 로딩(Data loading)

먼저 예제 `JSONL` 파일에서 챗 데이터셋(chat dataset)을 로드한다.

```python
data_path = "data/toy_chat_fine_tuning.jsonl"

# 데이터셋 로드
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# 초기 데이터셋 통계
print("예시 수:", len(dataset))
print("첫 번째 예시:")
for message in dataset[0]["messages"]:
    print(message)
```

```python
Num examples: 5
First example:
{'role': 'system', 'content': 'You are a happy assistant that puts a positive spin on everything.'}
{'role': 'user', 'content': 'I fell off my bike today.'}
{'role': 'assistant', 'content': "It's great that you're getting exercise outdoors!"}
```

<br>

## 1. Format validation(형식 유효성 검사)

데이터셋의 각 대화가 fine-tuning API에서 예상하는 형식을 준수하는지 확인하기 위해 **다양한 오류 검사를 수행**할 수 있다.

오류는 디버깅을 용이하게 하기 위해 성격에 따라 분류된다.

- 데이터 타입 확인: 데이터셋의 각 항목이 딕셔너리(`dict`)인지 확인한다. 오류 유형(Error type): `data_type`.
- 메시지 리스트 존재 여부: 각 항목에 `messages` 리스트가 있는지 확인한다. 오류 유형(Error type): `missing_messages_list`.
- 메시지 키 확인: `messages` 리스트의 각 메시지에 `role`과 `content` 키가 포함되어 있는지 확인한다. 오류 유형(Error type): `message_missing_key`.
- 메시지 내 인식할 수 없는 키: 메시지에 `role`, `content`, `weight`, `function_call`, `name` 이외의 키가 있는 경우 기록한다. 오류 유형(Error type): `message_unrecognized_key`.
- 역할 유효성 검사: `role`이 "system", "user", 또는 "assistant" 중 하나인지 확인한다. 오류 유형(Error type): `unrecognized_role`.
- 내용 유효성 검사: `content`가 텍스트 데이터를 가지고 있으며 문자열(`string`)인지 확인한다. 오류 유형(Error type): `missing_content`.
- 어시스턴스 메시지 존재 여부: 각 대화에 assistant로부터의 메시지가 하나 이상 있는지 확인한다. 오류 유형(Error type): `example_missing_assistant_message`.

<br>

아래 코드는 이러한 검사를 수행하고, 발견된 각 오류 유형에 대한 개수를 출력한다. 이는 데이터셋을 디버깅하고 다음 단계를 위해 준비되었는지 확인하는 데 유용하다.

```python
# 형식 오류 검사
format_errors = defaultdict(int)

for ex in dataset:
    # ex가 dict 타입인지 확인한다
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    # ex 딕셔너리에서 "messages" 키를 가져온다. 없으면 None을 반환한다.
    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    # messages 리스트의 각 message에 대해 반복한다
    for message in messages:
        # message에 "role"이나 "content" 키가 없는 경우 오류를 기록한다
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        # message의 키 중 허용된 키("role", "content", "name", "function_call", "weight") 외에 다른 키가 있는지 확인한다
        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1

        # message의 "role"이 허용된 역할("system", "user", "assistant", "function")이 아닌 경우 오류를 기록한다
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1

        # message에서 "content"와 "function_call"을 가져온다
        content = message.get("content", None)
        function_call = message.get("function_call", None)

        # content와 function_call이 모두 없거나 content가 문자열이 아닌 경우 오류를 기록한다
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    # messages 리스트에 어시스턴스 역할의 메시지가 없는 경우 오류를 기록한다
    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

# 발견된 오류가 있으면 출력한다
if format_errors:
    print("발견된 오류:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("오류가 발견되지 않았음")
```

<br>

## 2. 토큰 계산 유틸리티(Token Counting Utilities)

노트북의 나머지 부분에서 사용할 몇 가지 유용한 유틸리티를 정의한다.

```python
# "cl100k_base" 인코딩을 가져온다
encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    """메시지 리스트로부터 토큰 수를 반환한다."""
    num_tokens = 0
    for message in messages:
        # 모든 메시지는 기본 토큰을 가진다
        num_tokens += tokens_per_message
        for key, value in message.items():
            # 메시지 내의 각 키-값 쌍에 대해 값의 토큰 수를 더한다
            num_tokens += len(encoding.encode(value))
            # 키가 "name"인 경우, 추가 토큰을 더한다
            if key == "name":
                num_tokens += tokens_per_name
    # 대화의 끝을 나타내는 토큰을 추가한다
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    """어시스턴스의 메시지로부터 토큰 수를 반환한다."""
    num_tokens = 0
    for message in messages:
        # 메시지의 역할이 "assistant"인 경우
        if message["role"] == "assistant":
            # 해당 내용의 토큰 수를 더한다
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    """값들의 분포를 출력한다."""
    print(f"\n#### {name}의 분포:")
    print(f"최소 / 최대: {min(values)}, {max(values)}")
    print(f"평균 / 중앙값: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
```

<br>

## 3. 데이터 경고 및 토큰 수

가벼운 분석을 통해 데이터셋에서 누락된 메시지와 같은 잠재적인 문제를 식별하고 메시지 및 토큰 수에 대한 통계적 통찰력을 제공할 수 있다.

- 누락된 시스템/사용자 메시지: **"system" 또는 "user" 메시지가 누락된 대화 수**를 센다. 이러한 메시지는 어시스턴스의 행동을 정의하고 대화를 시작하는 데 중요하다.
- 예시당 메시지 수: **각 대화의 메시지 수 분포를 요약**하여 대화 복잡성에 대한 통찰력을 제공한다.
- 예시당 총 토큰 수: **각 대화의 총 토큰 수를 계산하고 분포를 요약**한다. 파인튜닝 비용을 이해하는 데 중요하다.
- 어시스턴스 메시지의 토큰: **대화당 어시스턴스 메시지의 토큰 수를 계산하고 이 분포를 요약**한다. 어시스턴스의 상세함 수준을 이해하는 데 유용하다.
- 토큰 한도 경고: **최대 토큰 한도(16,385 토큰)를 초과하는 예시가 있는지 확인**한다. 이러한 예시는 파인튜닝 중에 잘리게 되어 잠재적인 데이터 손실을 초래할 수 있다.

```python
# 경고 및 토큰 수 계산
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    # 시스템 메시지가 없는 경우 카운트
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    # 사용자 메시지가 없는 경우 카운트
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    # 각 대화의 메시지 수를 리스트에 추가
    n_messages.append(len(messages))
    # 각 대화의 토큰 길이를 계산하여 리스트에 추가
    convo_lens.append(num_tokens_from_messages(messages))
    # 각 대화의 어시스턴스 메시지 토큰 길이를 계산하여 리스트에 추가
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

print("시스템 메시지가 없는 예시 수:", n_missing_system)
print("사용자 메시지가 없는 예시 수:", n_missing_user)
print_distribution(n_messages, "예시당 메시지 수(num_messages_per_example)")
print_distribution(convo_lens, "예시당 총 토큰 수(num_total_tokens_per_example)")
print_distribution(assistant_message_lens, "예시당 어시스턴스 메시지 토큰 수(num_assistant_tokens_per_example)")

# 16,385 토큰 제한을 초과하는 예시 수를 계산
n_too_long = sum(l > 16385 for l in convo_lens)
print(f"\n{n_too_long}개의 예시가 16,385 토큰 제한을 초과할 수 있으며, 파인튜닝 중에 잘릴 것이다")
```

```python
시스템 메시지가 없는 예시 수: 1
사용자 메시지가 없는 예시 수: 1

예시당 메시지 수(num_messages_per_example)의 분포:
최소 / 최대: 2, 9
평균 / 중앙값: 3.8, 3.0
p5 / p95: 2.0, 6.6000000000000005

예시당 총 토큰 수(num_total_tokens_per_example)의 분포:
최소 / 최대: 26, 8032
평균 / 중앙값: 1648.4, 45.0
p5 / p95: 26.8, 4863.6

예시당 조수 메시지 토큰 수(num_assistant_tokens_per_example)의 분포:
최소 / 최대: 4, 8000
평균 / 중앙값: 1610.2, 10.0
p5 / p95: 6.0, 4811.200000000001

0개의 예시가 16,385 토큰 제한을 초과할 수 있으며, 파인튜닝 중에 잘릴 것이다
```

<br>

## 4. API 비용 예측

파인튜닝에 사용될 총 토큰 수를 추정하여 비용을 대략적으로 계산한다.

토큰 수가 증가함에 따라 파인튜닝 작업 기간도 길어진다는 점을 유의해야 한다.

```python
# 가격 및 기본 에포크(n_epochs) 추정
MAX_TOKENS_PER_EXAMPLE = 16385
TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

# 기본 에포크(epoch) 수를 목표 에포크 수로 설정
n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)

# 학습 예시 수와 목표 에포크 수를 곱한 값이 최소 목표 예시 수보다 작은 경우 에포크 수 조정
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
# 학습 예시 수와 목표 에포크 수를 곱한 값이 최대 목표 예시 수보다 큰 경우 에포크 수 조정
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

# 데이터셋에서 과금될 총 토큰 수를 계산 (최대 토큰 수를 초과하는 부분은 제외)
n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)

print(f"데이터셋에는 학습 중 요금이 부과될 약 {n_billing_tokens_in_dataset}개의 토큰(token)이 있다")
print(f"기본적으로, 이 데이터셋에 대해 {n_epochs} 에포크(epochs) 동안 학습할 것이다")
print(f"기본적으로, 약 {n_epochs * n_billing_tokens_in_dataset}개의 토큰(token)에 대해 요금이 부과될 것이다")
```

```python
데이터셋에는 학습 중 요금이 부과될 약 4306개의 토큰(token)이 있다
기본적으로, 이 데이터셋에 대해 20 에포크(epochs) 동안 학습할 것이다
기본적으로, 약 86120개의 토큰(token)에 대해 요금이 부과될 것이다
```

<br>

참고

- [Data preparation and analysis for chat model fine-tuning](https://cookbook.openai.com/examples/chat_finetuning_data_prep)
