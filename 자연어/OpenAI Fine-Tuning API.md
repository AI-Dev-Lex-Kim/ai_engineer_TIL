[OpenAI Fine-Tuning API Docs](https://platform.openai.com/docs/api-reference/fine-tuning)

---

# OpenAI Fine-Tuning API

## Install

```bash
pip install openai
```

<br>

## 파인튜닝 작업 생성

파인튜닝 작업 생성

```python
from openai import OpenAI
from openai.types.fine_tuning import SupervisedMethod, SupervisedHyperparameters

client = OpenAI()

client.fine_tuning.jobs.create(
  training_file="file-abc123",
  validation_file="file-abc456"
  model="gpt-4o-mini",
  method={
    "type": "supervised",
    "supervised": SupervisedMethod(
      hyperparameters=SupervisedHyperparameters(
        n_epochs=2,
        batch_size=16,
        learning_rate_multiplier=1e-5,
      )
    ),
    "seed": 42,
    "suffix": "custom-model-name"
  }
)
```

`POST https://api.openai.com/v1/fine_tuning/jobs`

<br>

### **Request body**

- **`model`** string, **필수**
  파인튜닝할 모델의 이름이다. 지원되는 모델 중 하나를 선택할 수 있다.
- **`training_file`** string, **필수**
  학습 데이터가 포함된, 업로드된 파일의 ID이다.
  파일 업로드 방법은 [파일 업로드](https://platform.openai.com/docs/api-reference/files/create) 문서를 참고해야 한다.
  데이터셋은 반드시 JSONL 파일 형식이어야 한다. 또한, 파일을 업로드할 때 `purpose`를 `fine-tune`으로 지정해야 한다.
  파일의 내용은 모델이 `chat` 또는 `completions` 형식을 사용하는지, 혹은 파인튜닝 방법이 `preference` 형식을 사용하는지에 따라 달라져야 한다.
- **`metadata`** map, **선택**
  객체에 첨부할 수 있는 최대 16개의 키-값 쌍의 집합이다. 객체에 대한 추가 정보를 구조화된 형식으로 저장하고, API나 대시보드를 통해 객체를 조회할 때 유용하다.
  키는 최대 64자의 문자열이고, 값은 최대 512자의 문자열이다.
- **`method`** object, **선택**
  파인튜닝에 사용되는 방법이다. - **`type`** string **Required**
  메서드의 유형이다. `supervised`, `dpo` 또는 `reinforcement` 중 하나이다. - **`supervised`** object Optional
  supervised fine-tuning method 를 위한 환경 설정 - **`hyperparameters`** object Optional
  파인튜닝 작업에 사용되는 하이퍼파라미터 - **`batch_size`**(“auto" or integer / Optional) **:** 각 배치에 포함된 예제의 수이다. - **`learning_rate_multiplier`**("auto" or number)**:** 학습률 - **`n_epochs`**: ("auto" or integer / Optional / Defaults to auto): 에폭 수
- **`seed`** integer 또는 null / **선택**
  **시드는 작업 결과의 재현성을 제어**한다.
  동일한 시드와 작업 매개변수를 전달하면 대부분 동일한 결과가 생성되지만, 드문 경우 결과가 다를 수 있다.
  시드를 지정하지 않으면 자동으로 생성된다.
- **`suffix`** string 또는 null / **선택**
  기본값은 `null`이다. **파인튜닝된 모델 이름에 추가될 최대 64자의 문자열**이다.
  예를 들어, `suffix`로 "custom-model-name"을 지정하면 `ft:gpt-4o-mini:openai:custom-model-name:7p4lURel`과 같은 모델 이름이 생성된다.
- **`validation_file`** string 또는 null, **선택**
  **검증 데이터가 포함된, 업로드된 파일의 ID**이다.
  이 파일을 제공하면, 해당 데이터는 **파인튜닝 중에 주기적으로 검증 지표를 생성**하는 데 사용된다.
  이 지표들은 파인튜닝 결과 파일에서 확인할 수 있다.
  학습 파일과 검증 파일에 동일한 데이터가 중복으로 포함되어서는 안 된다.
  데이터셋은 반드시 JSONL 파일 형식이어야 하며, 파일을 업로드할 때 `purpose`를 `fine-tune`으로 지정해야 한다.

<br>

### **반환값 (Returns)**

`fine-tuning.job` 객체를 반환한다.

```python
{
  "object": "이 객체의 유형이 'fine_tuning.job', 즉 파인튜닝 작업임을 나타낸다.",
  "id": "파인튜닝 작업의 고유 식별자(ID)이다.",
  "model": "파인튜닝의 기반으로 사용될 기본 모델의 이름이다.",
  "created_at": "작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "fine_tuned_model": "파인튜닝이 완료된 후 생성될 모델의 이름이다. 'null'은 작업이 아직 시작되지 않았거나 완료되지 않았음을 의미한다.",
  "organization_id": "이 작업을 소유한 조직의 고유 식별자이다.",
  "result_files": [
    "파인튜닝 결과 파일의 ID 목록이 담길 배열이다. 작업이 아직 결과를 생성하지 않았으므로 비어있다."
  ],
  "status": "파인튜닝 작업의 현재 상태이다. 'queued'는 작업이 실행 대기열에 있으며 시작을 기다리고 있음을 의미한다.",
  "validation_file": "모델 성능 검증에 사용될 파일의 ID이다. 'null'은 검증 파일을 사용하지 않음을 나타낸다.",
  "training_file": "모델 학습에 사용될 훈련 파일의 ID이다.",
  "method": {
    "type": "적용된 파인튜닝의 방법을 명시한다. 여기서는 'supervised'(지도 학습) 방식이다.",
    "supervised": {
      "hyperparameters": {
        "batch_size": "배치 크기이다. 'auto'는 시스템이 훈련 데이터를 기반으로 최적의 값을 자동으로 설정함을 의미한다.",
        "learning_rate_multiplier": "학습률 배수이다. 'auto'는 시스템이 자동으로 값을 설정함을 의미한다.",
        "n_epochs": "에포크 수이다. 'auto'는 시스템이 자동으로 값을 설정함을 의미한다."
      }
    }
  },
  "metadata": "사용자가 정의한 키-값 쌍의 메타데이터를 저장하는 공간이다. 'null'은 설정된 메타데이터가 없음을 의미한다."
}
```

<br>

### API

```bash
curl https://api.openai.com/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{ -> --data(body, 본문)
    "training_file": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
    "model": "gpt-4o-mini"
  }'

```

```python
import requests
import os

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/jobs",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    },
    json={
        "training_file": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
        "model": "gpt-4o-mini"
    }
).json()
```

<br>

## 파인튜닝 결과 상태 리스트

파인튜닝 작업에 대한 상태 업데이트를 가져온다.

```python
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.list_events(
  fine_tuning_job_id="파인튜닝 작업 ID",
  limit=2
)
```

**fine_tuning_job_id(string / Required): 이벤트를 가져올 파인튜닝 작업 ID**

**limit**: 검색할 이벤트의 수

<br>

```python
{
  "object": "이 객체의 유형이 'list'임을 나타낸다. 즉, 여러 항목을 담고 있는 목록 형태이다.",
  "data": [
    {
      "object": "목록에 포함된 각 항목의 유형이다. 'fine_tuning.job.event'는 파인튜닝 작업과 관련된 특정 사건(이벤트)을 의미한다.",
      "id": "이벤트의 고유 식별자(ID)이다.",
      "created_at": "이벤트가 발생한 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
      "level": "이벤트의 심각도 수준이다. 'info'는 정보성 메시지, 'warn'은 경고, 'error'는 오류를 의미한다.",
      "message": "이벤트에 대한 사람이 읽을 수 있는 설명이다. 여기서는 파인튜닝 작업이 성공적으로 완료되었음을 알린다.",
      "data": "이벤트와 관련된 추가적인 구조화된 데이터이다. 'null'은 추가 데이터가 없음을 의미한다.",
      "type": "이벤트의 종류를 나타낸다. 'message'는 일반적인 로그 메시지를 의미한다."
    },
    {
      "object": "목록에 포함된 두 번째 이벤트 객체이다.",
      "id": "두 번째 이벤트의 고유 식별자이다.",
      "created_at": "두 번째 이벤트가 발생한 시간이다.",
      "level": "이벤트의 심각도 수준으로, 'info'이다.",
      "message": "이벤트에 대한 설명이다. 새로운 파인튜닝 모델이 생성되었고 그 모델의 이름이 무엇인지 알려준다.",
      "data": "추가 데이터가 없음을 나타낸다.",
      "type": "이벤트의 종류로, 'message'이다."
    }
  ],
  "has_more": "이 목록 뒤에 더 많은 이벤트가 있는지 여부를 나타내는 불리언(boolean) 값이다. 'true'는 더 가져올 이벤트가 있음을 의미하며, 페이지네이션(pagination)에 사용된다."
}
```

<br>

## 파인튜닝 체크포인트 목록

```bash
curl https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123/checkpoints \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

```python
from openai import OpenAI
import os

client = OpenAI()
fine_tuning_job_id = "ftjob-abc123"
checkpoints_page = client.fine_tuning.jobs.list_checkpoints(job_id=fine_tuning_job_id)

print(checkpoints_page.model_dump_json(indent=2))
```

```python
import requests
import os

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/jobs",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    },
    json={
        "training_file": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
        "model": "gpt-4o-mini"
    }
).json()
```

<br>

## 파인튜닝 작업 정보

파인튜닝 작업에 대한 정보

```python
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.retrieve("파인튜닝 작업 ID")
```

```json
{
  "object": "이 객체의 유형을 나타낸다. 'fine_tuning.job'은 파인튜닝 작업을 의미한다.",
  "id": "파인튜닝 작업의 고유 식별자(ID)이다.",
  "model": "파인튜닝의 기반으로 사용된 기본 모델의 이름이다.",
  "created_at": "작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "finished_at": "작업이 완료된 시간을 나타내는 유닉스 타임스탬프이다.",
  "fine_tuned_model": "파인튜닝을 통해 최종적으로 생성된 새로운 모델의 고유 이름이다.",
  "organization_id": "이 작업을 생성한 조직의 식별자이다.",
  "result_files": [
    "파인튜닝 과정에서 생성된 결과 파일의 ID 목록이다. 학습 손실(loss)이나 정확도(accuracy) 같은 지표가 포함된다."
  ],
  "status": "파인튜닝 작업의 현재 상태이다. 'succeeded'는 성공, 'running'은 진행 중, 'failed'는 실패를 의미한다.",
  "validation_file": "모델 성능 검증에 사용된 파일의 ID이다. 'null'은 검증 파일을 사용하지 않았음을 나타낸다.",
  "training_file": "모델 학습에 사용된 훈련 파일의 ID이다.",
  "hyperparameters": {
    "n_epochs": "전체 훈련 데이터셋을 몇 번 반복하여 학습할지를 나타내는 에포크(epoch) 수이다.",
    "batch_size": "한 번의 학습 단계에서 사용할 데이터 샘플의 개수이다.",
    "learning_rate_multiplier": "기본 학습률(learning rate)에 적용되는 배수 값으로, 학습 속도를 조절한다."
  },
  "trained_tokens": "파인튜닝 과정에서 모델이 학습한 총 토큰의 수이다.",
  "integrations": "W&B(Weights & Biases)와 같은 외부 로깅 서비스와의 연동 정보를 담는 배열이다.",
  "seed": "학습 결과의 재현성을 보장하기 위해 사용된 난수 시드(seed) 값이다.",
  "estimated_finish": "작업이 진행 중일 때 예상 완료 시간을 나타내는 유닉스 타임스탬프이다.",
  "method": {
    "type": "적용된 파인튜닝의 방법을 명시한다. 여기서는 'supervised'(지도 학습) 방식이다.",
    "supervised": {
      "hyperparameters": {
        "n_epochs": "지도 학습 방법에 사용된 에포크 수이다.",
        "batch_size": "지도 학습 방법에 사용된 배치 크기이다.",
        "learning_rate_multiplier": "지도 학습 방법에 사용된 학습률 배수이다."
      }
    }
  }
}
```

<br>

## 파인튜닝 이어하기

파인튜닝 이어하기.

```python
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.resume("파인튜닝 작업 ID")
```

```python
{
  "object": "이 객체의 유형이 'fine_tuning.job', 즉 파인튜닝 작업임을 나타낸다.",
  "id": "파인튜닝 작업의 고유 식별자(ID)이다.",
  "model": "파인튜닝의 기반으로 사용된 기본 모델의 이름이다.",
  "created_at": "작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "fine_tuned_model": "파인튜닝이 완료된 후 생성될 모델의 이름이다. 'null'은 작업이 아직 완료되지 않아 모델이 생성되지 않았음을 의미한다.",
  "organization_id": "이 작업을 소유한 조직의 고유 식별자이다.",
  "result_files": [
    "파인튜닝 결과 파일의 ID 목록이 담길 배열이다. 작업이 아직 결과를 생성하지 않았으므로 비어있다."
  ],
  "status": "파인튜닝 작업의 현재 상태이다. 'paused'는 사용자의 요청 또는 시스템에 의해 작업이 일시 중지되었음을 의미한다.",
  "validation_file": "모델의 성능 검증에 사용될 파일의 ID이다.",
  "training_file": "모델 학습에 사용될 훈련 파일의 ID이다."
}
```

<br>

## 파인튜닝 작업 정지

파인튜닝 작업 정지

```python
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.pause("파인튜닝 작업 ID")
```

```python
{
  "object": "이 객체의 유형이 'fine_tuning.job', 즉 파인튜닝 작업임을 나타낸다.",
  "id": "파인튜닝 작업의 고유 식별자(ID)이다.",
  "model": "파인튜닝의 기반으로 사용된 기본 모델의 이름이다.",
  "created_at": "작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "fine_tuned_model": "파인튜닝이 완료된 후 생성될 모델의 이름이다. 'null'은 작업이 아직 완료되지 않아 모델이 생성되지 않았음을 의미한다.",
  "organization_id": "이 작업을 소유한 조직의 고유 식별자이다.",
  "result_files": [
    "파인튜닝 결과 파일의 ID 목록이 담길 배열이다. 작업이 아직 결과를 생성하지 않았으므로 비어있다."
  ],
  "status": "파인튜닝 작업의 현재 상태이다. 'paused'는 사용자의 요청 또는 시스템에 의해 작업이 일시 중지되었음을 의미한다.",
  "validation_file": "모델의 성능 검증에 사용될 파일의 ID이다.",
  "training_file": "모델 학습에 사용될 훈련 파일의 ID이다."
}
```

<br>

## 지도 학습 방식의 파인튜닝 훈련 데이터셋 형식

지도 학습 방식의 파인튜닝할때 훈련 데이터셋의 구조이다.

```python
{
  "messages": [
    {
      "role": "메시지를 생성한 주체의 역할이다. 'user'는 최종 사용자를 의미한다.",
      "content": "메시지의 내용이다. 사용자의 초기 질문 '샌프란시스코 날씨가 뭐야?'가 담겨있다."
    },
    {
      "role": "'assistant'는 AI 모델을 의미한다.",
      "tool_calls": [
        {
          "id": "도구 호출에 대한 고유 식별자이다. 나중에 이 ID를 사용하여 도구의 응답을 특정 호출과 연결한다.",
          "type": "호출되는 도구의 유형이다. 여기서는 'function'이다.",
          "function": {
            "name": "호출할 함수의 정확한 이름이다. AI 모델이 사용 가능한 도구 목록에서 'get_current_weather'를 선택했다.",
            "arguments": "함수에 전달될 인자들을 담고 있는 JSON 형식의 문자열이다. AI가 사용자의 질문을 해석하여 'location'과 'format' 값을 결정했다."
          }
        }
      ]
    }
  ],
  "parallel_tool_calls": "모델이 여러 함수를 병렬로 호출할 수 있는지 여부를 나타내는 불리언 값이다.",
  "tools": [
    {
      "type": "제공되는 도구의 유형으로, 'function'임을 명시한다.",
      "function": {
        "name": "AI가 호출할 수 있는 함수의 이름이다. 이 이름은 영문, 숫자, 밑줄(_)만 포함해야 한다.",
        "description": "해당 함수가 어떤 기능을 하는지에 대한 설명이다. AI는 이 설명을 보고 어떤 상황에서 이 함수를 사용해야 할지 판단한다.",
        "parameters": {
          "type": "함수에 전달될 인자들의 전체적인 타입을 정의한다. 항상 'object'여야 한다.",
          "properties": "함수가 받는 각 인자(parameter)의 이름, 타입, 설명을 JSON 스키마 형식으로 정의하는 곳이다.",
          "required": "함수 호출 시 반드시 포함되어야 하는 필수 인자들의 이름 목록이다. 여기서 'location'과 'format'은 필수이다."
        }
      }
    }
  ]
}
```

<br>

## 파인튜닝 작업 객체 형식

```python
{
  "object": "이 객체의 유형을 나타낸다. 'fine_tuning.job'은 파인튜닝 작업을 의미한다.",
  "id": "파인튜닝 작업의 고유 식별자(ID)이다.",
  "model": "파인튜닝의 기반으로 사용된 기본 모델의 이름이다.",
  "created_at": "작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "finished_at": "작업이 완료된 시간을 나타내는 유닉스 타임스탬프이다.",
  "fine_tuned_model": "파인튜닝을 통해 최종적으로 생성된 새로운 모델의 고유 이름이다.",
  "organization_id": "이 작업을 생성한 조직의 식별자이다.",
  "result_files": [
    "파인튜닝 과정에서 생성된 결과 파일의 ID 목록이다. 학습 손실(loss)이나 정확도(accuracy) 같은 지표가 포함된다."
  ],
  "status": "파인튜닝 작업의 현재 상태이다. 'succeeded'는 성공을 의미한다.",
  "validation_file": "모델 성능 검증에 사용된 파일의 ID이다. 'null'은 검증 파일을 사용하지 않았음을 나타낸다.",
  "training_file": "모델 학습에 사용된 훈련 파일의 ID이다.",
  "hyperparameters": {
    "n_epochs": "전체 훈련 데이터셋을 몇 번 반복하여 학습할지를 나타내는 에포크(epoch) 수이다.",
    "batch_size": "한 번의 학습 단계에서 사용할 데이터 샘플의 개수이다.",
    "learning_rate_multiplier": "기본 학습률(learning rate)에 적용되는 배수 값으로, 학습 속도를 조절한다."
  },
  "trained_tokens": "파인튜닝 과정에서 모델이 학습한 총 토큰의 수이다.",
  "integrations": "W&B(Weights & Biases)와 같은 외부 로깅 서비스와의 연동 정보를 담는 배열이다.",
  "seed": "학습 결과의 재현성을 보장하기 위해 사용된 난수 시드(seed) 값이다.",
  "estimated_finish": "작업이 진행 중일 때 예상 완료 시간을 나타내는 유닉스 타임스탬프이다.",
  "method": {
    "type": "적용된 파인튜닝의 방법을 명시한다. 여기서는 'supervised'(지도 학습) 방식이다.",
    "supervised": {
      "hyperparameters": {
        "n_epochs": "지도 학습 방법에 사용된 에포크 수이다.",
        "batch_size": "지도 학습 방법에 사용된 배치 크기이다.",
        "learning_rate_multiplier": "지도 학습 방법에 사용된 학습률 배수이다."
      }
    }
  },
  "metadata": {
    "key": "사용자가 정의한 메타데이터의 값(value)이다. 'key'는 사용자가 지정한 키(key)를 나타낸다."
  }
}
```

<br>

## 파인튜닝 이벤트 객체 형식

```python
{
  "object": "이 객체의 유형이다. 'fine_tuning.job.event'는 파인튜닝 작업과 관련된 특정 사건(이벤트)을 의미한다.",
  "id": "이벤트의 고유 식별자(ID)이다.",
  "created_at": "이벤트가 발생한 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "level": "이벤트의 심각도 수준이다. 'info'는 정보성 메시지를 의미한다.",
  "message": "이벤트에 대한 사람이 읽을 수 있는 설명이다. 여기서는 파인튜닝 작업이 생성되었음을 알린다.",
  "data": "이벤트와 관련된 추가적인 구조화된 데이터를 담는 객체이다. 비어있는 객체 '{}'는 추가 데이터가 없음을 의미한다.",
  "type": "이벤트의 종류를 나타낸다. 'message'는 일반적인 로그 메시지를 의미한다."
}
```

<br>

## 파인튜닝 작업 체크포인트 객체 형식

```python
{
  "object": "이 객체의 유형이다. 'fine_tuning.job.checkpoint'는 파인튜닝 작업 중 특정 시점에 저장된 체크포인트(중간 저장 상태)를 의미한다.",
  "id": "체크포인트의 고유 식별자(ID)이다.",
  "created_at": "체크포인트가 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "fine_tuned_model_checkpoint": "이 체크포인트에 해당하는 파인튜닝된 모델의 이름이다. 특정 스텝(step)의 모델 상태를 나타낸다.",
  "fine_tuning_job_id": "이 체크포인트가 속한 상위 파인튜닝 작업의 ID이다.",
  "metrics": {
    "step": "이 지표(metric)가 측정된 학습 스텝 번호이다.",
    "train_loss": "해당 스텝에서 훈련 데이터셋(training set)에 대해 계산된 손실(loss) 값이다. 값이 낮을수록 모델이 훈련 데이터를 잘 학습하고 있음을 의미한다.",
    "train_mean_token_accuracy": "훈련 데이터셋에 대한 평균 토큰 정확도이다. 모델이 다음 토큰을 얼마나 정확하게 예측했는지를 나타낸다.",
    "valid_loss": "해당 스텝에서 검증 데이터셋(validation set)에 대해 계산된 손실 값이다. 모델의 일반화 성능을 평가하는 데 사용된다.",
    "valid_mean_token_accuracy": "검증 데이터셋에 대한 평균 토큰 정확도이다.",
    "full_valid_loss": "하나의 에포크(epoch)가 끝났을 때 전체 검증 데이터셋에 대해 계산된 최종 손실 값이다.",
    "full_valid_mean_token_accuracy": "하나의 에포크가 끝났을 때 전체 검증 데이터셋에 대한 최종 평균 토큰 정확도이다."
  },
  "step_number": "이 체크포인트가 저장된 학습 스텝(step)의 번호이다."
}
```
