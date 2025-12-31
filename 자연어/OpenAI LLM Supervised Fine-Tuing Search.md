# OpenAI LLM Supervised Fine-Tuing

# SFT(Supervised Fine-Tuning, 지도 파인 튜닝)

<mark>**OpenAI SFT 워크플로우**</mark>

1. OpenAI API는 `gpt-4o-mini`와 같은 특정 모델에 대한 파인튜닝 기능을 제공한다.
2. 원하는 작업에 맞춰 질문과 답변으로 구성된 학습 <mark>**데이터를 JSONL 형식으로 준비**</mark>해야 한다.
3. 준비한 <mark>**데이터 파일을 API를 통해 OpenAI에 업로드**</mark>하고 <mark>**파인튜닝 작업을 생성하여 학습**</mark>을 시작한다.
4. 학습이 완료되면 <mark>**고유한 모델 ID가 생성**</mark>되며, 이 <mark>**ID를 API 요청 시 지정하여 맞춤 모델을 사용**</mark>할 수 있다.
5. 파인튜닝된 모델의 <mark>**가중치는 OpenAI 서버에 저장**</mark>되며, 사용자가 직접 다운로드할 수는 없다.

<br>

<mark>**SFT란?**</mark>

SFT(Supervised Fine-Tuning)은 <mark>**사전 훈련된 언어 모델을 특정 작업이나 도메인에 맞게 조정**</mark>하는 것이다.

<mark>**Supervise learning(지도 학습) 방식을 기반**</mark>으로 한다.

<br>

더 나은 결과와 효율성을 위해 예시 입력과 알려진 정답 출력으로 모델을 파인 튜닝한다.

지도 파인 튜닝(Supervised fine-tuning, SFT)을 사용하면 특정 도메인에 대한 정보로 OpenAI 모델을 훈련시킬 수 있다.

그 결과 <mark>**모델이 특정 작업에 더 전문성있고 안정적으로 생성하는 맞춤형 모델**</mark>이 된다.

<br>

### Overview

지도 파인 튜닝은 네 가지 주요 부분으로 구성된다.

1. "좋은" 결과가 무엇인지 결정하기 위해 <mark>**훈련 데이터셋(training dataset)을 구축**</mark>한다.
2. 데이터(입력 + 프롬프트)와 그에 맞는 정답(레이블)을 한 쌍으로 묶어 <mark>**OpenAI에 업로드**</mark>한다.
3. 훈련 데이터를 사용하여 기본 모델(base model)에 대한 <mark>**파인 튜닝 작업(fine-tuning job)**</mark>을 생성한다.
4. 파인 튜닝된 모델을 사용하여 <mark>**결과를 평가**</mark>한다.

<br>

프롬프트 + 질문(query) 예시

```python
[지시문]
아래의 "검색된 컨텍스트"만을 사용하여 사용자의 질문에 답하세요.
만약 컨텍스트에 답변의 근거가 없다면, "정보를 찾을 수 없습니다"라고 답하세요.

[검색된 컨텍스트]
- 문서 1: 연차 휴가는 회계 연도 기준으로 모든 정규직 직원에게 연 15일 부여됩니다. 입사 첫해에는 월 1일의 연차가 발생합니다.
- 문서 2: 병가는 개인의 질병이나 부상 시 사용할 수 있으며, 연간 최대 60일까지 사용 가능합니다. 병가 사용 시 진단서 제출이 필요할 수 있습니다.
- 문서 3: 출장 시 발생하는 교통비는 실비 정산을 원칙으로 하며, 법인카드를 사용해야 합니다.

[사용자 질문]
신입사원인데, 병가를 얼마나 쓸 수 있나요?
```

<br>

정답(Label)

```python
연간 최대 60일까지 병가를 사용할 수 있으며, 필요시 진단서를 제출해야 할 수 있습니다 [문서 2].
```

<br>

데이터와 정답을 한쌍으로 JSON 형식에 맞게 업로드 해주어야한다.

```python
{
  "messages": [
    {
      "role": "user",
      "content": "[지시문]\n아래의 \"검색된 컨텍스트\"만을 사용하여 사용자의 질문에 답하세요.\n만약 컨텍스트에 답변의 근거가 없다면, \"정보를 찾을 수 없습니다\"라고 답하세요.\n\n[검색된 컨텍스트]\n- 문서 1: 연차 휴가는 회계 연도 기준으로 모든 정규직 직원에게 연 15일 부여됩니다. 입사 첫해에는 월 1일의 연차가 발생합니다.\n- 문서 2: 병가는 개인의 질병이나 부상 시 사용할 수 있으며, 연간 최대 60일까지 사용 가능합니다. 병가 사용 시 진단서 제출이 필요할 수 있습니다.\n- 문서 3: 출장 시 발생하는 교통비는 실비 정산을 원칙으로 하며, 법인카드를 사용해야 합니다.\n\n[사용자 질문]\n신입사원인데, 병가를 얼마나 쓸 수 있나요?"
    },
    {
      "role": "assistant",
      "content": "연간 최대 60일까지 병가를 사용할 수 있으며, 필요시 진단서를 제출해야 할 수 있습니다 [문서 2]."
    }
  ]
}
```

<br>

<mark>**평가 지표를 정해야한다!**</mark>

평가를 설정한 후에만 파인 튜닝에 투자해야 한다.

파인 튜닝된 모델이 기본 모델보다 더 나은 성능을 보이는지 판단할 신뢰할 수 있는 방법이 필요하다.

<br>

## 1. 데이터셋 구축(Build your dataset)

---

파인 튜닝된 모델에서 좋은 결과를 얻으려면 견고하고 대표적인 데이터셋을 구축해야 한다.

<br>

### 적절한 예제 수

- 파인 튜닝을 위해 제공할 수 있는 최소 예제 수는 10개이다.
- 50–100개의 예제로 파인 튜닝 시 성능 향상을 볼 수 있지만, 적절한 수는 사용 사례에 따라 크게 달라진다.
- <mark>**50개의 잘 만들어진 데이터로 결과를 평가하는 것을 권장**</mark>한다.

<br>

50개의 좋은 예제로 성능이 향상되면, 예제를 추가하여 결과가 더 나아지는지 확인해 본다.

만약 <mark>**50개의 예제가 효과가 없다면**</mark>, 훈련 데이터를 추가하기 전에 <mark>**작업이나 프롬프트를 다시 생각**</mark>해 본다.

<br>

### 좋은 예제의 조건

- 애플리케이션에서 예상되는 프롬프트와 출력이 무엇이든, <mark>**실제 사용자가 쓸법한 현실적인 질문과 최종적으로 원하는 이상적인 답변으로 구성**</mark>해야한다.(오타 포함)
  - 좋은 샘플
    ```
    프롬프트 (Prompt): "안녕하세요~ 재품 환불받고싶은대요 어떻게하나요??" (오타 포함)
    정답 (Label): "안녕하세요, 고객님. 제품 환불 절차에 대해 안내해 드리겠습니다. 먼저 주문번호를 알려주시겠어요?"
    ```
  - 나쁜 샘플
    ```
    프롬프트 (Prompt): "환불 절차"
    정답 (Label): "주문번호 필요"
    ```
- <mark>**구체적이고 명확한 질문과 답변**</mark>이어야 한다.
- 과거 데이터, 전문가 데이터, 기록된 데이터 또는 [수집된 다른 유형의 데이터](https://www.google.com/search?q=/docs/guides/evals)를 사용한다.

<br>

### 데이터 형식 지정

- JSONL 형식을 사용하며, 훈련 데이터 파일의 모든 줄에 완전한 JSON 구조 하나가 포함되어야 한다.
- [채팅 완료 형식(chat completions format)](https://platform.openai.com/docs/api-reference/fine-tuning/chat-input)을 사용한다.
- 파일은 최소 10줄 이상이어야 한다.

<br>

### JSONL 형식 예제 파일

모델이 `get_weather` 함수를 호출하는 JSONL 훈련 데이터의 예시이다.

```json
{"messages":[{"role":"user","content":"What is the weather in San Francisco?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"San Francisco, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. San Francisco, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Minneapolis?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Minneapolis, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Minneapolis, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in San Diego?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"San Diego, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. San Diego, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Memphis?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Memphis, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Memphis, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Atlanta?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Atlanta, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Atlanta, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Sunnyvale?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Sunnyvale, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Sunnyvale, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Chicago?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Chicago, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Chicago, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Boston?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Boston, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in Honolulu?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Honolulu, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. Honolulu, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
{"messages":[{"role":"user","content":"What is the weather in San Antonio?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"San Antonio, USA\", \"format\": \"celsius\"}"}}]}],"parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. San Antonio, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
```

<br>

### 더 큰 모델로부터 증류(Distilling from a larger model)

고성능 큰 모델을 Teacher Model(ex: `gpt-4.1`)이라고 부른다.

1. Teacher Model을 사용해서 고품질의 <mark>**질의응답(QA, Question-Answering)을 쌍을 생성**</mark>한다.
2. 최상의 데이터셋을 구축하기 위해, <mark>**평가 지표를 사용해서 높은 점수**</mark>가 나올때까지 <mark>**프롬프트 엔지니어링**</mark>을 한다.
3. 이렇게 <mark>**생성된 데이터셋**</mark>이 SFT에 사용되서 우리가 <mark>**사용할 작은 모델(ex: `gpt-4.1-mini`)을 훈련**</mark>시킨다.
4. 특정 작업에 한해서, <mark>**작은 모델이 큰 모델과 유사한 성능**</mark>을 낼 수 있게된다.

<br>

## 2. 훈련 데이터 업로드(Upload training data)

[OpenAI Upload File Docs](https://platform.openai.com/docs/api-reference/files/upload)

---

훈련 데이터셋을 <mark>**OpenAI에 업로드**</mark>한다.

SFT을 통해서 훈련 데이터셋을 학습했기 때문에, 특정 작업에 대해서 응답의 퀄리티가 높아진다.

모델이 항상 텍스트만 응답하는것이 아니라, JSON이나 함수 코드를 응답하도록 훈련 시킬 수 있다.

- <mark>**JSON 예시**</mark>: "서울 날씨 정보 알려줘"라는 질문에 대해 모델이 `{ "city": "서울", "temperature": "25도", "condition": "맑음" }`을 응답
- <mark>**함수코드 예시**</mark>: "내일 뉴욕 날씨 알려줘"라는 질문에 대해 모델이 직접 날씨 정보를 아는 대신, `call_weather_api(location="뉴욕", date="내일")`을 응답

<br>

훈련 데이터는 두가지 방법으로 업로드 할 수 있다.

1. OpenAI 웹사이트에 접속해 업로드
2. API를 사용해 업로드

### 1. OpenAI 웹사이트 GUI를 통해 데이터 업로드

1. [OpenAI developer platform](https://platform.openai.com/docs/overview)에 접속한다.
2. 대시보드 > [fine-tuning](https://platform.openai.com/finetune)으로 이동한다.
3. `+ Create`을 클릭한다.
4. <mark>**Training data**</mark> 아래에서 <mark>**JSONL**</mark> 파일을 업로드한다.

<br>

### 2. API를 호출하여 데이터 업로드

훈련 데이터가 `mydata.jsonl`이라는 파일에 저장되어 있다고 가정하고, 아래 코드를 사용하여 OpenAI에 업로드할 수 있다.

업로드된 파일의 `purpose`가 `fine-tune`으로 설정되어 파인튜닝을 목적으로 설정한다.

```bash
curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F purpose="fine-tune" \
  -F file="@mydata.jsonl"
```

- <mark>**`file`: 업로드할 파일 객체**</mark>(File object) 자체이다. 파일 이름이 아니라 실제 파일 데이터를 의미한다.
  - `curl` 명령어에서 `-F` 옵션과 함께 사용되는 `@`는 `"mydata.jsonl"`이라는 이름의 문자열 자체를 보내는 것이 아니라, `mydata.jsonl`이라는 파일이 위치한 경로를 찾아서 데이터를 body에 담아 전송한다.
  - 경로는 현재 위치를 기준으로 파일을 찾는다. 만약 다른 디렉토리에 존재한다면 경로를 적어주면 된다. `file="@/home/user/data/mydata.jsonl”`
- <mark>**`purpose`**</mark>: 업로드된 <mark>**파일의 사용 목적**</mark>(Intended purpose)을 나타내는 문자열이다. 다음 중 하나를 선택해야 한다.
  - `assistants`: Assistants API에 사용된다.
  - `batch`: Batch API에 사용된다.
  - `fine-tune`: 모델 파인 튜닝(fine-tuning)에 사용된다.
  - `vision`: 비전 파인 튜닝(vision fine-tuning)을 위한 이미지 파일에 사용된다.
  - `user_data`: 어떠한 목적에든 유연하게 사용할 수 있는 파일 유형이다.
  - `evals`: 평가 데이터를 업로드할때 사용한다.

<br>

API에서 반환된 데이터에 업로드된 파일의 `id`를 기록해 둔다. 후속 API 요청에서 해당 파일 식별자가 필요하다.

```json
{
  "object": "file",
  "id": "file-RCnFCYRhFDcq1aHxiYkBHw",
  "purpose": "fine-tune",
  "filename": "mydata.jsonl",
  "bytes": 1058,
  "created_at": 1746484901,
  "expires_at": null,
  "status": "processed",
  "status_details": null
}
```

<br>

### Fine-Tuning 실행(Create a fine-tuning job)

[OpenAI Fine-Tuing Docs](https://platform.openai.com/docs/api-reference/fine-tuning)

---

평가 데이터와 훈련 데이터를 업로드 하여 기본 모델을 파인튜닝을 한다.

파인 튜닝하기 위해서 사전 준비물이 있다.

- <mark>**파인 튜닝에 사용할 기본 모델**</mark>: OpenAI 모델 ID 또는 이전에 파인 튜닝된 모델의 ID일 수 있다. 모델 문서에서 파인 튜닝을 지원하는 모델을 확인한다.
- <mark>**훈련 데이터셋 ID**</mark>: 업로드한 훈련 데이터셋 파일 ID이다.
- <mark>**파인 튜닝 방법**</mark>: 파인 튜닝 방법을 지정한다. Supervised Fine-Tuning 이 기본값이다.

<br>

파인 튜닝을 하는 두가지 방법이 있다.

1. OpenAI 웹 사이트 GUI
2. API 호출

<br>

### 1. OpenAI 웹 사이트 GUI

1. <mark>**+ Create 버튼을 클릭**</mark>해서 이전에 훈련 데이터, 평가 데이터를 업로드 하지 않았으면 해준다.
2. <mark>**Supervise Fine-Tuning 방법**</mark>으로 선택하고 <mark>**훈련하려는 모델을 선택**</mark>한다.
3. 준비가 되면 <mark>**Create을 클릭하여 작업을 시작**</mark>한다.

<br>

### 2. API를 호출

Supervised Fine-Tuing 작업을 API로 불러온다.

```bash
curl https://api.openai.com/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "training_file": "file-RCnFCYRhFDcq1aHxiYkBHw",
    "model": "gpt-4.1-nano-2025-04-14"
  }'
```

<br>

API는 진행 중인 파인 튜닝에 대한 정보로 응답한다.

훈련 데이터의 크기에 따라 훈련 과정은 몇 분 또는 몇 시간이 걸릴 수 있다.

<mark>**폴링해서 작업의 상태를 주기적으로 확인**</mark>할 수 있다.

파인 튜닝 작업이 완료되면, 파인 튜닝 모델을 사용할 준비가 된 것이다.

완료된 파인 튜닝 작업은 다음과 같은 데이터를 반환한다.

```json
{
  "object": "fine_tuning.job",
  "id": "ftjob-uL1VKpwx7maorHNbOiDwFIn6",
  "model": "gpt-4.1-nano-2025-04-14",
  "created_at": 1746484925,
  "finished_at": 1746485841,
  "fine_tuned_model": "ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH",
  "organization_id": "org-abc123",
  "result_files": ["file-9TLxKY2A8tC5YE1RULYxf6"],
  "status": "succeeded",
  "validation_file": null,
  "training_file": "file-RCnFCYRhFDcq1aHxiYkBHw",
  "hyperparameters": {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate_multiplier": 1
  },
  "trained_tokens": 1700,
  "error": {},
  "user_provided_suffix": null,
  "seed": 1935755117,
  "estimated_finish": null,
  "integrations": [],
  "metadata": null,
  "usage_metrics": null,
  "shared_with_openai": false,
  "method": {
    "type": "supervised",
    "supervised": {
      "hyperparameters": {
        "n_epochs": 10,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
      }
    }
  }
}
```

`fine_tuned_model` 속성을 주목한다.

이것이 <mark>**파인 튜닝된 모델을 사용할때 필요한 모델 ID**</mark>이다.

<br>

다음은 파인 튜닝된 모델 ID로 추론하기 위해 사용할 Responses API를 호출하는 예시이다.

```json
{
  "object": "fine_tuning.job",
  "id": "ftjob-uL1VKpwx7maorHNbOiDwFIn6",
  "model": "gpt-4.1-nano-2025-04-14",
  "created_at": 1746484925,
  "finished_at": 1746485841,
  "fine_tuned_model": "ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH",
  "organization_id": "org-abc123",
  "result_files": ["file-9TLxKY2A8tC5YE1RULYxf6"],
  "status": "succeeded",
  "validation_file": null,
  "training_file": "file-RCnFCYRhFDcq1aHxiYkBHw",
  "hyperparameters": {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate_multiplier": 1
  },
  "trained_tokens": 1700,
  "error": {},
  "user_provided_suffix": null,
  "seed": 1935755117,
  "estimated_finish": null,
  "integrations": [],
  "metadata": null,
  "usage_metrics": null,
  "shared_with_openai": false,
  "method": {
    "type": "supervised",
    "supervised": {
      "hyperparameters": {
        "n_epochs": 10,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
      }
    }
  }
}
```

<br>

## 3. Evaluate the result

---

아래 접근 방식을 사용하여 <mark>**파인 튜닝된 모델의 성능을 확인**</mark>한다.

<mark>**원하는 결과를 얻을 때까지 프롬프트, 데이터, 파인 튜닝 작업을 계속한다.**</mark>

파인 튜닝의 <mark>**가장 좋은 방법은 계속해서 반복**</mark>하는 것이다.

<br>

### 평가(evals)와 비교

파인 튜닝된 모델이 원래 기본 모델보다 성능이 더 좋은지 확인하려면, [평가를 사용](https://platform.openai.com/docs/guides/evals)한다.

파인 튜닝 작업을 실행하기 전에, 1단계에서 수집한 동일한 훈련 데이터셋에서 데이터를 분리해 둔다.

이 홀드아웃(holdout) 데이터는 평가에 사용할 때 통제 그룹 역할을 한다.

훈련 데이터와 홀드아웃 데이터가 사용자 입력 유형과 모델 응답의 다양성에서 거의 동일하도록 한다.

[평가 실행에 대해 더 알아보기](https://www.google.com/search?q=/docs/guides/evals).

<br>

### 상태 모니터링

대시보드 또는 API에서 작업 ID를 폴링하여 파인 튜닝 작업의 상태를 확인한다.

### UI에서 모니터링

1. [파인 튜닝 대시보드](https://platform.openai.com/finetune)로 이동한다.
2. 모니터링하려는 작업을 선택한다.
3. 상태, 체크포인트, 메시지, 메트릭을 검토한다.

<br>

### API 호출로 모니터링

이 curl 명령을 사용하여 파인 튜닝 작업에 대한 정보를 얻는다.

```bash
curl https://api.openai.com/v1/fine_tuning/jobs/[파인 튜닝된 모델 ID] \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

<br>

작업에는 `fine_tuned_model` 속성이 포함되어 있으며, 이것이 새로운 파인 튜닝 모델의 고유 ID이다.

```json
{
  "object": "fine_tuning.job",
  "id": "ftjob-uL1VKpwx7maorHNbOiDwFIn6",
  "model": "gpt-4.1-nano-2025-04-14",
  "created_at": 1746484925,
  "finished_at": 1746485841,
  "fine_tuned_model": "ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH",
  "organization_id": "org-abc123",
  "result_files": ["file-9TLxKY2A8tC5YE1RULYxf6"],
  "status": "succeeded",
  "validation_file": null,
  "training_file": "file-RCnFCYRhFDcq1aHxiYkBHw",
  "hyperparameters": {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate_multiplier": 1
  },
  "trained_tokens": 1700,
  "error": {},
  "user_provided_suffix": null,
  "seed": 1935755117,
  "estimated_finish": null,
  "integrations": [],
  "metadata": null,
  "usage_metrics": null,
  "shared_with_openai": false,
  "method": {
    "type": "supervised",
    "supervised": {
      "hyperparameters": {
        "n_epochs": 10,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
      }
    }
  }
}
```

### 파인 튜닝된 모델 사용해보기

새롭게 최적화된 모델을 직접 사용하여 평가한다.

파인 튜닝된 모델의 훈련이 끝나면, OpenAI 기본 모델처럼 Responses 또는 Chat Completions API에서 해당 ID를 사용한다.

<br>

두가지 방법으로 모델을 사용할 수 있다.

1. 웹 사이트 Playground GUI
2. API 호출

<br>

### 1. 플레이그라운드에서 모델 사용

1. 대시보드에서 파인 튜닝 작업으로 이동한다.
2. 오른쪽 창에서 출력 모델(Output model)로 이동하여 모델 ID를 복사한다. `ft:…`로 시작해야 한다.
3. [플레이그라운드](https://platform.openai.com/chat/edit)를 연다.
4. <mark>**모델(Model)**</mark> 드롭다운 메뉴에 모델 ID를 붙여넣는다. 여기서 이전에 생성한 다른 파인 튜닝 모델도 볼 수 있다.
5. 몇 가지 프롬프트를 실행하고 파인 튜닝된 모델의 성능을 확인한다.

<br>

### 2. API 호출로 모델 사용

```bash
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH",
    "input": "What is 4+4?"
  }'
```

### 필요한 경우 체크포인트(checkpoints) 사용

각 훈련 <mark>**에포크가 끝날 때**</mark>마다 전체 모델 <mark>**체크포인트를 생성**</mark>한다.

이는 파인 튜닝된 모델이 <mark>**초반에는 개선되다가 일반화 가능한 지식을 배우는 대신 데이터셋을 암기하는 경우**</mark>, 즉 <mark>**과적합(overfitting)이라 불리는 경우에 유용**</mark>하다.

<br>

체크포인트된 모델 찾는법도 2가지이다.

1. 웹 사이트 대시보드 GUI
2. API 호출

<br>

### 1. 대시보드에서 체크포인트 찾기

1. 파인 튜닝 대시보드로 이동한다.
2. 왼쪽 패널에서 조사하려는 작업을 선택한다. 성공할 때까지 기다린다.
3. 오른쪽 패널에서 체크포인트 목록으로 스크롤한다.
4. 체크포인트 위로 마우스를 가져가면 플레이그라운드에서 실행할 수 있는 링크가 표시된다.
5. 플레이그라운드에서 프롬프트를 입력하여 체크포인트 모델의 동작을 테스트한다.

<br>

### 2. API로 체크포인트 쿼리하기

1. 작업이 성공할 때까지 기다린다. 이는 [작업 상태를 쿼리](https://www.google.com/search?q=/docs/api-reference/fine-tuning/retrieve)하여 확인할 수 있다.
2. 파인 튜닝 작업 ID로 [체크포인트 엔드포인트를 쿼리](https://www.google.com/search?q=/docs/api-reference/fine-tuning/list-checkpoints)하여 해당 파인 튜닝 작업에 대한 모델 체크포인트 목록에 접근한다.
3. 모델 체크포인트의 이름을 `fine_tuned_model_checkpoint` 필드에서 찾는다.
4. 최종 파인 튜닝된 모델처럼 이 모델을 사용한다.

<br>

체크포인트 모델 리스트 불러온다.

```python
import requests
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
fine_tuning_job_id = "ftjob-abc123"

url = f"https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

response = requests.get(url, headers=headers)
checkpoints_data = response.json()

print(checkpoints_data)
```

체크포인트 객체는 이 모델의 유용성을 판단하는 데 도움이 되는 `metrics` 데이터를 포함한다. 예시로, 응답은 다음과 같다.

```json
{
  "object": "fine_tuning.job.checkpoint",
  "id": "ftckpt_zc4Q7MP6XxulcVzj4MZdwsAB",
  "created_at": 1519129973,
  "fine_tuned_model_checkpoint": "ft:gpt-3.5-turbo-0125:my-org:custom-suffix:96olL566:ckpt-step-2000",
  "metrics": {
    "full_valid_loss": 0.134,
    "full_valid_mean_token_accuracy": 0.874
  },
  "fine_tuning_job_id": "ftjob-abc123",
  "step_number": 2000
}
```

- `step_number`: 체크포인트가 생성된 스텝(여기서 각 에포크는 훈련 세트의 스텝 수를 배치 크기로 나눈 값)
- `metrics`: 체크포인트가 생성된 스텝에서의 파인 튜닝 작업에 대한 메트릭을 포함하는 객체

<br>

## 안전성 검사(Safety checks)

---

프로덕션에 출시하기 전에 다음 안전성 정보를 검토하고 따른다.

### 안전성 평가 방법

파인 튜닝 작업이 완료되면, 결과 모델의 동작을 13개의 개별 안전 범주에 걸쳐 평가한다. 각 범주는 적절히 제어되지 않을 경우 AI 출력이 잠재적으로 해를 끼칠 수 있는 중요한 영역을 나타낸다.

| 이름(Name)                          | 설명(Description)                                                                                                                                                                                  |     |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| advice(조언)                        | 우리의 정책을 위반하는 조언 또는 지침.                                                                                                                                                             |     |
| harassment/threatening(괴롭힘/위협) | 모든 대상에 대한 폭력이나 심각한 해를 포함하는 괴롭힘 콘텐츠.                                                                                                                                      |     |
| hate(증오)                          | 인종, 성별, 민족, 종교, 국적, 성적 지향, 장애 상태 또는 카스트에 기반한 증오를 표현, 선동 또는 조장하는 콘텐츠. 보호받지 않는 그룹(예: 체스 선수)을 대상으로 한 증오성 콘텐츠는 괴롭힘에 해당한다. |     |
| hate/threatening(증오/위협)         | 인종, 성별, 민족, 종교, 국적, 성적 지향, 장애 상태 또는 카스트를 기반으로 대상 그룹에 대한 폭력이나 심각한 해를 포함하는 증오성 콘텐츠.                                                            |     |
| highly-sensitive(고도로 민감한)     | 우리의 정책을 위반하는 고도로 민감한 데이터.                                                                                                                                                       |     |
| illicit(불법적인)                   | 불법 행위를 저지르는 방법에 대한 조언이나 지침을 제공하는 콘텐츠. "소매치기하는 법"과 같은 문구가 이 범주에 해당한다.                                                                              |     |
| propaganda(선전)                    | 우리의 정책을 위반하는 이데올로기에 대한 찬양 또는 지원.                                                                                                                                           |     |
| self-harm/instructions(자해/지침)   | 자살, 자해, 섭식 장애와 같은 자해 행위를 장려하거나, 그러한 행위를 저지르는 방법에 대한 지침이나 조언을 제공하는 콘텐츠.                                                                           |     |
| self-harm/intent(자해/의도)         | 화자가 자살, 자해, 섭식 장애와 같은 자해 행위를 하고 있거나 하려는 의도를 표현하는 콘텐츠.                                                                                                         |     |
| sensitive(민감한)                   | 우리의 정책을 위반하는 민감한 데이터.                                                                                                                                                              |     |
| sexual/minors(성적/미성년자)        | 18세 미만의 개인이 포함된 성적인 콘텐츠.                                                                                                                                                           |     |
| sexual(성적)                        | 성적 흥분을 유발하기 위한 콘텐츠, 예를 들어 성행위 묘사 또는 성적 서비스 홍보(성교육 및 건강 제외).                                                                                                |     |
| violence(폭력)                      | 죽음, 폭력 또는 신체적 상해를 묘사하는 콘텐츠.                                                                                                                                                     |     |

각 범주에는 사전 정의된 통과 임계값이 있다. 특정 범주에서 평가된 예제가 너무 많이 실패하면 OpenAI는 파인 튜닝된 모델의 배포를 차단한다. 파인 튜닝된 모델이 안전성 검사를 통과하지 못하면, OpenAI는 파인 튜닝 작업에서 어떤 범주가 요구되는 임계값을 충족하지 못했는지 설명하는 메시지를 보낸다. 결과는 파인 튜닝 작업의 검토(moderation checks) 섹션에서 볼 수 있다.

<br>

### 안전성 검사 통과 방법

파인 튜닝 작업 객체에서 실패한 안전성 검사를 검토하는 것 외에도, [파인 튜닝 API 이벤트 엔드포인트](https://platform.openai.com/docs/api-reference/fine-tuning/list-events)를 쿼리하여 어떤 범주가 실패했는지에 대한 세부 정보를 검색할 수 있다. 범주 결과 및 시행에 대한 자세한 내용은 `moderation_checks` 유형의 이벤트를 찾는다. 이 정보는 재훈련 및 개선을 위해 어떤 범주를 목표로 삼아야 할지 좁히는 데 도움이 될 수 있다. [모델 사양](https://cdn.openai.com/spec/model-spec-2024-05-08.html#overview)에는 추가 훈련 데이터 영역을 식별하는 데 도움이 될 수 있는 규칙과 예제가 있다.

이러한 평가는 광범위한 안전 범주를 다루지만, 사용 사례에 적합한지 확인하기 위해 파인 튜닝된 모델에 대한 자체 평가를 수행해야 한다.

<br>

참고

- [Fine-Tuning Techniques - Choosing Between SFT, DPO, and RFT (With a Guide to DPO)](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)
- [Fine-tuning best practices](https://platform.openai.com/docs/guides/fine-tuning-best-practices)
- [How to fine-tune chat models](https://cookbook.openai.com/examples/how_to_finetune_chat_models)
- [Open AI Supervised fine-tuning Docs](https://platform.openai.com/docs/guides/supervised-fine-tuning?checkpoints=api#page-top)
