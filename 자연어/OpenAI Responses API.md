참고: [OpenAI Responses API Docs](https://platform.openai.com/docs/api-reference/responses)

---

OpenAI의 <mark>**가장 진보된 모델 응답 생성 인터페이스**</mark>이다.

텍스트와 이미지 입력을 지원하며, 텍스트 출력을 생성한다.

이전 응답의 출력을 입력으로 사용하여 지속적으로 대화를 이어나갈 수 있다.

<mark>**파일 검색, 웹 검색, 컴퓨터 사용 등 내장 도구를 활용**</mark>하여 모델의 기능을 확장한다.

또한, <mark>**함수 호출을 통해 모델이 외부 시스템과 데이터에 접근**</mark>할 수 있도록 한다.

<br>

## Create a model response

Response 모델을 생성한다.

텍스트 또는 이미지를 입력으로 제공하여 텍스트 또는 JSON 출력을 생성할 수 있다.

모델이 직접 만든 코드를 호출하거나, 웹 검색 또는 파일 검색과 같은 내장 도구를 사용하여 자신의 데이터를 모델 응답의 입력으로 활용할 수 있다.

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
  model="gpt-4.1-nano",
  input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

<br>
아래는 `client.responses.create()` 함수에 사용 가능한 모든 파라미터를 적용한 예시 코드이다.

```python
# OpenAI 클라이언트를 초기화한다.
# API 키는 환경 변수 'OPENAI_API_KEY'에서 자동으로 로드된다.
client = OpenAI()

response = client.responses.create(
    # --- 기본 파라미터 ---
    model="응답을 생성하는 데 사용할 모델의 ID이다. 예: 'gpt-4o', 'o3'",
    input=[
        {"role": "text", #  메시지 인풋의 역활 (user, assistant, system, developer)
         "content": "모델에 전달할 텍스트, 이미지, 파일 입력이다.",
        },
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ],
    instructions="모델의 컨텍스트에 삽입될 시스템(개발자) 메시지이다.",

    # --- 대화 및 상태 관리 ---
    previous_response_id="이전 응답의 고유 ID이다. 여러 턴에 걸친 대화를 만들 때 사용한다.",
    store=True,  # boolean | 생성된 응답을 나중에 검색할 수 있도록 저장할지 여부이다. 기본값: true
    background=False,  # boolean | 모델 응답을 백그라운드에서 실행할지 여부이다. 기본값: false

    # --- 출력 제어 ---
    max_output_tokens=1024,  # integer | 생성될 수 있는 토큰의 최대 상한선이다.
    stream=False,  # boolean | true로 설정 시, 응답 데이터가 생성되는 대로 스트리밍된다. 기본값: false
    temperature=1.0,  # number | 샘플링 온도로, 0~2 사이 값이다. 높을수록 무작위성이 커진다. 기본값: 1
    top_p=1.0,  # number | temperature 대신 사용하는 핵 샘플링(nucleus sampling) 방식이다. 기본값: 1
    truncation="auto",  # string | 컨텍스트 창 초과 시 입력을 자르는 전략이다. 'auto' 또는 'disabled'. 기본값: 'disabled'

    # --- 도구(Tool) 사용 ---
    tools=[{"type": "web_search"}],  # array | 모델이 호출할 수 있는 도구의 배열이다. 웹 검색, 함수 호출 등이 있다.
    tool_choice="auto",  # string or object | 모델이 어떤 도구를 선택할지 제어한다. 'auto', 'none' 등.
    parallel_tool_calls=True,  # boolean | 모델이 도구 호출을 병렬로 실행하도록 허용할지 여부이다. 기본값: true
    max_tool_calls=10,  # integer | 응답 처리 중 내장 도구를 호출할 수 있는 최대 횟수이다.

    # --- 고급 및 추가 기능 ---
    include=["message.output_text.logprobs"],  # array | 응답에 추가로 포함할 데이터를 지정한다. 로그 확률, 코드 실행 결과 등이 있다.
    top_logprobs=5,  # integer | 각 토큰 위치에서 반환할 가장 가능성 높은 토큰의 개수(0~20)를 지정한다.
    metadata={"project_id": "prj_123", "user_tier": "premium"},  # map | 객체에 첨부할 수 있는 최대 16개의 키-값 쌍이다.
    prompt={"id": "prompt_abc", "variables": {"topic": "AI"}},  # object | 미리 정의된 프롬프트 템플릿과 그 변수들을 참조한다.
    reasoning={"effort": "high"},  # object | o-시리즈 모델에서 사용되며, 모델의 추론 기능에 대한 구성 옵션이다.
    text={"format": {"type": "json_object"}},  # object | 모델의 텍스트 응답 형식을 일반 텍스트 또는 구조화된 JSON으로 지정한다.

    # --- 서비스 및 정책 ---
    service_tier="auto",  # string | 요청을 처리할 서비스 등급이다. 'auto', 'default', 'flex' 등. 기본값: 'auto'
    safety_identifier="정책 위반 탐지를 위한 사용자의 안정적인 식별자이다. 사용자 이름이나 이메일을 해시하여 사용한다.",
    prompt_cache_key="유사한 요청에 대한 응답 속도 최적화를 위해 캐시 키로 사용된다.",

    # --- 사용 중단된 파라미터 ---
    user="[사용 중단됨] safety_identifier와 prompt_cache_key로 대체된 이전 사용자 식별자이다."
)

# 위 코드는 파라미터 설명을 위한 예시이며, 그대로 실행되지 않는다.
# 실제 응답을 확인하려면 아래 코드의 주석을 해제하고, 파라미터에 실제 값을 넣어야 한다.
# print(response)
```

`instructions` 매개변수는 모델이 응답을 생성하는 동안 어떻게 동작해야 하는지에 대한 높은 수준의 지침을 제공하며, 톤, 목표 및 올바른 응답의 예시를 포함한다. 이 방식으로 제공된 모든 지침은 `input` 매개변수의 프롬프트보다 우선한다.

지침으로 텍스트를 생성한다.

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    instructions="해적처럼 말해.",
    input="JavaScript에서 세미콜론은 선택 사항인가요?",
)

print(response.output_text)
```

위 예시는 `input` 배열에서 다음 입력 메시지를 사용하는 것과 거의 동일하다.

다른 역할을 사용하여 메시지로 텍스트를 생성한다.

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "developer",
            "content": "해적처럼 말해."
        },
        {
            "role": "user",
            "content": "JavaScript에서 세미콜론은 선택 사항인가요?"
        }
    ]
)

print(response.output_text)
```

<br>

### instructions

`instructions` 매개변수는 현재 응답 생성 요청에만 적용된다는 점에 유의한다.

`previous_response_id` 매개변수로 [대화 상태를 관리](https://www.google.com/search?q=/docs/guides/conversation-state)하는 경우, 이전 턴에 사용된 `instructions`는 컨텍스트에 존재하지 않는다.

<br>

### role

| <mark>**`developer`**</mark> | <mark>**`user`**</mark> | <mark>**`assistant`**</mark> |
| --------------- | ---------- | --------------- |

| 1. `developer` 메시지는 애플리케이션 개발자가 제공하는 지침이다.

2. 개발자 메시지는 사용자 메시지보다 우선순위를 가진다. 개발자가 애플리케이션 동작을 제어하기 위함이다.

3. 사용자 입력에 앞서 개발자의 의도를 명확히 전달하는 수단이다. 예를들어 특정 기능의 사용 제한이나 필수 입력 사항 등을 명시할 수 있다.

4. `developer` 메시지는 함수 정의와 같이 시스템의 규칙과 비즈니스 로직을 제공한다. | 1. 사용자 메시지는 최종 사용자가 제공하는 지침이다.

2.사용자 메시지는 개발자 메시지보다 우선순위가 낮다. 이는 사용자가 애플리케이션에 원하는 작업을 지시할 때 사용된다.

3. 개발자가 설정한 기본적인 틀 안에서 사용자의 상호작용을 가능하게 한다. 예를 들어, 검색어 입력, 특정 버튼 클릭 등이 사용자 메시지에 해당한다.

4.`developer` 메시지 지침이 적용되는 입력 및 구성을 제공한다. | 1. 모델이 생성하는 메시지는 <mark>**어시스턴트 역할**</mark>을 가진다. 이는 모델이 사용자 또는 개발자의 지시에 따라 <mark>**응답을 제공**</mark>함을 의미한다.

2. 어시스턴트 메시지는 <mark>**정보 제공, 질문 답변, 코드 생성**</mark> 등 다양한 형태를 띤다.

3. 시스템 대화에서 모델의 <mark>**출력 결과**</mark>를 나타내는 중요한 부분이다. 예를 들어, 사용자의 질문에 대한 답변이나 요청된 작업의 결과를 포함한다. |

<br>

### Prompt

OpenAI 대시보드에서 코드에 프롬프트의 내용을 지정하는 대신 API 요청에 사용할 수 있는 <mark>**재사용 가능한 [프롬프트](https://www.google.com/search?q=/chat/edit)를 개발**</mark>할 수 있다.

이렇게 하면 프롬프트를 더 쉽게 구축하고 평가할 수 있으며, <mark>**통합 코드를 변경하지 않고도 개선된 버전의 프롬프트를 배포**</mark>할 수 있다.

작동 방식은 다음과 같다.

1. 대시보드에서 `{{customer_name}}`과 같은 플레이스홀더로 <mark>**재사용 가능한 프롬프트를 생성한다**</mark>.
2. `prompt` 매개변수를 사용하여 <mark>**API 요청에서 프롬프트를 사용한다**</mark>. 프롬프트 매개변수 객체에는 구성할 수 있는 세 가지 속성이 있다.
   - `id` — 대시보드에서 찾을 수 있는 <mark>**프롬프트의 고유 ID**</mark>
   - `version` — 프롬프트의 특정 버전 (대시보드에 지정된 "현재" 버전으로 기본 설정됨)
   - `variables` — <mark>**프롬프트의 변수에 대체할 값의 맵**</mark>이다. 대체 값은 문자열이거나 `input_image` 또는 `input_file`과 같은 다른 응답 입력 메시지 유형일 수 있다.

<br>

프롬프트 템플릿으로 텍스트를 생성한다.

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box"
        }
    }
)

print(response.output_text)
```

<br>

파일 인풋이 있는 프롬프트 템플릿

```python
import openai, pathlib

client = openai.OpenAI()

# 변수에서 참조할 PDF를 업로드한다.
file = client.files.create(
    file=open("draconomicon.pdf", "rb"),
    purpose="user_data",
)

response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_abc123",
        "variables": {
            "topic": "Dragons",
            "reference_pdf": {
                "type": "input_file",
                "file_id": file.id,
            },
        },
    },
)

print(response.output_text)
```

<br>

## 마크다운 및 XML을 사용한 메시지 형식 지정

<br>

## Get a model response

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.retrieve("resp_123")
print(response)
```

<br>

## Response 객체

```json
{
  "id": "생성된 응답 객체 전체에 대한 고유 식별자(ID)이다.",
  "object": "이 객체의 유형이다. 'response'는 API의 응답 객체임을 나타낸다.",
  "created_at": "응답 객체가 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.",
  "status": "응답 생성 작업의 현재 상태이다. 'completed'는 성공적으로 완료되었음을 의미한다.",
  "error": "오류가 발생했을 경우, 오류에 대한 정보를 담는 객체이다. 'null'은 오류가 없음을 나타낸다.",
  "incomplete_details": "응답이 불완전할 경우(예: 토큰 제한 도달) 그 이유에 대한 세부 정보를 담는다.",
  "instructions": "응답 생성 시 사용된 시스템 지시사항(system prompt)이다.",
  "max_output_tokens": "생성될 수 있는 최대 출력 토큰의 수이다.",
  "model": "이 응답을 생성하는 데 사용된 AI 모델의 이름이다.",
  "output": [
    {
      "type": "출력의 유형이다. 'message'는 응답이 메시지 형식임을 나타낸다.",
      "id": "출력 메시지에 대한 고유 식별자이다.",
      "status": "이 메시지 부분의 생성 상태이다.",
      "role": "메시지 작성자의 역할이다. 'assistant'는 AI 모델을 의미한다.",
      "content": [
        {
          "type": "콘텐츠 블록의 유형이다. 'output_text'는 일반 텍스트 콘텐츠임을 나타낸다.",
          "text": "모델이 실제로 생성한 텍스트 내용이다.",
          "annotations": "텍스트에 대한 주석 정보가 담기는 배열이다. 여기서는 비어있다."
        }
      ]
    }
  ],
  "parallel_tool_calls": "도구(tool) 병렬 호출 기능이 활성화되었는지 여부를 나타낸다.",
  "previous_response_id": "대화의 연속성을 위해 이전 응답의 ID를 참조할 때 사용된다.",
  "reasoning": {
    "effort": "모델의 추론 과정에 대한 내부 평가 지표이다.",
    "summary": "모델의 추론 과정에 대한 요약이다."
  },
  "store": "이 응답을 저장할지 여부를 나타내는 불리언 값이다.",
  "temperature": "샘플링 온도를 나타낸다. 값이 높을수록(최대 2) 응답이 더 무작위적이고 창의적으로 변한다.",
  "text": {
    "format": {
      "type": "요청된 출력의 형식을 지정한다. 'text'는 일반 텍스트를 의미한다."
    }
  },
  "tool_choice": "모델의 도구 사용 방식을 제어한다. 'auto'는 모델이 자동으로 판단함을 의미한다.",
  "tools": "모델이 사용할 수 있는 도구의 목록을 담는 배열이다. 여기서는 비어있다.",
  "top_p": "핵심 샘플링(nucleus sampling) 값을 나타낸다. temperature와 함께 응답의 무작위성을 제어한다.",
  "truncation": "입력 텍스트가 길 경우 잘라내는 방식을 제어하는 전략이다.",
  "usage": {
    "input_tokens": "입력에 사용된 토큰의 수이다.",
    "input_tokens_details": {
      "cached_tokens": "캐시에서 재사용된 토큰의 수이다."
    },
    "output_tokens": "출력(응답) 생성에 사용된 토큰의 수이다.",
    "output_tokens_details": {
      "reasoning_tokens": "모델의 내부 추론 과정에 사용된 토큰의 수이다."
    },
    "total_tokens": "요청 처리 전체에 사용된 총 토큰의 수이다."
  },
  "user": "요청을 보낸 최종 사용자의 고유 식별자이다.",
  "metadata": "사용자가 정의한 키-값 형식의 메타데이터를 저장하는 공간이다."
}
```

<br>
