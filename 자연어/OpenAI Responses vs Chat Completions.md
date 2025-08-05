**참고: [Responses vs. Chat Completions](https://platform.openai.com/docs/guides/responses-vs-chat-completions?api-mode=responses)**

---

### Responses API vs. Chat Completions API

**Responses API**와 **Chat Completions API**는 OpenAI 모델과 상호작용하는 두 가지 다른 방법이다.

<br>

### Responses API를 사용해야 하는 이유

Responses API는 **OpenAI의 최신 핵심 API**이자 **에이전트 중심의 API 프리미티브**이다.

이 API는 **Chat Completions의 단순함과 에이전트 작업 수행 능력을 결합한 방식**이다.

모델 기능이 발전함에 따라 Responses API는 다음과 같은 **내장 도구**를 통해 액션 지향적인 애플리케이션을 구축하기 위한 유연한 기반을 제공한다.

- 웹 검색
- 파일 검색
- 컴퓨터 사용

**새로운 사용자에게는 Responses API 사용을 권장**한다.

### 기능 비교

| 기능            | Chat Completions API | Responses API |
| --------------- | -------------------- | ------------- |
| 텍스트 생성     | 지원                 | 지원          |
| 오디오          |                      | 예정          |
| 비전            | 지원                 | 지원          |
| 구조화된 출력   | 지원                 | 지원          |
| 함수 호출       | 지원                 | 지원          |
| 웹 검색         |                      | 지원          |
| 파일 검색       |                      | 지원          |
| 컴퓨터 사용     |                      | 지원          |
| 코드 인터프리터 |                      | 지원          |
| MCP             | 지원                 | 지원          |

### Chat Completions API는 사라지지 않는다

**Chat Completions API는 AI 애플리케이션 구축을 위한 업계 표준이며**, OpenAI는 이 API를 **무기한으로 계속 지원할 계획**이다. Responses API는 도구 사용, 코드 실행, 상태 관리와 관련된 워크플로우를 단순화하기 위해 도입되었다.3 OpenAI는 이 새로운 API 프리미티브가 미래에 OpenAI 플랫폼을 더욱 효과적으로 강화할 수 있게 할 것이라 믿는다.

<br>

### 상태 저장 API와 의미론적 이벤트

**Responses API를 사용하면 이벤트 처리가 더 간단**해진다. Responses API는 **예측 가능한 이벤트 중심 아키텍처**를 가지고 있지만, Chat Completions API는 토큰이 생성될 때마다 `content` 필드에 내용을 계속 추가하므로 각 상태 간의 차이점을 수동으로 추적해야 한다. **Responses API를 사용하면 다단계 대화 논리와 추론을 더 쉽게 구현**할 수 있다.

Responses API는 **무엇이 변경되었는지 정확하게 설명하는 의미론적 이벤트를 명확**하게 내보낸다(예: 특정 텍스트 추가).

이를 통해 개발자는 특정 이벤트(예: 텍스트 변경)를 대상으로 통합 코드를 작성하여 통합을 단순화하고 타입 안정성을 향상시킬 수 있다.

<br>

### 각 API의 모델 가용성

가능한 한 모든 새로운 모델은 **Chat Completions API와 Responses API 모두에 추가**될 것이다. 일부 모델은 내장 도구(예: 컴퓨터 사용 모델)를 사용하거나, 내부적으로 여러 모델 생성 턴을 트리거하는 경우(예: o1-pro) Responses API를 통해서만 사용할 수 있다. 각 **모델의 상세 페이지에는 해당 모델이 Chat Completions, Responses 또는 둘 다를 지원하는지 표시**된다.

<br>

### 코드 비교

다음 예시는 Chat Completions API와 Responses API에 대한 기본적인 API 호출 방법을 보여준다.

### 텍스트 생성 예시

두 API 모두 모델로부터 출력을 쉽게 생성할 수 있다.

- `completion`은 `messages` 배열이 필요
- `response`는 `input` (문자열 또는 배열)이 필요하다.

<br>

**Chat Completions API 예시 코드**

```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
  model="gpt-4.1",
  messages=[
      {
          "role": "user",
          "content": "Write a one-sentence bedtime story about a unicorn."
      }
  ])
print(completion.choices[0].message.content)
```

<br>

**Responses API 예시 코드**

```python
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
  model="gpt-4.1",
  input=[
      {
          "role": "user",
          "content": "Write a one-sentence bedtime story about a unicorn."
      }
  ])
print(response.output_text)
```

<br>

### 응답 필드 차이점

Responses API

- `message` 대신 **`id`를 가진 유형화된 `response` 객체**를 받는다.
- Responses는 기본적으로 저장된다.

<br>

Chat Completions

- 새 계정의 경우 기본적으로 저장된다.

<br>

두 API 중 하나를 사용할 때 저장을 비활성화하려면 `store: false`로 설정하면 된다.

<br>

**Chat Completions API 응답 예시**

```python
[{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
    "refusal": null
  },
  "logprobs": null,
  "finish_reason": "stop"
}]
```

<br>

**Responses API 응답 예시**

```python
[{
  "id": "msg_67b73f697ba4819183a15cc17d011509",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "output_text",
      "text": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
      "annotations": []
    }
  ]
}]
```

<br>

### 기존 API에 미치는 영향

### Chat Completions

Chat Completions API는 OpenAI의 가장 널리 사용되는 API로 남아 있다. **OpenAI는 새로운 모델과 기능을 계속해서 지원**할 것이다. 애플리케이션에 내장 도구가 필요하지 않다면, 안심하고 Chat Completions를 계속 사용하면 된다.

내장 도구나 여러 모델 호출에 기능이 의존하지 않는 한, 새로운 모델은 Chat Completions에 계속 출시될 것이다. **에이전트 워크플로우를 위해 특별히 설계된 고급 기능이 필요하다면, Responses API를 권장**한다.

<br>

### Assistants

Assistants API 베타의 개발자 피드백을 바탕으로, Responses API에 주요 개선 사항을 통합하여 더 유연하고 빠르며 사용하기 쉽게 만들었다. **Responses API는 OpenAI에서 에이전트를 구축하는 미래 방향을 제시**한다.

OpenAI는 **Assistants API와 Responses API 간의 완전한 기능 동등성을 달성하기 위해 노력**하고 있으며, 여기에는 Assistant와 Thread와 같은 객체 및 코드 인터프리터 도구에 대한 지원이 포함된다. 완료되면 2026년 상반기를 목표로 Assistants API의 공식적인 지원 중단을 발표할 계획이다.

지원 중단 시, 개발자들이 모든 데이터를 보존하고 애플리케이션을 마이그레이션할 수 있도록 **Assistants API에서 Responses API로의 명확한 마이그레이션 가이드를 제공**할 것이다. 공식적인 지원 중단 발표 전까지는 Assistants API에 새로운 모델을 계속 제공할 것이다.
