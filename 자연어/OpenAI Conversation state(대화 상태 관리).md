[OpenAI Conversation state Docs](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses)

---

### 대화 상태 (Conversation state)

모델 상호작용 중 대화 상태를 관리하는 방법을 알아보세요.

OpenAI는 **대화 상태를 관리**하는 몇 가지 방법을 제공하며, 이는 **대화의 여러 메시지나 턴(turn)에 걸쳐 정보를 보존하는 데 중요**합니다.

### 수동으로 대화 상태 관리하기

각 텍스트 생성 요청은 독립적이고 상태를 저장하지 않지만(stateless) (`Assistants API`를 사용하지 않는 경우), 텍스트 생성 요청에 추가 메시지를 매개변수로 제공하여 다중 턴 대화를 구현할 수 있습니다.

<br>

**과거 대화 수동으로 구성하기**

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "user", "content": "knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
)
print(response.output_text)
```

`user`와 `assistant` 메시지를 번갈아 사용함으로써, 한 번의 요청으로 이전 대화 상태를 모델에 전달할 수 있습니다.

수동으로 생성된 응답 간에 컨텍스트를 공유하려면, 모델의 이전 응답 출력을 입력으로 포함하고, 그 입력을 다음 요청에 추가하세요.

다음 예시에서는 모델에게 농담을 해달라고 요청한 후, 또 다른 농담을 요청합니다. 이런 방식으로 이전 응답을 새 요청에 추가하면 대화가 자연스럽게 느껴지고 이전 상호작용의 컨텍스트를 유지하는 데 도움이 됩니다.

<br>

**Responses API로 수동으로 대화 상태 관리하기**

```python
from openai import OpenAI
client = OpenAI()

history = [
    {
        "role": "user",
        "content": "tell me a joke"
    }
]

response = client.responses.create(
    model="gpt-4o-mini",
    input=history,
    store=False
)
print(response.output_text)

# 대화에 응답 추가하기
history += [{"role": el.role, "content": el.content} for el in response.output]
history.append({ "role": "user", "content": "tell me another" })

second_response = client.responses.create(
    model="gpt-4o-mini",
    input=history,
    store=False
)
print(second_response.output_text)
```

<br>

### 대화 상태를 위한 OpenAI API

저희 **API를 사용하면 대화 상태를 자동으로 더 쉽게 관리**할 수 있으므로, 대화의 매 턴마다 수동으로 입력을 전달할 필요가 없습니다.

`previous_response_id` 매개변수를 사용하여 생성된 응답 간에 컨텍스트를 공유하세요. 이 매개변수를 사용하면 응답을 연결하여 스레드 형식의 대화를 만들 수 있습니다.

다음 예시에서는 모델에게 농담을 해달라고 요청합니다. 별도로, 그 농담이 왜 웃긴지 설명해달라고 요청하면, 모델은 좋은 응답을 제공하는 데 필요한 모든 컨텍스트를 갖게 됩니다.

**Responses API로 수동으로 대화 상태 관리하기**

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="tell me a joke",
)
print(response.output_text)

second_response = client.responses.create(
    model="gpt-4o-mini",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "explain why this is funny."}],
)
print(second_response.output_text)
```

실제 워크플로우

```python
행동 발생: 사용자가 질문 입력 후 엔터
if response.id가 존재하지 않으면
	 response = 모델 생성
 	 response.output_text는 사용자에게 보여줌.
	 response.id는 변수에 저장
else: response.id가 존재한다면
	 모델 생성시 previous_respose_id=reponse.id 파라미터 할당
```

response.id를 웹 사이트에서 나가기전까지 저장해야함.

- 로컬 스터리지 저장
- 세션 스토리지 저장

<br>

그냥 코드상 구현한다면, 그냥 글로벌 변수에 저장

<br>

### 모델 응답에 대한 데이터 보존

`previous_response_id`를 사용하는 경우에도, 체인에 있는 **모든 이전 응답의 입력 토큰은 API에서 입력 토큰으로 청구**됩니다.

<br>

### 컨텍스트 윈도우 관리하기

컨텍스트 윈도우을 이해하면 스레드 형식의 대화를 성공적으로 만들고 모델 상호작용 전반에 걸쳐 상태를 관리하는 데 도움이 됩니다.

- 컨텍스트 윈도우(context window)은 **단일 요청에서 사용할 수 있는 최대 토큰 수**입니다. 이 최대 토큰 수에는 **입력, 출력, 그리고 추론 토큰이 포함**됩니다. 사용하시는 모델의 컨텍스트 윈도우을 알아보려면 **모델 세부 정보**를 참조하세요.

<br>

### 텍스트 생성을 위한 컨텍스트 관리

입력이 더 복잡해지거나 대화에 더 많은 턴을 포함하게 되면, **출력 토큰**과 **컨텍스트 윈도우** 제한을 모두 고려해야 합니다. 모델의 입력과 출력은 **토큰** 단위로 측정됩니다. 토큰은 입력으로부터 파싱되어 내용과 의도를 분석하고, 논리적인 출력을 렌더링하기 위해 조합됩니다. 모델들은 **텍스트 생성 요청의 생명주기 동안 토큰 사용량에 제한**이 있습니다.

**출력 토큰**은 프롬프트에 대한 응답으로 모델이 생성하는 토큰입니다. 각 모델은 **출력 토큰에 대한 제한**이 다릅니다. 예를 들어, `gpt-4o-2024-08-06`은 최대 16,384개의 출력 토큰을 생성할 수 있습니다.

**컨텍스트 윈도우**은 입력 토큰과 출력 토큰(그리고 일부 모델의 경우 **추론 토큰**) 모두에 사용할 수 있는 총 토큰을 설명합니다. 저희 모델들의 **컨텍스트 윈도우 제한**을 비교해 보세요. 예를 들어, `gpt-4o-2024-08-06`은 총 128k 토큰의 컨텍스트 윈도우을 가집니다.

매우 큰 프롬프트를 생성하는 경우—종종 모델을 위해 추가적인 컨텍스트, 데이터, 또는 예시를 포함함으로써—모델에 **할당된 컨텍스트 윈도우을 초과할 위험**이 있으며, 이는 출력이 잘리는 결과로 이어질 수 있습니다.

**`tiktoken` 라이브러리**로 구축된 토크나이저 도구(tokenizer tool)를 사용하여 **특정 텍스트 문자열에 얼마나 많은 토큰이 있는지 확인**하세요.

<br>

예를 들어, `o1` 모델과 같이 추론이 활성화된 모델을 사용하여 `Responses API`에 API 요청을 할 때, 다음 토큰 수가 컨텍스트 윈도우 합계에 적용됩니다:

- **입력 토큰** (`Responses API`의 `input` 배열에 포함하는 입력)
- **출력 토큰** (프롬프트에 대한 응답으로 생성된 토큰)
- **추론 토큰** (모델이 응답을 계획하는 데 사용)

컨텍스트 윈도우 제한을 초과하여 생성된 토큰은 API 응답에서 잘릴 수 있습니다.

![image.png](<../images/자연어/OpenAI%20Conversation%20state(대화%20상태%20관리)/1.png>)

**토크나이저 도구**를 사용하여 메시지가 사용할 토큰 수를 추정할 수 있습니다.

### 다음 단계

더 구체적인 예시와 사용 사례는 **OpenAI Cookbook**을 방문하거나, API를 사용하여 모델 기능을 확장하는 방법에 대해 더 알아보세요:

- `Structured Outputs`으로 JSON 응답 받기
- 함수 호출로 모델 확장하기
- 실시간 응답을 위한 스트리밍 활성화하기
- 에이전트를 사용하여 컴퓨터 빌드하기

<br>
