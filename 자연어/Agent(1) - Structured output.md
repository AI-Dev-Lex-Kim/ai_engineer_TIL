- [Agent(1) - Structured output (ProviderStrategy, ToolStrategy)](#agent1---structured-output-providerstrategy-toolstrategy)
  - [Response format](#response-format)
  - [Provider strategy](#provider-strategy)
    - [자동 ProviderStrategy 선택](#자동-providerstrategy-선택)
  - [Tool calling strategy](#tool-calling-strategy)
    - [`tool_message_content`](#tool_message_content)
  - [`handle_errors`](#handle_errors)
    - [Multiple structured outputs error](#multiple-structured-outputs-error)
    - [Schema validation error](#schema-validation-error)
    - [커스텀 에러 메시지](#커스텀-에러-메시지)
    - [특정 예외만 처리하기](#특정-예외만-처리하기)
    - [커스텀 에러 핸들러 함수](#커스텀-에러-핸들러-함수)
    - [에러 처리 비활성화](#에러-처리-비활성화)
    - [정리](#정리)

# Agent(1) - Structured output (ProviderStrategy, ToolStrategy)

Structured output은 에이전트가 <mark>**항상 일정하고 예측가능한 형식으로 데이터를 반환**</mark>하도록 만드는 기능이다.

LLM이 생성한 자연어 응답을 다시 파싱할 필요 없이, 애플리케이션이 바로 사용할 수 있는 <mark>**JSON 객체, Pydantic 모델, dataclass 형태의 구조화된 데이터**</mark>를 얻을 수 있다.

<br>

LangChain의 `create_agent`는 structured output을 따로 정하지 않으면 자동으로 처리한다.

만약 사용자가 원하는 출력 스키마를 지정하면, 모<mark>**델이 해당 스키마에 맞는 구조화된 데이터를 생성**</mark>한다.

이후 LangChain이 검증한뒤 에이전트 상태(state)의 `structured_response` 키에 담아 반환한다.

```python
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
```

<br>

## Response format

`create_agent`의 파라마터 중 `response_format`은 에이전트가 결과값을 생성할때, 정해진 형식으로 반환하게 결정하는 값이다.

`response_format` 에는 4가지 값을 전달할 수 있다.

- `ToolStrategy`
  <mark>**도구 호출(tool calling)을 사용**</mark>해 구조화된 출력을 생성한다.
- `ProviderStrategy`
  <mark>**모델 제공자(provider)가 지원**</mark>하는 <mark>**네이티브 structured output 기능을 사용**</mark>한다.
- `type`
  <mark>**스키마 타입**</mark>만 전달한다.
  LangChain이 모델의 기능을 보고 가장 적절한 전략을 자동 선택한다.
- `None`
  structured output을 <mark>**명시적으로 요청하지 않는다.**</mark>

<br>

위의 4가지중 `type` 스키마 타입을 전달하면 LangChain은 다음과 같이 행동한다.

- OpenAI, Anthropic, xAI 처럼 네이티브 structured output을 지원하는 모델이라면, `ProviderStrategy`를 사용한다.
- 그 외의 모델은 `ToolStrategy`를 사용한다.

<br>

## Provider strategy

OpenAI, Gemini, Anthropic, xAI 같은 모델들은 네이티브 structured output을 지원한다.

<mark>**이런 방식은 가장 신뢰도가 높은 방법**</mark>이다.

<mark>**모델 제공자(API)**</mark>가 구조화를 직접 보장한다.

<mark>**요청 단계**</mark>부터 이러한 “JSON/Pydantic 스키마로만 답하라”는 제약이 걸린다.

모델이 스키마를 벗어난 출력을 아예 못하도록 <mark>**제공자 레벨에서 검증**</mark>을 한다.

그래서 <mark>**실패 확률이 낮고**</mark>, <mark>**재시도 로직도 단순**</mark>하며, <mark>**결과 신뢰도가 가장 높다.**</mark>

```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None
```

<br>

`schema`

반드시 들어가야하는 값이다. 구조화된 출력을 정의하는 스키마 이다.

다음과 같은 값이 들어가야한다.

- Pydantic 모델
  `BaseModel` 을 상속한 클래스이며 필드 검증을 수행한다.
  에이전트 결과는 <mark>**Pydantic 인스턴스로 반환**</mark>된다.
- Dataclass
  타입 어노테이션이 있는 파이쎤 dataclass이다.
  에이전트 결과는 dict로 반환된다.
- TypedDict
  타입이 지정된 딕셔너리 클래스이다.
  결과는 dict으로 반환된다.
- JSON Schema
  JSON Schema 명세를 담은 딕셔너리이다.
  결과는 dict로 반환된다.

<br>

`strict`

boolean 파라미터로, 스키마를 엄격하게 강제할지 설정한다. OpenAI, xAI 등 일부 Provider에서 지원한다.

기본값은 None이다.

<br>

### 자동 ProviderStrategy 선택

스키마를 `ProViderStartegy`로 랩핑하지 않아도 <mark>**자동으로 랩핑**</mark>된다.

네이터브 structured output을 지원하는 모델이면, `create_agent.response_format`에 스키마 타입을 전달하면 ProviderStrategy가 자동으로 랩핑된다.

지원되는 스키마는 이전에 설명한 4가지 스키마들이다.

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # ProviderStrategy가 자동 선택된다
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

```

<br>

<mark>**정리**</mark>

Provider-native structured output은 <mark>**모델 제공자가 스키마를 직접 강제**</mark>하기 때문에 신뢰도와 검증 수준이 매우 높다.

사용 가능하다면 이 방식을 사용하는 것이 권장된다.

<br>

모델이 네이티브 structured output을 지원하는 경우,

`response_format=ProductReview`와 `response_format=ProviderStrategy(ProductReview)` 는 <mark>**기능적으로 동일**</mark>하다.

<br>

## Tool calling strategy

Tool calling 전략은 모델이 스키마를 이해하고 따라야 하는 <mark>**지침으로 취급**</mark>한다.

LangChain이 <mark>**스키마를 도구로 만들어 모델에게 제공**</mark>한다.

<mark>**모델이 이 도구를 호출**</mark>하면서 JSON/Pydantic 스키마 형태로 만들어낸다.

```python
class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")
```

LangChain은 위의 `ProductReview` 스키마를 다음과 같이 변환한다.

```python
@tool
def structured_output_tool(
    rating: int,
    sentiment: "positive | negative",
    key_points: list[str]
): ...
```

```python
tool name: structured_output_tool
arguments:
  rating: integer
  sentiment: "positive" | "negative"
  key_points: array[string]
```

모델이 받는 프롬프트에는 다음과같은 의미가 숨어있다.

> **“너는 지금 대답을 글로 하면 안된다.
> 필요하면 이 `structured_output_tool`을 호출해야한다.
> 그리고 호출할때는 이 파라미터 스키마를 반드시 지켜라.”**

<br>

이렇게 하는 이유는

- 그냥 “JSON 형식으로 답해” 라고 말하면,
  모델은 말을 안듣거나 필드를 빼먹을수있다.
- 하지만 “이 함수를 호출하라”고 도구로 주면
  모델은 <mark>**도구 호출을 해서 구조화된 JSON을 만들어낼 확률이 매우 올라간다.**</mark>

<br>

그리고 마지막단계로

모델이 파라미터를 llm이 맞게 `structured_output_tool` 을 호출하며

1. 스키마로 검증한다.
2. 실패하면 에러 메시지를 다시 모델에게 주고 재시도한다.
3. 성공하면 `structured_response`로 확정한다.

<br>

정리하자면,

<mark>**“이 형식으로 써라” 라고 말하지 않고, “이 함수를 호출해라” 라고 모델에게 명령한다.**</mark>

<br>

이후 LangChain이 검증하고, 틀리면 다시 이 도구를 시도하게 만든다.

Provider가 강제하는 것이 아니라, <mark>**LangChain 프레임워크 로직으로 검사**</mark>하는 방식이다.

그래서 모든 tool-calling 지원 모델에서 동작하지만<mark>**, 재시도/에러 처리 같은 방어 로직**</mark>이 중요해진다.

<br>

모델이 네이티브 structured output을 지원하면 <mark>**ProviderStrategy가 정답**</mark>이다.

지원하지 않으면 Tool calling이 대안이다.

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

<br>

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

`schema`

구조화된 출력 스키마이며, 이전의 4가지 타입을 지원한다.

<br>

`tool_messge_content`

구조화된 출력이 생성되었을때 반환되는 툴 메시지의 내용을 커스텀한다.

정하지 않으면, 기본적으로 구조화된 응답 데이터를 보여주는 메시지가 들어간다.

<br>

`handle_errors`

구조화된 출력 검증 실패시의 에러처리 전략이다. 기본값은 `True`이다.

- True: 모든 에러를 잡고 기본 에러 템플릿으로 처리한다.
- str: 모든 에러를 잡고 이 커스텀 메시지를 사용한다.
- type: 해당 예외 타입만 잡고 기본 메시지를 사용한다.
- tuple: 해당 예외 타입들만 잡고 기본메시지를 사용한다.
- Callable: 예외를 받아 에러 메시지를 생성하는 커스텀 함수이다.
- False: 재시도 없이 예외를 그대로 전파한다.

<br>

### `tool_message_content`

`tool_message_content` 파라미터는 structured output이 생성되었을때,

대화 히스토리 안에 남는 <mark>**툴 메시지의 내용을 커스터마이즈**</mark> 할 수 있게해주는 값이다.

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class MeetingAction(BaseModel):
    """회의 기록에서 추출한 작업."""
    task: str = Field(description="완료해야 할 구체적인 작업")
    assignee: str = Field(description="작업 담당자")
    priority: Literal["low", "medium", "high"] = Field(description="우선순위")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="작업 캡처되어 회의 노트에 추가되었다!"
    )
)

agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "회의 내용: 어진은 가능한 한 빨리 프로젝트 타임라인을 업데이트해야 한다"
        }
    ]
})

```

실행 결과

```yaml
================================ Human Message =================================
From our meeting: 어진은 가능한 한 빨리 프로젝트 타임라인을 업데이트해야 한다

================================== Ai Message ==================================
Tool Calls:
  MeetingAction (call_1)
 Call ID: call_1
  Args:
    task: 프로젝트 타임라인 업데이트
    assignee: 어진
    priority: high

================================= Tool Message =================================
Name: MeetingAction

작업 캡처되어 회의 노트에 추가되었다!
```

<br>

`tool_message_content`를 지정하지 않은경우

```yaml
================================= Tool Message =================================
Name: MeetingAction

Returning structured response: {
  'task': 'update the project timeline',
  'assignee': 'Sarah',
  'priority': 'high'
}
```

<br>

## `handle_errors`

```yaml
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

`handle_errors`

구조화된 출력 검증 실패시의 에러처리 전략이다. 기본값은 `True`이다.

- True: 모든 에러를 잡고 기본 에러 템플릿으로 처리한다.
- str: 모든 에러를 잡고 이 커스텀 메시지를 사용한다.
- type: 해당 예외 타입만 잡고 기본 메시지를 사용한다.
- tuple: 해당 예외 타입들만 잡고 기본메시지를 사용한다.
- Callable: 예외를 받아 에러 메시지를 생성하는 커스텀 함수이다.
- False: 재시도 없이 예외를 그대로 전파한다.

<br>

tool calling 스키마 전략은 <mark>**에러/재시도 방어 코드가 중요**</mark>하다.

LangChain은 이런 처리를 하기위해 똑똑한 <mark>**재시도(retry) 메커니즘을 제공**</mark>한다.

<br>

### Multiple structured outputs error

모델이 실수로 structured output tool을 <mark>**여러개 동시에 호출**</mark>하면,

에이전트는 `ToolMessage`로 에러 피드백을 제공하고 모델에게 다시 시도하라고 프롬프트한다.

```python
from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # 기본값: handle_errors=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

```

실행 결과

```yaml
================================ Human Message =================================
Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th

================================== Ai Message ==================================
Tool Calls:
  ContactInfo (call_1)
  Args:
    name: John Doe
    email: john@email.com
  EventDetails (call_2)
  Args:
    event_name: Tech Conference
    date: March 15th

================================= Tool Message =================================
Name: ContactInfo

Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.
Please fix your mistakes.

================================= Tool Message =================================
Name: EventDetails

Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.
Please fix your mistakes.

================================== Ai Message ==================================
Tool Calls:
  ContactInfo (call_3)
  Args:
    name: John Doe
    email: john@email.com

================================= Tool Message =================================
Name: ContactInfo

Returning structured response: {'name': 'John Doe', 'email': 'john@email.com'}
```

`Union[ContactInfo, EventDetails]`처럼 “둘 중 하나”를 기대했는데

모델이 둘 다를 뽑아서 <mark>**tool call을 2개**</mark> 해버릴 수 있다.

<br>

이때 LangChain은 “한 개만 내야 한다”는 에러를 `ToolMessage`로 전달한다.

모델은 그 피드백을 보고 재시도해서, 최종적으로 <mark>**하나의 structured response만**</mark> 반환한다.

<br>

### Schema validation error

기대한 스키마와 맞지 않으면(타입, 범위, 필수 필드 등), 에이전트는 구체적인 검증 에러 메시지를 ToolMessage로 제공하고, <mark>**모델에게 수정해서 다시 내라고 요구**</mark>한다.

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(ProductRating),  # 기본값: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
)

agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})

```

실행 결과

```yaml
================================ Human Message =================================
Parse this: Amazing product, 10/10!

================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_1)
  Args:
    rating: 10
    comment: Amazing product

================================= Tool Message =================================
Name: ProductRating

Error: Failed to parse structured output for tool 'ProductRating': 1 validation error for ProductRating.rating
  Input should be less than or equal to 5 [type=less_than_equal, input_value=10, input_type=int].
Please fix your mistakes.

================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_2)
  Args:
    rating: 5
    comment: Amazing product

================================= Tool Message =================================
Name: ProductRating

Returning structured response: {'rating': 5, 'comment': 'Amazing product'}
```

스키마에서 `rating`은 `1~5` 범위로 제한되어 있다(`ge=1`, `le=5`).

모델이 처음에 `rating=10`을 넣어버려서 Pydantic 검증이 실패한다.

LangChain은 <mark>**실패 원인을 정확히 적은 에러 메시지**</mark>를 툴 메시지로 모델에게 돌려준다.

모델은 그 피드백을 보고 `rating=5`로 수정해서 다시 tool call을 한다.

검증을 통과하면 최종 `structured_response`가 확정된다.

<br>

### 커스텀 에러 메시지

`handle_errors`에 문자열을 넣으면, 어떤 오류가 발생하든 <mark>**항상 이 고정된 메시지로 모델에게 재시도를 요청**</mark>한다.

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors="Please provide a valid rating between 1-5 and include a comment."
)
```

결과

```python
================================= Tool Message =================================
Name: ProductRating

Please provide a valid rating between 1-5 and include a comment.
```

<br>

<mark>**왜 이런 옵션이 필요할까?**</mark>

모델은 에러 메시지를 “디버그 로그”처럼 읽지 않는다.

그걸 <mark>**새로운 지시(prompt)**</mark> 로 해석한다.

<br>

예를 들면 이런 상황이다.

- 에러 메시지:
  “rating은 1 이상 5 이하의 정수여야 한다”
- 모델의 내부 반응:
  “아, 그럼 5를 넣으면 되겠네”
  → <mark>**근거 없이 값을 만들어냄**</mark>

<br>

즉, <mark>**에러를 ‘힌트’로 써서 추론을 조작**</mark>해버릴 수 있다.

<br>

특히 이런 경우에 위험하다.

- “만들지 말라”고 했는데
- 에러 메시지가 너무 구체적이라
- 모델이 값을 보정해서 <mark>**환각을 합법화**</mark>해버리는 경우

<br>

그래서 어떤 시스템에서는

에러 원인을 자세히 알려주기보다, <mark>**행동 규칙만 다시 상기**</mark>시키는 게 더 낫다.

```
"Please provide a valid rating between 1-5 and include a comment."
```

이 메시지는

- 왜 틀렸는지 설명하지 않는다
- 어떤 값을 써야 하는지도 말하지 않는다
- 단지 <mark>**형식과 범위만 재강조**</mark>한다

<br>

결과적으로 모델은

- 기존 텍스트에서 <mark>**근거를 다시 찾으려 시도**</mark>하거나
- 없으면 `None` 같은 안전한 값으로 두게 된다.

<br>

이 패턴은 특히 이런 상황에서 유용하다.

1. <mark>**정보를 만들어내면 안 되는 파서**</mark>

   - 로그 파서
   - 계약서/법률 문서 파서
   - 리뷰 요약기

   <br>

2. <mark>**UX 메시지가 중요한 경우**</mark>

   - 사용자에게 보여질 히스토리
   - 회의 노트, 액션 아이템 시스템

   <br>

3. <mark>**에러 로그를 단순화하고 싶을 때**</mark>
   - 수백 개의 다른 검증 에러를
   - 하나의 행동 지침으로 통합

<br>

<mark>**정리**</mark>

- `handle_errors=True`
  → “여기 틀렸어, 이 규칙을 봐” (상세 피드백)
- `handle_errors="고정 메시지"`
  → “규칙 지켜서 다시 해” (행동 교정)
- `handle_errors=False`
  → “실패, 개발자가 고쳐라” (엄격 실패)

<br>

### 특정 예외만 처리하기

`handle_errors`에 예외 타입을 지정하면,

해당 예외가 발생했을 때만 기본 에러 메시지로 재시도한다.

그 외 예외는 LangChain이 대신 처리하지 않고, <mark>**파이썬 예외로 바로 나온다.**</mark>

```python
# 예외 타입 하나.
ToolStrategy(
    schema=ProductRating,
    handle_errors=ValueError  # ValueError일 때만 재시도, 나머지는 예외 발생
)

# 여러 예외 타입도 가능하다.
ToolStrategy(
    schema=ProductRating,
    handle_errors=(ValueError, TypeError)  # ValueError와 TypeError에서만 재시도
)
```

<br>

<mark>**왜 이런 옵션이 필요할까?**</mark>

모든 에러를 재시도하게 하면 위험한 경우가 있다.

<br>

예를 들면,

- 스키마 설계 자체가 잘못된 경우
- 코드 버그(타입 정의 오류, 필드 이름 오타)
- 절대 모델이 고칠 수 없는 논리적 오류

<br>

이런 경우까지 “다시 해”라고 하면

<mark>**무한 재시도**</mark>나 이상한 결과로 이어진다.

<br>

그래서

- <mark>**모델 실수로 보이는 에러**</mark> → 재시도
- <mark>**코드/설계 문제로 보이는 에러**</mark> → <mark>**즉시 실패**</mark>

<br>

이렇게 선을 긋기 위해

“그 외 예외는 그대로 던진다”라는 옵션이 존재한다.

<br>

정리

<mark>**LangChain이 에러를 삼키거나 교정하지 않고, 파이썬 예외로 바로 올려보내서 개발자가 직접 처리하게 만든다는 의미**</mark>

<br>

### 커스텀 에러 핸들러 함수

에러가 발생했을때, 모델에게 똑같은 말을 시키면 <mark>**오히려 성능/안정성/보안이 떨어지는 경우**</mark>가 있다.

함수로 받아 에러 종류에 따라 다른 지시를 줄수있다.

```python
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"

```

이렇게 <mark>**함수**</mark>를 넘기면,

발생한 예외를 인자로 받아 <mark>**상황별로 다른 에러 메시지**</mark>를 만들어 모델에게 전달한다.

사용 예시는 다음과 같다.

```python
agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=custom_error_handler
    )
)

result = agent.invoke({
"messages": [{"role":"user","content":"Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

for msgin result["messages"]:
iftype(msg).__name__ =="ToolMessage":
print(msg.content)
elifisinstance(msg,dict)and msg.get("tool_call_id"):
print(msg["content"])
```

이때 에러별 출력은 다음과 같다.

- `StructuredOutputValidationError` 발생 시

```
================================= Tool Message =================================
Name: ToolStrategy

There was an issuewith the format.Try again.
```

- `MultipleStructuredOutputsError` 발생 시

```
================================= Tool Message =================================
Name: ToolStrategy

Multiple structured outputs were returned. Pick the most relevant one.
```

- 그 외 에러 발생 시

```
================================= Tool Message =================================
Name: ToolStrategy

Error: <error message>
```

<br>

1. <mark>**에러마다 “고치는 방법”이 다르다**</mark>

   같은 문장으로 재시도시키면, <mark>**어떤 에러에서는 효과가 없거나**</mark> 오히려 재발한다.

   <br>

2. <mark>**불필요한 내부 에러를 모델에게 노출하지 않기 위함이다**</mark>

   기본 <mark>**에러 메시지**</mark>는 상세 로그(필드명, 제약, 타입 등)를 그대로 포함한다.

   모델이 그걸 “힌트”로 삼아 <mark>**억지로 값을 맞출수 있다.**</mark>

   <br>

3. <mark>**UX/제품 톤을 통제하기 위함이다**</mark>

   “1 validation error for blah blah…” 같은 로그는 보기 흉하고 혼란스럽다.

   에러별로 톤으로 정리해 주면 <mark>**운영 경험**</mark>이 좋아진다.

   <br>

4. <mark>**재시도 비용을 줄이기 위함이다**</mark>

   모든 에러에 같은 재시도 메시지를 던지면 재시도가 길어지고 토큰/비용이 증가한다.

   반대로 “이 에러는 이렇게 고치면 된다”는 <mark>**정확한 지시를 주면 1~2번 안에 정리**</mark>되는 비율이 올라간다.

   특히 Union 스키마(여러 후보 중 하나 선택)에서 이 효과가 크다.

   <br>

5. <mark>**“계속 재시도”와 “즉시 실패”를 에러별로 갈라치기 위함이다**</mark>

   어떤 에러는 모델이 고칠 수 있다(형식/필드/범위).

   어떤 에러는 <mark>**코드/환경 문제**</mark>일 수 있다(네트워크, 툴 정의 충돌, 내부 예외).

   커스텀 핸들러는 “이건 사용자/모델 교정으로 해결”, “이건 개발자에게 올려야 한다”를 <mark>**분리해서 시스템을 안정화**</mark>한다.

### 에러 처리 비활성화

```python
response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=False# 모든 에러를 그대로 발생
)
```

`handle_errors=False`로 설정하면,

재시도나 교정 없이 <mark>**모든 에러를 즉시 예외로 던진다.**</mark>

<br>

### 정리

- 문자열 → 항상 같은 메시지로 재시도한다
- 예외 타입 / 튜플 → 지정한 예외에서만 재시도한다
- 함수 → 에러 종류에 따라 메시지를 동적으로 제어한다
- `False` → 에러를 잡지 않고 즉시 실패한다

<br>

즉, `handle_errors`는 ToolStrategy에서 <mark>**모델을 교정하며 재시도할지, 아니면 엄격하게 실패시킬지**</mark>를 결정하는 안전장치이다.

<br>

참고

- https://docs.langchain.com/oss/python/langchain/structured-output#provider-strategy
