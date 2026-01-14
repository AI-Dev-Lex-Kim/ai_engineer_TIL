- [Agent(2) - Tools (ToolRuntime)](#agent2---tools-toolruntime)
  - [도구 생성](#도구-생성)
  - [도구 이름 네이밍](#도구-이름-네이밍)
  - [도구 추가 설명](#도구-추가-설명)
  - [고급 스키마 정의](#고급-스키마-정의)
  - [ToolRuntime](#toolruntime)
    - [State 설정](#state-설정)
    - [State 업데이트](#state-업데이트)
    - [Context 설정](#context-설정)
    - [Memory(Store)](#memorystore)
    - [Stream Writer](#stream-writer)

# Agent(2) - Tools (ToolRuntime)

도구(tool)는 에이전트가 할 수 있는 일을 확장한다.

- 실시간 데이터를 가져오고
- 코드를 실행하며
- 외부 데이터베이스를 조회하고
- 실제세계에서 어떤 행동을 취할 수 있게 해준다.

<br>

“도구”란 입력을 받고 출력을 가진 <mark>**호출 가능한 함수**</mark>이다.

모델은 사용자의 입력 대화를 분석해

- 어떤 함수(tool)를 호출할지
- 어떤 입력값을 함수에 전달할지
  결정한다.

<br>

## 도구 생성

툴을 만드는 방법은 `@tool` 데코레이터를 사용하는 것이다.

함수의 <mark>**docstring**</mark>을 사용해 생성된 툴이 <mark>**어떤 역활**</mark>인지, <mark>**언제**</mark> 이 도구를 사용해야 하는지 설명한다.

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """ <- docstring
    query와 매칭되는 기록을 고객 데이터베이스에서 검색해라.

    Args:
        query: 찾아야할 검색어
        limit: 출력될 최대 결과 개수
    """ <- docstring
    return f"Found {limit} results for '{query}'"
```

- <mark>**타입 힌트**</mark>
  타입 힌트도 반드시 넣어주어야 하는 필수이다.
  타입 힌트가 <mark>**도구의 입력 스키마를 정의**</mark>하기 때문이다.
- <mark>**docstring**</mark>
  docstring은 도구의 <mark>**목적**</mark>을 모델이 이해할 수 있도록 <mark>**간결하고 명확**</mark>하게 작성해야한다.

<br>

## 도구 이름 네이밍

기본적으로 도구 이름은 <mark>**함수 이름**</mark>과 똑같이 자동으로 생성된다.

더 의미 있는 이름이 필요하면 직접 지어줄 수 있다.

```python
@tool("web_search")  # 커스텀 이름
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search
```

<br>

## 도구 추가 설명

모델에게 더 명확한 사용 가이드를 주고 싶다면 자동 생성된 설명을 덮어쓸 수 있다.

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

<br>

## 고급 스키마 정의

<mark>**복잡한 입력**</mark>이 필요한 경우, <mark>**Pydantic 모델**</mark>이나 <mark>**JSON Schema**</mark>를 사용해 입력 구조를 정의할 수 있다.

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

이 방식은 입력 <mark>**필드별 의미**</mark>와 <mark>**제약 조건**</mark>을 <mark>**명확**</mark>히 정의할 수 있어, 모델이 도구를 더 정확하게 호출하도록 돕는다.

이에 대한 더 자세한 설명은 이전에 작성한 글인 [structured output](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Agent(1)%20-%20Structured%20output.md>)에서 볼수있다.

<br>

## ToolRuntime

도구는 단순 함수일때보다, 에이전트 <mark>**상태/컨텍스트/메모리**</mark>에 접근할 수 있을때 훨씬 강력한다.

이렇게 되면 다음이 가능해진다.

- 이전 대화 맥락을 반영한 판단을 한다.
- 사용자별로 응답을 개인화한다.
- 세션을 넘어 정보를 기억한다.
- 실행 중 진행 상황을 스트리밍 한다.

<br>

이런 것들을 가능하게 하는 것이 `ToolRuntime`이다.

사용자 ID, 앱 설정 같은 의존성을 실행 시점에 주입하는 메커니즘이다.

![image.png](<../images/Agent(2)/1.png>)

`ToolRuntime` 파라미터로 다음 런타임 정보들에 접근 할 수 있다.

- State - 실행 중 계속 <mark>**변하는 상태**</mark> 데이터이다. ex) Messages, counters, 커스텀 필드
- Context - <mark>**변하지 않는 설정**</mark> 데이터이다. ex) user id, session 정보, 앱 설정
- Store - 대화를 넘어 유지되는 <mark>**장기 메모리**</mark>이다.
- Stream Writer - 도구 실행 중 중간 결과를 <mark>**실시간으로 전송**</mark>한다.
- Config - 실행에 사용된 RunnableConfig이다. RunnableConfig는 실행 한번을 어떻게 돌리지 정하는 설정값이다.
- Tool Call ID - 현재 도구 호출의 고유 ID이다.

<br>

`ToolRuntime`은 도구에 자동을 주입되는 런타임 설정 객체이다.

이 파라미터는 <mark>**LLM에게 보이지 않으며, 영향을 주지 않는다.**</mark>

<br>

### State 설정

State는 현재 그래프 실행 중 흐르는 <mark>**가변 데이터**</mark> 이다.

```python
from langchain.tools import tool, ToolRuntime

# 현재 대화 상태 접근
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# 상태 필드 수정
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

`runtime.state["messages"]` 로 전체 메시지에 접근 할 수 있다.

`runtime.state.get("user_preferences")` 처럼 사용자 설정을 읽어 올 수 있다.

중요한 점은 `runtime` 파라미터는 모델 입력 스키마에 포함되지 않는다.

<br>

### State 업데이트

`Command` 을 사용해 <mark>**에이전트 상태**</mark> 또는 그래프 실행 흐름을 <mark>**업데이트**</mark> 할수있다.

<br>

아래 예시는 다음과 같다.

- 모든 대화 메시지를 삭제한다.
- agent state에 저장된 user_name 값을 갱신한다.

<br>

도구는 단순 문자열이 아니라 <mark>**그래프 상태 변경 명령**</mark>을 반환할 수 있다.

```python
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime
f
# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})
```

<br>

### Context 설정

Context는 <mark>**변하지 않는 설정 정보**</mark>이다.

- 사용자 ID
- 세션 정보
- 앱 전역 설정

<br>

`runtime.context`를 통해 접근할 수 있다.

Context는 보통 `@dataclass`로 스키마를 정의하고, 에이전트 실행 시 주입한다.

```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)
```

<br>

### Memory(Store)

Store는 대화를 넘어 유지되는 <mark>**메모리**</mark>이다.

- 사용자별 정보 저장
- 여러 세션에서 동일 데이터 재사용

<br>

접근 방식

- `runtime.store.get(namespace, key)`로 조회한다.
- `runtime.store.put(namespace, key, value)`로 저장한다.

이를 통해 “이전에 저장한 사용자 정보”를 다음 대화에서 다시 불러올 수 있다.

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev
```

<br>

### Stream Writer

`runtime.stream_writer`는 도구 실행 중 중간 상태를 <mark>**실시간으로 전달**</mark>하는 기능이다.

예를 들면

- “도시 데이터 조회중”
- “데이터 수집 완료”

이 기능을 사용하면 <mark>**사용자는 도구가 멈춘 것처럼 보이지 않고**</mark>, 진행 상황을 바로 확인 할 수 있다.

단 이기능은 <mark>**LangGraph 실행 컨텍스트 안에서만**</mark> 동작한다.

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
```

<br>

참고

- https://docs.langchain.com/oss/python/langchain/tools
