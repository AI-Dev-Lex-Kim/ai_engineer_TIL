
## Long Context 최적화

에이전트가 단기 메모리를 사용하면, 이전 대화들을 불러와 <mark>**컨텍스트 윈도우를 초과**</mark>하게 된다.

일반적인 해결 방법은 다음과 같다.

- Delete messages - LangGraph state에서 <mark>**메시지를 삭제**</mark>한다.
- Trim messages - <mark>**처음**</mark> 또는 <mark>**마지막 N개의 메시지를 제거**</mark>한다.
- Summarize messages - 히스토리에서 <mark>**이전 메시지들을 요약**</mark>한다.
- Custom strategies - 다른 커스텀 전략

<br>

### Delete messages

메시지 히스토리를 관리하기 위해 state에서 <mark>**메시지를 삭제**</mark>할 수 있다.

특정 메시지나 전체 메시지 히스토리를 비우고 싶을때 사용할 수 있다.

`RemoveMessage`를 사용하면 된다.

특정 메시지 삭제

```python
from langchain.messages import RemoveMessage
from langchain.agents import AgentState

def delete_messages(state: AgentState):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

<br>

모든 메시지 삭제

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState

def delete_messages(state: AgentState):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

<br>

오래된 메시지 삭제

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    "gpt-5-nano",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.')]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
[('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
```

- 여기서는 `@before_model`이 아니라 `@after_model`을 쓴다.
- 모델이 한 번 응답을 만든뒤, 그 결과 까지 포함한 messages를 보고 앞쪽을 삭제 수행한다.

<br>

### Trim messages

메시지를 언제 잘라낼지 결정하는 방법은 메시지 히스토리의 토큰 수를 세고,

그 수가 <mark>**한계에 가까워질 때마다 잘라내는 것**</mark>이다.

LangChain을 사용하면, trim messages 유틸리티를 사용할 수 있다.

- 유지할 토큰수
- 바운더리(boundary)
  설정 등

<br>

`@before_model` 미들웨어 데코레이터를 사용하면 된다.

`@before_model` 이 데코레이터가 붙은 함수는 <mark>**매번 모델을 부르기 직전에 실행**</mark>된다.

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

		# messages가 3개 이하면 그대로 둔다.
    if len(messages) <= 3:
        return None  # No changes needed

		# 첫 메시지는 무조건 남긴다.
		# 보통 시스템 명령 같은 핵심이 들어가서 남기는 패턴이다.
    first_msg = messages[0]

    # 최근 메시지는 마지막 3개 또는 4개만 남긴다.
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""

```

<br>

```python
first_msg = messages[0]
```

- 첫 메시지는 무조건 남긴다.
- <mark>**보통 시스템 명령**</mark> 같은 핵심이 들어가서 남기는 패턴이다.

<br>

```python
recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
```

- 최근 메시지는 마지막3개 또는 4개만 남긴다.
- 채팅은 보통 Human → AI → Human → AI …로 번갈아 쌓인다.
- len(messages)가 짝수, 홀수냐에 따라 마지막을 잘랐을때 Human 과 AI 짝이 깨질 수 있다.
- 그래서 <mark>**홀수 일때**</mark>는 `-4:`로 더 남겨서 <mark>**턴 구조를 맞추려는 의도**</mark>이다.

<br>

```python
new_messages = [first_msg] + recent_messages

"messages": [
  RemoveMessage(id=REMOVE_ALL_MESSAGES),
  *new_messages
]
```

- 기존의 messages 전부를 먼저 삭제하고
- 그 다음에 new_messages만 다시 넣어라는 강제 리셋 패치이다.
- 그래서 state 남아있는 메시지 자체가 줄어든다.
- 결과적으로 저장되는 메모리도 같이 줄어들게된다.

<br>

Trim messages 장단점

장점

- 구현이 아주 단순
- 토큰, 비용, 지연시간이 매우 줄어든다.
- 스레드가 길어져도 안정적이다.

단점

- 잘려진 메시지 정보를 기억하지 못한다.

<br>

이런 점을 <mark>**보완하기 위한 아이디어**</mark>로는

- “시스템 메시지 + 최근 N턴” 대신
- <mark>**“시스템 메시지 + (중간 요약 메시지) + 최근 N턴”**</mark> 형태로 구성하면 좋다.
- 또는 “<mark>**중요 태그**</mark>(이름, 선호, 제약조건) 포함 메시지는 <mark>**유지**</mark> 같은 필터링”을 해도 좋다.

<br>

### Summarize messages

위의 두 전략은 메시지를 삭제해서 <mark>**정보를 잃을 수 있다**</mark>는 점이다.

LLM을 사용해 <mark>**이전 메시지들을 요약**</mark>해서 효과적인 최적화를 수행할수있다.

<br>

내장된 `SummarizationMiddleware`를 사용하면된다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob!
"""

```

<br>

```python
SummarizationMiddleware(
    model="gpt-4o-mini",
    trigger=("tokens", 4000),
    keep=("messages", 20)
)
```

- `model="gpt-4o-mini"`
  요약용 모델로 `gpt-4o-mini`를 사용한다.
  메인 모델보다 <mark>**요약모델은 보통 더 저렴한 모델**</mark>로 사용한다.
  `gpt-4o` → `gpt-4o-mini`
- `trigger=("tokens", 4000)`
  컨텍스트가 토큰 기준으로 <mark>**4000에 도달**</mark>하면, <mark>**요약**</mark>을 한다는 뜻이다.
  항상 요약하는게 아니라 길어졌을때만 발동한다.
- `keep=("messages", 20)`
  요약을 하더라도 최근 메시지 <mark>**20개는 원문 그대로 유지**</mark>한다는 뜻이다.
  최근 메시지는 정확도가 중요하니, 오래된건 요약으로 압축한다.

<br>

Summarize 전략 단점

- 요약 모델이 <mark>**중요한 내용을 누락**</mark>할수있다.
- 요약 과정에서 <mark>**미세한 왜곡**</mark>이 발생할 수 있다.
- 요약은 설정값에 따라 <mark>**매번 조금씩 달라**</mark>질 수 있다.(temperature 등)

<br>

이런 점을 보완한 아이디어

- “이전 대화 요약 + 최근 N개 메시지” + <mark>**“핵심 정보는 state로 구조화 저장”**</mark>
- 메시지가 요약,삭제되도 핵심은 유지하게 된다.

<br>

## Use Memory

단기 메모리(state)에 접근해서 사용하는 경우는 보통 다음과 같다.

1. Tools
2. Prompt
3. Before model
4. After model

<br>

### 1. Tools

에이전트에 Tool을 등록한 뒤, 단기 메모리 state에 접근해 업데이트 및 활용할 수 있다.

<br>

1. Read Memory

   `runtime` 파라미터를 사용해서 도구 안에서 <mark>**단기메모리에 접근**</mark>한다.

   ```python
   from langchain.agents import create_agent, AgentState
   from langchain.tools import tool, ToolRuntime

   class CustomState(AgentState):
       user_id: str

   @tool
   def get_user_info(
       runtime: ToolRuntime
   ) -> str:
       """Look up user info."""
       user_id = runtime.state["user_id"]
       return "User is John Smith" if user_id == "user_123" else "Unknown user"

   agent = create_agent(
       model="gpt-5-nano",
       tools=[get_user_info],
       state_schema=CustomState,
   )

   result = agent.invoke({
       "messages": "look up user information",
       "user_id": "user_123"
   })
   print(result["messages"][-1].content)
   # > User is John Smith.
   ```

   - `state`에는: user_id, thread 진행상태, 확정된 슬롯값 같은 <mark>**신뢰 가능한 키**</mark>
   - `tool(runtime)`은: 그 키를 이용해 DB/API 조회 후 <mark>**팩트 리턴**</mark>
   - 모델은: 툴이 준 팩트를 근거로 답변

   <br>

2. Update Memory

   실행 중에 에이전트의 <mark>**단기 메모리를 수정**</mark>할 수 있다.

   중간 결과를 지속시키거나, 정보를 도구 또는 프롬프트에 제공할때 유용하다.

   ```python
   from langchain.tools import tool, ToolRuntime
   from langchain_core.runnables import RunnableConfig
   from langchain.messages import ToolMessage
   from langchain.agents import create_agent, AgentState
   from langgraph.types import Command
   from pydantic import BaseModel

   class CustomState(AgentState):
       user_name: str

   class CustomContext(BaseModel):
       user_id: str

   @tool
   def update_user_info(
       runtime: ToolRuntime[CustomContext, CustomState],
   ) -> Command:
       """Look up and update user info."""
       user_id = runtime.context.user_id
       name = "John Smith" if user_id == "user_123" else "Unknown user"
       return Command(update={
           "user_name": name,
           # update the message history
           "messages": [
               ToolMessage(
                   "Successfully looked up user information",
                   tool_call_id=runtime.tool_call_id
               )
           ]
       })

   @tool
   def greet(
       runtime: ToolRuntime[CustomContext, CustomState]
   ) -> str | Command:
       """Use this to greet the user once you found their info."""
       user_name = runtime.state.get("user_name", None)
       if user_name is None:
          return Command(update={
               "messages": [
                   ToolMessage(
                       "Please call the 'update_user_info' tool it will get and update the user's name.",
                       tool_call_id=runtime.tool_call_id
                   )
               ]
           })
       return f"Hello {user_name}!"

   agent = create_agent(
       model="gpt-5-nano",
       tools=[update_user_info, greet],
       state_schema=CustomState,
       context_schema=CustomContext,
   )

   agent.invoke(
       {"messages": [{"role": "user", "content": "greet the user"}]},
       context=CustomContext(user_id="user_123"),
   )
   ```

   <br>

   `Command(update=...)`

   - <mark>**Command**</mark>를 사용해서 state을 업데이트한다.
   - 그래서 “툴 실행 → 결과 state 저장 → 툴이 그 state를 읽음” 흐름이 가능하다.

   <br>

   `CustomeContext`, `CustomState` 분리

   - `runtime.context`
     <mark>**불변하는 외부 정보**</mark>들이다.
     ex) user_id, auth 정보
   - `runtime.state`
     <mark>**대화 진행 중**</mark>에 저장할 값들이다.
     ex) messages, user_name

   <br>

   이렇게 분리해서

   Context: `user_id`는 요청마다 주입

   State: `user_name`은 한번 조회하면 state 저장해서 재사용한다.

   <br>

   ```python
   ToolMessage(
       "Please call the 'update_user_info' tool it will get and update the user's name.",
       tool_call_id=runtime.tool_call_id
   )
   ```

   - 이 <mark>**ToolMessage가 어떤 tool call의 결과**</mark>인지 에이전트에게 알려준다.
   - 일부 Provider는 이 두개의 짝이 맞아야 메시지를 처리해준다.

   <br>

   <mark>**중간결과 state를 저장하는 이유**</mark>

   1. 한번 조회한 값을 <mark>**반복하지 않게**</mark>
   2. <mark>**후속 툴과 프롬프트**</mark>에서 접근할수있게

<br>

### 2. Prompt

대화 히스토리, state 스키마 필드 기반으로 <mark>**동적 프롬프트**</mark>를 만들 수 있다.

미들웨어에서 단기 메모리에 접근해 사용한다.

```python
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class CustomContext(TypedDict):
    user_name: str

def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)

for msg in result["messages"]:
    msg.pretty_print()
```

<br>

```python
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    ...
    return system_prompt
```

`@dynamic_prompt`는 <mark>**매 호출마다 시스템 프롬프트를 동적으로 생성**</mark>한다.

실햄 시점마다 프롬프트를 만들어서 모델에 넣어준다.

그래서 사용자 또는 상태에 따라

- 호칭
- 말투
- 정책
- 출력 포맷
  같은 것들을 즉석에서 바꿀수있다.

<br>

```python
user_name = request.runtime.context["user_name"]
```

`runtime.context`에서 가져온다.

- context: 불변하는 사용자 정보
- state: 대화 스레드에서 누적되는 단기 메모리

여기서는 `user_name`을 매 요청마다 확실히 주입하는 값으로 본거라 context에서 가져왔다.

<br>

### 3. Before model

![image.png](<../images//Agent(3)%20-%20Short-term%20memory%20(Delete,%20Trim,%20Summarize)/1.png>)

<mark>**모델 호출전에 메시지**</mark>를 처리하기 위해 `@before_model` 미들웨어를 사용해 단기 메모리에 접근할 수 있다.

<br>

이전 Trim messages의 예제코드가 이런 경우이다.

```python
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
```

<br>

### 4. After model

![image.png](<../images/Agent(3)%20-%20Short-term%20memory%20(Delete,%20Trim,%20Summarize)/2.png>)

<mark>**모델 호출 이후**</mark>에 메시지를 처리하기 위해, `@after_model` 미들웨어를 사용할 수 있다.

```python
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)
```

최근 응답을 가져오고 “password”, “secret” 같은 <mark>**금칙어가 포함**</mark>되면,

방금 생성된 메시지를 state에서 삭제한다.

- 민감정보/금칙어 필터링
- 정책 위반 출력 제거
- 출력 포맷 검증(JSON 형식이 아니면 제거/재시도 유도)
- 너무 긴 답변 삭제 후 재요청
- 특정 토큰/마크업 제거(예: html/script 제거)

<br>

<mark>**삭제 이후 다음 행동**</mark>들이 필요하다.

1. <mark>**대체 메시지 넣기**</mark> - “민감한 내용이라 출력할 수 없습니다”
2. <mark>**다시 모델 호출**</mark>하도록 유도 - 리트라이 전략

삭제만하면 사용자 입장에서는 빈 응답이 되는 상황이 생기게된다.

<br>

<mark>**before_model과 after_model의 역할 차이**</mark>

- `before_model`
  → 모델이 답변 만들기 전에 “입력 메시지 정리/변형”
- `after_model`
  → 모델이 답변 만든 뒤에 “출력 결과 검증/정리/삭제”

<br>

참고

- https://docs.langchain.com/oss/python/langchain/short-term-memory
- https://docs.langchain.com/oss/python/langchain/middleware/overview#summarization
