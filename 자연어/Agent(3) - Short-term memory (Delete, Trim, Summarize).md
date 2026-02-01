- [Agent(3) - Agent State, Short-term memory](#agent3---agent-state-short-term-memory)
  - [Long Context가 문제가 되는이유](#long-context가-문제가-되는이유)
  - [단기 메모리 생성](#단기-메모리-생성)
  - [프로덕션시 단기메모리](#프로덕션시-단기메모리)
  - [에이전트 상태 스키마 설정](#에이전트-상태-스키마-설정)

# Agent(3) - Agent State, Short-term memory

메모리는 LLM이 **다음 응답을 만들때 참고**로 사용할 수 있도록 유지, 관리되는 정보이다.

에이전트에서 메모리는 매우 중요한 역활을 가진다.

더 복잡한 작업을 수행할 수록 기억하고, **피드백**을 통해 **학습**하여 효율성과 사용자 만족도를 높인다.

<br>

단기 메모리(Short-term memory)는 보통 다음과 같은 형태이다.

- 지금 대화에서 오간 메시지들
- 바로 이전 질문과 답변
- 현재 세션에서만 유효한 정보

<br>

즉, 브라우저 새로고침하면 사라지는 기억들이다.

<br>

단기 메모리를 사용하는 유용한 주요 이유는 **대화기록이 점점 길어지는 문제**가 있기 때문이다.

### Long Context가 문제가 되는이유

첫번째로 물리적 한계가 있다.

- LLM에는 **컨텍스트 윈도우 길이 제한**이 있다.
- 메시지가 너무 많으면, 앞부분이 잘리거나, 에러가 발생한다.

<br>

두번째로 성능 한계가 존재한다.

- 설령 컨텍스트가 다 들어가도
  - 중요하지 않은 이전 정보
  - 이미 끝난 주제
  - 현재 질문과 무관한 대화
    이것들이 **모델의 주의를 분산**시키게된다.

<br>

이런 점때문에 답변품질, 추론속도가 안좋아지고 토큰 비용도 높아지게된다.

<br>

이런것들을 해결하기위해서, 애플리케이션은 **오래된 정보를 제거**한다.

- 모델이 스스로 기억을 삭제하는 것이 아니라
- 애플리케이션이 어떤 정보를 컨텍스트에 **넣지 않을지 선택**한다는 것이다.

<br>

대표적인 전략은 다음과 같다.

- 최근 N개 메시지만 유지
- 요약(Summarization)후 원문 삭제
- 중요 태그만 남기고 나머지 제거
- RAG로 필요한 정보만 다시 주입

<br>

단기 메모리는 **‘대화를 얼마나 많이 저장하느냐’**가 아니라, **‘다음 응답에 꼭 필요한 맥락만 얼마나 잘남기느냐’**의 문제이다.

<br>

## 단기 메모리 생성

단기 메모리를 추가하려면, 에이전트를 생성할때 **체크포인터(checkpointer)를 생성**해야한다.

체크포인터는 스레드별 대화 상태(state)를 저장하고 다시 불러오기 위한 저장소이다.

<br>

만약 없다면,

- agent.invoke() 한번 실행하고 끝난다.
- 이전 대화는 완전히 사라진다

있다면

- 스레드 ID 기준으로 **상태를 저장**한다.
- 다음 호출에서 **이전 대화**를 그대로 이어서 사용한다.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)

```

<br>

**thread_id가 하는 역활**

```python
{"configurable": {"thread_id": "1"}}
```

“이 호출은 `thread_id=1`번 대화에 속한다” 라는 뜻이다.

- 같은 `thread_id` → 같은 대화 이어서 진행
- 다른 `thread_id` → 완전히 다른 대화

<br>

**메모리가 저장되고 읽히는 시점**은 다음과 같다.

1. `agent.invoke()` 시작

   → 체크포인터에서 기존 state 로드

2. LLM 호출 / Tool 호출

   → state 변경(message 추가)

3. 응답 완료

   → state 저장

4. 다음 에이전트 호출 시작(다음 step)

   → 다시 state 읽기

step 단위로 자동 저장, 복구가 이루어진다.

<br>

## 프로덕션시 단기메모리

**InMemorySaver 사라지는 시점**

`InMemorySaver`는 프로세스 종료시 매모리는 사라져서, **개발 / 테스트용**으로 적절하다.

<br>

그래서 실제 서비스 환경(프로덕션)에서는 보통

- SQLite
- Postgress
- Redis
  같은 **외부 저장소 기반 체크포인터**를 사용한다.

```bash
pip install langgraph-checkpoint-postgres
```

<br>

```python
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

<br>

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
```

- DB 연결 기반 체크포인트 객체 생성하고
- `with` 블록이 끝나면 연결을 정리한다.

<br>

```python
checkpointer.setup()
```

- 체크포인터가 사용할 **테이블을 생성**한다.
- `setup()`은 매 요청마다 하지 말고, **앱 시작 시 1회 실행**하는 형태가 일반적이다.

<br>

```python
agent = create_agent(..., checkpointer=checkpointer)
```

이제 이 agent는

- invoke 시작 시: DB에서 thread_id에 해당하는 state를 로드한다
- step 완료 시: state를 DB에 저장한다.

<br>

```python
agent.invoke(
  {"messages": [...]},
  {"configurable": {"thread_id": "1"}}
)
```

이렇게 호출하면, `thread_id=”1”` 의 상태가 **Postgres에 저장**되고 다음 호출에서 그대로 이어진다.

<br>

## 에이전트 상태 스키마 설정

기본적으로 에이전트는 **단기메모리를 관리**하기 위해 `AgentState`를 사용한다.

구체적으로 `AgentState`의 `messages` 키를 통해 대화기록을 관리한다.

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})

```

<br>

```python
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict
```

- `user_id`
- `preferences`

이런 값들은 매 요청마다 외부 DB에서 다시 가져오지 않아도 되고,

**thread_id 기준으로 대화 state와 함께 저장**된다.

즉, 이 대화 스레드에서의 사용자 설정, 상태를 유지하기 좋다.

<br>

```python
result = agent.invoke(
  {
    "messages": ...,
    "user_id": "user_123",
    "preferences": {"theme": "dark"}
  },
  {"configurable": {"thread_id": "1"}}
)
```

이 호출이 끝나면

- `thread_id="1"`의 state 안에
  `messages`, `user_id`, `preferences`가 함께 저장된다.
- 다음에 같은 `thread_id`로 호출하면
  이 값들이 다시 로드되어 이어서 쓸 수 있다.

<br>

참고

- https://docs.langchain.com/oss/python/langchain/short-term-memory
- https://docs.langchain.com/oss/python/langchain/middleware/overview#summarization
