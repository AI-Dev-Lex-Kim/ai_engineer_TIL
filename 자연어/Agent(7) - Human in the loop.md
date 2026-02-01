- [Agent(7) - Human in the loop](#agent7---human-in-the-loop)
  - [HumanInTheLoopMiddleware](#humanintheloopmiddleware)
  - [approve](#approve)
  - [reject](#reject)
  - [edit](#edit)
  - [replace](#replace)
  - [cancle](#cancle)
  - [feedback](#feedback)

# Agent(7) - Human in the loop

![image.png](<../images//Agent(7)%20-%20Human%20in%20the%20loop/1.png>)

에이전트가 <mark>**모든 행동을 자동으로 처리하지 않게**</mark> 하고싶을 수 있다.

다음과 같은 행동을 <mark>**사람이 중간에 개입**</mark>에 실행할 수 있다.

1. <mark>**민감한 조치를 승인**</mark>
2. <mark>**누락 컨텐츠 추가**</mark>
3. <mark>**에이전트 디버깅**</mark>

<br>

## HumanInTheLoopMiddleware

```python
from langchain.tools import tool, ToolRuntime

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    # take email from state
    return runtime.state["email"]

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    # fake email sending
    return f"Email sent"
```

<br>

에이전트 생성시 사람이 중간개입할 수 있는 <mark>**미들웨어를 넣어준다.**</mark>

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

class EmailState(AgentState):
    email: str

agent = create_agent(
    model="gpt-5-nano",
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False, # 승인이 필요없음
                "send_email": True, # 승인이 필요함
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)
```

<br>

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "read_email": False, # 승인이 필요없음
        "send_email": True, # 승인이 필요함
		},
```

- <mark>**True는 승인이 필요**</mark>하다는 뜻이다.
- <mark>**False**</mark>는 승인이 필요없으며 <mark>**자동으로 승인**</mark>된다.

<br>

## approve

중지 된뒤, <mark>**사람의 승인을 하는 방법**</mark>은 다음과 같다.

```python
from langgraph.types import Command

response = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config # Same thread ID to resume the paused conversation
)

pprint(response)
```

LLM이 의사 결정(dicision)을 한다.

- send_email 툴을 호출하게 된다.

<br>

그런데 사람이 중간에 멈춰서

- 내용이 위험 or 틀림 or 부적절 or 조금만 수정하고 싶을때가 있다.

<br>

이때 <mark>**decision을 어떻게 처리할지 정하는게**</mark> `type` 이다

<br>

## reject

```python
response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    # An explanation of why the request was rejected
                    "message": "No please sign off - Your merciful leader, Seán."
                }
            ]
        }
    ),
    config=config # Same thread ID to resume the paused conversation
    )

pprint(response)
```

`decisions` `type`을 `reject`을 하면 거절하게 된다.

`message` 에 거절한 이유를 선택적으로 작성해도된다.

- 거절한 이유를 작성하면, 에이전트는 이유를 바탕으로 수정한뒤 다시 전달하려 한다.
- 수정한뒤, 다시 한번 승인을 요청받게 된다.

<br>

## edit

`edit`는 에이전트가 결정한 툴은 맞는데 <mark>**내용을 사람이 직접 수정**</mark>하고 싶을때 사용한다.

<br>

예를들어서

에이전트가

```python
send_email(
  body="너는 해고야!!"
)
```

너무 공격적이게 작성했다.

여기서는 툴자체는 유지하고 내용만 변경하려면 `edit` `type` 사용한다.

```python
"type": "edit",
"edited_action": {
    "name": "send_email",
    "args": {"body": "We regret to inform you that your contract has ended."}
}
```

<br>

## replace

에이전트가 <mark>**Tool 선택을 잘못했을때**</mark> 사용한다.

예를들어

```python
send_email({ "body": "Report attached" })
```

이메일 말고 Slack으로 보내야한다면, replace를 사용해야한다.

```python
{
  "type": "replace",
  "replacement_action": {
      "name": "send_slack_message",
      "args": {"text": "Report is ready"}
  }
}

```

Tool을 교체했다.

<br>

`“name”`을 교체하려는 Tool 이름을 넣어준다.

<br>

## cancle

<mark>**액션 자체를 없던일**</mark>로 하고싶을때 사용한다.

- 보안, 법적, 윤리, 중복 실행 방지

<br>

예시

```python
delete_user_account({ "user_id": 123 })
```

삭제를 하는 행동이 위험하다고 사람이 느꼈다면

```python
{
  "type": "cancel"
}
```

아무 액션도 실행하지 않고 다음 상태로 간다.

<br>

## feedback

실행하지 말고 사<mark>**람이 피드백을 반영해서 다시 추론**</mark>을 시킬때 사용한다.

- 조건을 잘못 이하핼때
- 맥락을 놓칠때
- 더 나은 전략이 있을때

<br>

예시

```python
book_flight({ "date": "2024-08-01" })
```

사람이 느낄때 성수기라 너무 비싸서 예약을 하면 안된다.

```python
{
  "type": "feedback",
  "feedback": "8월은 너무 비싸서 더 저렴한 달로 변경해줘."
}
```

Tool을 실행하지말고 LLM이 다시 추론하게 시킨다.

<br>

| 타입       | 한 줄 요약     | 실행       |
| ---------- | -------------- | ---------- |
| `edit`     | content만 고침 | 바로 실행  |
| `replace`  | 행동 교체      | 바로 실행  |
| `cancel`   | 행동 금지      | 실행 안 함 |
| `feedback` | 다시 생각      | 재추론     |

<br>
