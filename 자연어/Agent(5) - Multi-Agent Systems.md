- [Agent(5) - Multi-Agent Systems](#agent5---multi-agent-systems)
  - [Agent State](#agent-state)
  - [Sub Agent tool](#sub-agent-tool)
  - [Prompt](#prompt)
    - [Supervisor Agent](#supervisor-agent)
    - [Sub Agent](#sub-agent)
    - [Sub Agent Tool](#sub-agent-tool-1)

# Agent(5) - Multi-Agent Systems

에이전트의 태스크가 복잡해질수록, 에이전트의 성능은 점점 나빠진다.

예를들어 에이전트가 고품질 시장조사 보고서를 작성하길 원한다고 해보자.

<br>

이작업 흐름은 꽤 길고 시기에 따라 서로다른 전문가의 기술이 필요할 수 있다.

- 개요를 제안하고
- 웹에서 조사하고
- 출처를 검증하고
- 보고서를 작성한뒤
- 편집하는 과정까지

![image.png](<../images/Agent(5)%20-%20Multi-Agent%20Systems/1.png>)

이 모든것을 하나의 에이전트에게 맡기면,

- 선택할 도구들이 너무 많아서 고르기 어려워지며
- 컨텍스트 윈도우를 넘쳐서 정보가 사라져 결과가 안좋아질수있다.

<br>

![image.png](<../images/Agent(5)%20-%20Multi-Agent%20Systems/2.png>)

멀티 에이전트 시스템은 여러 특화된 에이전트로 분해해서 함께 협력해 문제를 해결하도록 만든다.

모든 단계를 하나의 에이전트가 처리하도록 의존하는 것보다 매우 성능이 뛰어나진다.

<br>

상위 에이전트가 판단해 하위에이전트에게 작업을 맡기는 형태이다.

```python
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2
```

<br>

```python
from langchain.agents import create_agent

# create subagents

subagent_1 = create_agent(
    model='gpt-5-nano',
    tools=[square_root]
)

subagent_2 = create_agent(
    model='gpt-5-nano',
    tools=[square]
)
```

<br>

```python
from langchain.messages import HumanMessage

@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

@tool
def call_subagent_2(x: float) -> float:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content

## Creating the main agent

main_agent = create_agent(
    model='gpt-5-nano',
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.")
```

<br>

```python
question = "What is the square root of 456?"

response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
```

```python
from pprint import pprint

pprint(response)
```

이러한 형태는 코드를 출력하는것만으로는 추적하기 굉장히 힘들어진다.

LangSmith를 사용해서 전체 흐름을 보는게 굉장히 디버깅하기 쉽다.

<br>

## Agent State

멀티 에이전트는 Agent State를 사용하는 경우가 많다.

서로 다른 Agent들끼리 가져온 정보를 참조하는 경우가 꽤 많기 때문이다.

<br>

update state tool을 생성해서 Supervisor agent가 지속적으로 업데이트 할수 있도록 만들어준다.

```python
@tool
def update_agent_state(destination: str | None, flight: str | None, guest_count: str | None, music_genre: str | None, runtime: ToolRuntime) -> Command:
    """destination, guest_count, music_genre, flight를 모두 안다면 상태를 업데이트 한다."""

    return Command(update={
        "destination": destination,
        "guest_count": guest_count,
        "music_genre" : music_genre,
        "flight": flight,
        "messages" : [ToolMessage(
            "성공적으로 에이전트 상태를 업데이트 했습니다.",
            tool_call_id=runtime.tool_call_id,
        )]
    })
```

<br>

읽어오는 경우도 필요하다.

```python
@tool
def read_agent_state(runtime: ToolRuntime):
    """destination, guest_count, music_genre, flight 상태 정보를 가져온다."""

    destination: str | None = runtime.state.get('destination')
    flight: str | None = runtime.state.get('flight')
    guest_count: str | None = runtime.state.get('guest_count')
    playlist: str | None = runtime.state.get('playlist')

    return destination, flight, guest_count, playlist
```

<br>

## Sub Agent tool

다른 서브 에이전트들의 정보를 참고 하는경우가 많다.

서브 에이전트 툴에서 Agent State을 가져올때,

일반적으로 `runtime.state[”<state name>”]` 방식으로 상태를 가져온다.

이런 식으로 데이터를 접근하는 경우, Agent State 값이 없으면 에러가 발생한다.

<br>

Agent State에 Defualt 값을 줄수있다.

`runtime.state.get(”<state name>”, “<defualt>”)` 을 사용한다.

- 기본적으로 None이 들어가있다.
- 작성해주면 defulat 값을 넣어줄수있다.

```python
@tool
def search_venues(runtime: ToolRuntime) -> str:
    """venue agent는 결혼식에 가장 적합한 장소를 추천해준다."""
    destination = runtime.state.get('destination', '한국')
    guest_count = runtime.state.get('guest_count', '100명')
    query = f'{destination}에서 {guest_count}에 알맞은 최고의 결혼식 장소를 찾아야한다.'

    response = venue_agent.invoke(
            {'messages' : [HumanMessage(query)]}
        )
    return response['messages'][-1].content
```

<br>

## Prompt

### Supervisor Agent

감독형 에이전트의 시스템 프롬프트는 다음과 같이 작성해야한다.

1. 감독형 에이전트가 어떤 전문가인지 명시한다. ex) 너는 웨딩 플래너 이다.
2. 다른 전문 에이전트에게 업무를 분담한다고 작성한다.

   ```python
   항공편, 장소, 플레이리스트를 담당하는 전문 에이전트에게 업무를 분담해야한다.
   ```

3. 상태를 업데이트 하기위해 정보를 찾아야한다고 작성한다.

   ```python
   먼저 상태를 업데이트 하기위해 필요한 모든 정보를 찾아야한다.
   ```

4. 이후 업무 분담을 해야한다고 한다.

   ```python
   그 다음 업무를 분담해야한다.
   ```

5. 서브에이전트에게 정보를 받은후 종합해야한다고 명시한다.

   ```python
   각 전문 에이전트에게 정보를 받은후, 이를 종합해서 완벽한 결혼식을 추천해줘야한다.
   ```

6. 옵션으로 추가질문을 할지 않할지 정한다.

   ```python
   너는 추가적인 질문을 할 수 없다.
   or
   너는 추가질문을 할 수 있다.
   ```

<br>

### Sub Agent

서브 에이전트의 시스템 프롬프트를 작성할때는 넣으면 좋은 프롬프트들이 있다.

1. 에이전트가 어떤 전문가인지 명시한다. ex) 너는 여행사 직원이다.
2. 에이전트 목적을 작성한다. ex) 항공편을 검색해야한다.
3. 추가질문을 하지 못하게 한다. ex) 너는 추가질문을 할 수 없다.
4. 다음 기준에 맞는 최고의 옵션을 찾아야한다고 말한다.

   ```python
   너는 다음기준에 따라 가장 좋은 항공편 옵션을 반드시 찾아야한다.
   - 가격(가장 낮음, 이코노미 클래스)
   - 비행시간(가장 짧음)
   - 날짜(너가 판단하기에 이 장소에서 결혼식을 올리기 가장 좋은 시기)
   ```

5. 반복적으로 진행할수 있다고 말해야한다.

   ```python
   너는 가장 좋은 옵션을 찾기위해서 여러번 검색을 반복적으로 진행해야할수도있다.
   너에게는 추가적인 정보는 없다.
   ```

<br>

### Sub Agent Tool

서브 에이전트들을 Tool로 등록할때, Description 작성에 유의해야한다.

- 에이전트 이름을 명시하며, 간단하게 에이전트의 목적을 작성한다.
- ex)
  - venue agent는 결혼식에 가장 적합한 장소를 추천해준다.
  - flight agent는 장소에 알맞은 결혼 항공편을 검색한다.
  - music agent는 결혼식 분위기에 알맞은 노래를 추천해준다.

<br>

참고

- https://academy.langchain.com/courses/take/foundation-introduction-to-langchain-python/lessons/71234863-lesson-3-multi-agent-systems
