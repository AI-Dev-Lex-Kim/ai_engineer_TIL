- [Agent(4) - MCP (MultiServerMCPClient, FastMCP)](#agent4---mcp-multiservermcpclient-fastmcp)
  - [MCP 중요한 개념 3가지](#mcp-중요한-개념-3가지)
  - [헷갈렸던점](#헷갈렸던점)
  - [MCP 서버 생성](#mcp-서버-생성)
  - [MCP 서버 사용](#mcp-서버-사용)
  - [Session](#session)
  - [주의해야할점](#주의해야할점)

# Agent(4) - MCP (MultiServerMCPClient, FastMCP)

Model Context Protocol(MCP)는 애플리케이션이 LLM에게 도구(tools)와 컨텍스트를 제공하는 방식을 <mark>**표준화한 오픈 프로토콜**</mark>이다.

<br>

기존에는 LangChain Tool, OpenAI Function Calling 등 <mark>**프레임워크별, 플랫폼별 툴 사용방식이 제각각**</mark> 이였다.

MCP는 이를 통합된 <mark>**인터페이스로 표준화**</mark>한것이다.

<br>

LangChain 라이브러리 에서는 `langchain-mcp-adpaters` 라이브러리를 사용해서 MCP 도구들을 사용할 수 있다.

<br>

## MCP 중요한 개념 3가지

1. <mark>**MCP Server**</mark>

   <mark>**실제 기능을 제공하는 쪽**</mark>이다.
   - 계산기 서버
   - 날씨 조회 서버
   - DB 조회 서버
   - 사내 API 서버

   <br>

   <mark>**예시를 들어 쉽게 설명**</mark>해보자.

   `mcp_server.py` 파일안에 tool, prompt, resource 등등 코드들을 포함하고 있다.

   `agent.py` 파일안에는 agent를 생성하는 코드를 작성하고 있다.

   <br>

   `agent.py`는 `mcp_server.py`에 있는 툴을 가져와 사용하므로 `mcp_server.py` 파일을 <mark>**“서버”**</mark> 라고 부르는것이다.

   코드들을 MCP 프로토콜 인터페이스를 통일시켰으니 <mark>**“MCP 서버”**</mark>라고 부른다.

   <br>

2. <mark>**MCP Client**</mark>

   LLM / Agent 쪽에서 <mark>**서버를 호출하는 역활**</mark>이다.

   MultiServerMCPClient는
   - 여러 MCP 서버를 동시에 등록할 수 있게 한다.
   - 하나의 Agent에게 도구를 묶어서 제공한다.

   <br>

   ```python
   ┌──────────┐
   │   LLM    │
   └────┬─────┘
        │ tool call
   ┌────▼────────────────────────┐
   │ MultiServerMCPClient         │
   │  ├─ math MCP server (stdio)  │
   │  └─ weather MCP server (http)│
   └────────┬────────────────────┘
            │
      각 호출마다 세션 생성/종료
   ```

   <mark>**stido**</mark>
   - MCP 커뮤니티에서 Python 파일 코드를 가져와서
   - `python mcp_server.py` 같은 <mark>**파일을 직접 실행**</mark>(`python mcp_server.py`)
   - 설정 간단, <mark>**로컬 개발**</mark>/테스트용

   <mark>**http**</mark>
   - `http://localhost:8000/mcp` 같이 내가 <mark>**로컬 서버를 운용중**</mark>이여야한다.
   - 에이전트를 생성중인 코드에서 `http://localhost:8000/mcp` 로 호출해서 도구를 불러오는것이다.
   - 회사 서버일수도, 로컬서버일 수 도있다.
   - 가장 중요한 점은 <mark>**http 주소를 호출해 실행**</mark>한다는것이다.
   - <mark>**배포 단계**</mark>에서 사용한다.

<br>

1. <mark>**Tool**</mark>

   MCP 서버가 이러한 함수가 있다고 명세하면,

   LLM은 <mark>**이 명세를 보고 도구를 호출**</mark>한다.

<br>

정리하면

한번 MCP 서버를 만들면,

- LangChain Agent
- Claude Desktop
- 다른 클라이언트
  에서도 재사용이 가능하다.

<br>

## 헷갈렸던점

MCP 서버라고 용어를 부르길래,

온라인에 올린 MCP 툴을 개개인이 서버를 운용해서 올려둔 상태인줄 알았다.

<br>

실은, <mark>**“MCP 서버 = 실행 가능한 코드”**</mark> 라고 이해하면된다.

내 로컬에서 코드를 가져와 실행하는것이다.

<br>

즉,

- API 주소를 받는것이 아니라
- 레포를 받아서
- 직접 실행하는것이다.

<br>

실제 흐름은 다음과 같이 흐른다.

```bash
git clone mcp-some-tool
cd mcp-some-tool
pip install -r requirements.txt
python server.py
```

<br>

LangChain에서는 위를 다음과 같이 작성한다.

```json
{
  "transport": "stdio",
  "command": "python",
  "args": ["server.py"]
}
```

<br>

## MCP 서버 생성

> 헷갈릴 수도 있으니 다시 한번 강조하자면(나는 헷갈렸음),
> Tool을 MCP 표준 프로토콜로 만들어서 <mark>**어떤 클라이언트여도 재사용**</mark>가능하게 하는것이다.

<br>

MCP 서버를 만들때는 <mark>**FastMCP 라이브러리**</mark>를 사용하면 빠르게 만들수있다.

```bash
pip install fastmcp
```

<br>

```python
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from requests import get

# MCP 서버를 쉽게 만들어주는 프레임워크
mcp = FastMCP("mcp_server")

tavily_client = TavilyClient()

# Tool for searching the web
@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """웹에서 정보를 검색한다."""

    results = tavily_client.search(query)

    return results

# Resources - langchain-ai repo 파일을 제공한다.
@mcp.resource("github://langchain-ai/langchain-mcp-adapters/blob/main/README.md")
def github_file():
    """
    Resource for accessing langchain-ai/langchain-mcp-adapters/README.md file

    """
    url = f"https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md"
    try:
        resp = get(url)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"

# Prompt template
@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
    You are a helpful assistant that answers user questions about LangChain, LangGraph and LangSmith.

    You can use the following tools/resources to answer user questions:
    - search_web: Search the web for information
    - github_file: Access the langchain-ai repo files

    If the user asks a question that is not related to LangChain, LangGraph or LangSmith, you should say "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

    You may try multiple tool and resource calls to answer the user's question.

    You may also ask clarifying questions to the user to better understand their question.
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

위의 MCP 서버는 기능을 3가지를 제공하고있다.

- <mark>**Tool**</mark>: LLM이 호출할 수 있는 함수(search_web)
- <mark>**Resource**</mark>: LLM이 읽을 수 있는 <mark>**데이터/문서**</mark>(github_file)
- <mark>**Prompt**</mark>: 서버가 제공하는 기본 시스템 <mark>**프롬프트 템플릿**</mark>(prompt)
- 마지막에 `mcp.run(transport=’stdio’)`로 stdio 방식 MCP 서버를 실행

<br>

```python
@mcp.resource("github://langchain-ai/langchain-mcp-adapters/blob/main/README.md")
def github_file():
    url = f"https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md"
    try:
        resp = get(url)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"
```

`@mcp.resource(”<URL>”)`

- 이 <mark>**URL로 리소스를 제공**</mark>한다는 뜻이다.
- LLM은 리소스를 읽어서 답변 근거로 쓸수있다.
- `github_file()`은 실제로 GitHub의 README 내용을 HTTP로 받아와 텍스트로 반환한다.

<br>

```python
@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
    You are a helpful assistant ...
    """
```

`@mcp.propmt`

- mcp.prompt 데코레이터가 붙으면, <mark>**프롬프트를 제공**</mark>해준다.
- 보통 에이전트가 툴을 <mark>**수행잘할 수 있게 해주는 프롬프트**</mark>를 제공한다.
- 에이전트가 이 프롬프트를 <mark>**시스템 프롬프트**</mark>로 입력받아 사용할 수 있다.
- 예를 들어
  - “search_web, github_file”을 써라
  - 관련 없으면 거절해라
  - 필요하면 여러 번 tool, resource 호출해라

이 Prompt를 사용하도록 설계해서 <mark>**에이전트의 성격을 고정**</mark>할 수 있다.

<br>

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- 이 파일을 python server.py로 실행하면
- MCP 서버가 stdio로 통신하는 모드로 뜬다.

<br>

## MCP 서버 사용

위에서 정의한 MCP 서버를 사용해보자.

<br>

설치

```bash
pip install langchain-mcp-adapters
```

<br>

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "info_langchain": {
                "transport": "stdio",
                "command": "python",
                "args": ["resources/2.1_mcp_server.py"],
            }
    },
    {
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=America/New_York"
            ]
        }
    }
)

# get tools
tools = await client.get_tools()

# get resources
resources = await client.get_resources("info_langchain")

# get prompts
prompt = await client.get_prompt("info_langchain", "prompt")
prompt = prompt[0].content

from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt=prompt
)

from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
    config=config
)
```

<br>

```python
# get tools
tools = await client.get_tools()
```

- 연결된 MCP 서버에 등록된 Tool 목록을 가져온다.
- 여기서 Tool은 `@mcp.tool()`로 등록한 함수들이다.(`search_web`)
- 반환값 `tools`는 LangChain이 바로 쓸수있는 Tool 객체 리스트 형태가 오고 에이전트에 그대로 넘길수있다.
- `client.get_tools(”<server_name>”)` <mark>**특정 서버 이름**</mark>을 넣으면 그 툴만 가져온다.

<br>

```python
# get resources
resources = await client.get_resources("info_langchain")
```

- `client.get_resources(”<server_name>”)` 특정 서버 이름을 넣지 않으면, <mark>**모든 MCP 서버의 모든 리소스**</mark>를 받게된다.

<br>

```python
# get prompts
prompt = await client.get_prompt("info_langchain", "prompt")
prompt = prompt[0].content
```

- `“info_langchain”` 서버에서 등록된 프롬프트중에서
- 이름이 `“prompt”`인 것을 가져오라는 뜻이다.

<br>

## Session

<mark>**Stateless 상태에서 동작**</mark>

MulitServerMCPClient는 기본적으로 툴을 한번 호출할 때마다

1. MCP 세션을 새로 만들고
2. 툴 실행하고
3. <mark>**세션을 닫는다.**</mark>

그래서 이전 툴 호출에서 이어지는 서버 측 상태가 있더라도 <mark>**다음 호출엔 안이어 질 수 있다.**</mark>

<br>

<mark>**Stateful 상태유지**</mark>

서버가 세션에 상태를 저장하는 타입이면, <mark>**같은 세션을 유지**</mark>해야한다.

예시

- <mark>**로그인, 인증 핸드셰이크 토큰**</mark>을 세션에 저장
- 세션별로 <mark>**대화 컨텍스트 유지**</mark>
- 세션별로 <mark>**캐시**</mark>을 들고있음

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

async def main():
    # 예시: stdio 서버(로컬 subprocess) + http 서버(로컬/원격) 같이 등록 가능
    client = MultiServerMCPClient(
        {
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["/absolute/path/to/your_mcp_server.py"],
            },
        }
    )

    # 세션을 명시적으로 열고(with 블록 동안 유지)
    async with client.session("local_server") as session:
        # 이 세션에 묶인 tools 로드
        tools = await load_mcp_tools(session)

        # 에이전트 생성
        agent = create_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools,
        )

        # 같은 세션 안에서 여러 번 tool call이 일어날 수 있게 질문을 연속으로 던짐
        res1 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "search_web로 'LangChain MCP adapters'를 검색해줘"}]}
        )
        print("\n--- Response 1 ---")
        print(res1["messages"][-1].content)

        res2 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "방금 검색 결과를 3줄로 요약해줘"}]}
        )
        print("\n--- Response 2 ---")
        print(res2["messages"][-1].content)

        res3 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "README 리소스를 읽어서 핵심 사용법만 bullet로 정리해줘"}]}
        )
        print("\n--- Response 3 ---")
        print(res3["messages"][-1].content)

    # 여기 도달하면 session은 자동 close/cleanup 됨
    print("\nSession closed")

if __name__ == "__main__":
    asyncio.run(main())

```

<br>

```python
async with client.session("server_name") as session:
```

- server_name 서버에 대해 <mark>**세션을 명시적**</mark>으로 열어둔다.
- with 블록이 끝나면 <mark>**자동으로 세션을 닫는다.**</mark>
- 리소스 누수 방지에 좋다

<br>

```python
tools = await load_mcp_tools(session)
```

- 클라이언트 전체가 아니라, 이 세션기반으로 MCP 서버가 제공하는 tool들을 로드한다.
- 이렇게 로드된 도구들은 이후 호출할때도 같은 세션을 사용하도록 묶이는 행태가 된다.
- 이후 에이전트가 tool call을 하면, 이 tool들은 방금 만든 session 컨텍스트 안에서 실행된다.

<br>

```python
# 같은 세션 안에서 여러 번 tool call이 일어날 수 있게 질문을 연속으로 던짐
res1 = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "search_web로 'LangChain MCP adapters'를 검색해줘"}]}
)
print("\n--- Response 1 ---")
print(res1["messages"][-1].content)

res2 = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "방금 검색 결과를 3줄로 요약해줘"}]}
)
print("\n--- Response 2 ---")
print(res2["messages"][-1].content)

res3 = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "README 리소스를 읽어서 핵심 사용법만 bullet로 정리해줘"}]}
)
print("\n--- Response 3 ---")
print(res3["messages"][-1].content)
```

같은 세션에서 도구를 불러와 데이터를 가져와 사용가능하다.

## 주의해야할점

MCP 서버 transport가 http 인 경우, 비동기적으로 도구와 에이전트를 호출해야한다.

<br>

비동기적으로 호출하기 위해서 async, await를 잊지말고 잘 적용해야한다.

```python
client = MultiServerMCPClient(
    {
        "kiwi-com-flight-search": {
            "transport": "streamable_http",
            "url": "https://mcp.kiwi.com",
        }
    }
)

flight_tool = await client.get_tools

flight_agent = create_agent(
    model = flight_agent_param['model'],
    system_prompt= flight_agent_param['system_prompt'],
    tools = flight_agent_param['tools']
)

@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """flight agent는 장소에 알맞은 결혼 항공편을 검색한다."""
    origin = runtime.state.get('origin', '김포')
    destination = runtime.state.get('destination', '제주도')
    query = f'{origin}에서 {destination}으로 향하는 항공편을 찾아야한다.'

    response = await flight_agent.ainvoke(
            {'messages' : [HumanMessage(content=query)]}
        )
    return response['messages'][-1].content

response = await main_agent.ainvoke(
    input = messages,
    config = config
)
```

- `clinet.get_tools` 을 호출할때 `awiat`을 사용했다.
- 멀티 에이전트를 만들때, 서브 에이전트를 tool에 넣을때도 `async`를 사용해야한다.
- 상위 에이전트(감독) 또한 실행할때 `invoke`가 아닌 `ainvoke`를 사용해야한다.

참고

- https://docs.langchain.com/oss/python/langchain/mcp
- https://reference.langchain.com/python/langchain_mcp_adapters/?h=multiservermcpclient#langchain_mcp_adapters.client.MultiServerMCPClient.get_resources
