- [LangaChain(2) - Message (SystemMessage, HumanMessage, AIMessage)](#langachain2---message-systemmessage-humanmessage-aimessage)
  - [SystemMessage](#systemmessage)
  - [HumanMessage](#humanmessage)
  - [AIMessage](#aimessage)

# LangaChain(2) - Message (SystemMessage, HumanMessage, AIMessage)

GPT-4, Claude 같은 <mark>**채팅 기반 모델**</mark>들은 단순 문자열이 아니라, <mark>**역할(role)과 content(내용), Metadata(메타데이터)를 가진 메시지들을 입력**</mark>으로 받는다.

- <mark>**Role(역할)**</mark>
  프롬프트를 전달하는 주체를 의미한다.
  예: system, user, AI
- <mark>**Content(내용)**</mark>
  메시지의 실제 내용을 의미한다.
- <mark>**Metadata(메타데이터)**</mark>
  선택적인 정보로, 응답 관련 정보, 메시지 ID, 토큰 사용량 등이 들어온다.

<br>

LangChain은 <mark>**모든 채팅기반 모델들이 공통으로 사용할수있게 표준 메시지 타입을 제공**</mark>한다.

어떤 모델을 호출하더라도 같은 형식으로 <mark>**일관성있게 동작**</mark>한다.

<br>

메시지의 역할로는 3가지가 있다.

- 시스템(System)
- 사용자(Human)
- 모델(AI)

<br>

각 역할마다 모델은 다르게 의미를 받아들이고 동작한다.

1. 시스템(System): 모델이 어떻게 동작해야하는지 지시한다.
2. 사용자(Human): 사용자의 입력이다.
3. 모델(AI): 모델이 생성한 응답을 의미한다.

<br>

LangChain은 위의 3가지 역할의 템플릿을 제공한다.

1. SystemMessage: 시스템
2. HumanMessage: 사용자
3. AIMessage: 모델

<br>

## SystemMessage

<mark>**모델의 행동**</mark>을 미리 설정해 <mark>**지시하는 메시지**</mark>이다.

- 응답의 톤을 설정하고,
- 모델의 역활을 정의하며,
- 응답에 대한 가이드라인
  을 명확히 한다.

<br>

```bash
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

<br>

## HumanMessage

<mark>**사용자의 입력 메시지**</mark>이다.

<br>

사용자의 입력 메시지를 모델에게 전달하는 방법은 2가지가있다.

1. HumanMessage
2. model.invoke()

<br>

HumanMessage 방법

```bash
response = model.invoke([
  HumanMessage("What is machine learning?")
])
```

<br>

model.invoke() 방법

```bash
# Using a string is a shortcut for a single HumanMessage
response = model.invoke("What is machine learning?")
```

<br>

## AIMessage

모델이 생성한 <mark>**응답의 결과 메시지**</mark> 이다.

모델을 호출했을때 반환되고 응답과 관련된 모든 메타데이터를 같이 담고있다.

```bash
response = model.invoke("Explain AI")

# AI가 응답한 결과를 이곳에 넣는다.
ai_msg = AIMessage(response)
```

<br>

참고

- https://docs.langchain.com/oss/python/langchain/messages
