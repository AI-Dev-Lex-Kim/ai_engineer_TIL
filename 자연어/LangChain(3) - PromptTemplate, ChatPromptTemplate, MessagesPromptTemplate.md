- [LangChain(3) - PromptTemplate, ChatPromptTemplate, MessagesPromptTemplate](#langchain3---prompttemplate-chatprompttemplate-messagesprompttemplate)
  - [PromptTemplate 이란?](#prompttemplate-이란)
  - [PromptTemplate](#prompttemplate)
  - [ChatPromptTemplate](#chatprompttemplate)
  - [MessagesPromptTemplate](#messagesprompttemplate)

# LangChain(3) - PromptTemplate, ChatPromptTemplate, MessagesPromptTemplate

## PromptTemplate 이란?

LangChain 프롬프트 템플릿은 반복되는 <mark>**프롬프트를 미리 템플릿을 만들어 재사용**</mark> 가능하게 해주는 방법이다.

- 프롬프트를 일관되게 유지
- 모델의 응답을 더 예측 가능하고 안정하게 만듬
- 프롬프트 작성 방식의 무작위성을 줄여 결과 품질 개선

<br>

LangChain이 제공하는 템플릿 종류는 3가지 유형이 있다.

- 문자열 기반의 기본 프롬프트를 만드는 `PromptTemplate`
- 여러 메시지를 포함하는 채팅 기반 프롬프트용 `ChatPromptTemplate`
- 대화 기록과 같은 동적 메시지 리스트를 삽입하는 `MessagesPlaceholder`

<br>

## PromptTemplate

```bash
from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("Share an interesting fact about {animal}.")

filled_prompt = prompt.format(animal="octopus")
print(filled_prompt)
```

문자열안에 <mark>**빈칸을 채우는 방식으로 프롬프트를 생성**</mark>한다.

`{animal}` 같이 플레이스홀더를 사용할 수 있다.

<br>

## ChatPromptTemplate

GPT-4, Claude 같은 <mark>**채팅 기반 모델**</mark>들은 단순 문자열이 아니라, <mark>**역활(role)과 content를 가진 메시지들을 입력**</mark>으로 받는다.

<br>

이런 구조를 쉽게 구성할수있도록 도와준다.

PromptTemplate 처럼 빈칸을 넣고 파라미터로 값을 전달하는 메커니즘이다.

```bash
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a patient tutor who explains things clearly."),
    ("human", "Can you explain {concept} like I'm five?")
])

formatted_messages = chat_prompt.format_messages(concept="gravity")

print(formatted_messages)
```

`{concept}` 는 `“gravity”` 값이 들어가게 된다.

<br>

이렇게 생성된 포맷 결과는 다음과 같은 구조로 만들어진다

```bash
[
    SystemMessage(content="You are a patient tutor who explains things clearly.", role="system"),
    HumanMessage(content="Can you explain gravity like I'm five?", role="user")
]
```

System, User, AI의 메시지 객체를 직접 만들필요 없이 템플릿이 쉽게 처리해준다.

PromptTemplate 보다는 <mark>**ChatPromptTemplate을 주로 사용**</mark>하면된다.

<br>

## MessagesPromptTemplate

채팅 모델을 사용할때, <mark>**이전 대화 기록을 프롬프트에 포함**</mark>시키는 경우가 많다.

`MessagesPlaceholder`는 이전 대화 기록을 직접 넣는 대신, 이곳에 넣어 이전 대화 기록이라는 것을 표시한다.

<br>

placeholder는 사용자가 입력하는 창에서도 어떤값을 넣어야하는지 텍스트로 도와주는 역활을 한다.

예를 들어 아래에서는 “무엇이든 물어보세요” 같은 부분이다.

![image.png](<../images/Langchain(3)/1.png>)

<br>

`MessagesPlaceholder`는 마찬가지로 이전 대화기록 자리라는 것을 알려주는 표시와 마찬가지다.

```bash
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful career coach."),
    MessagesPlaceholder("conversation"),
    ("human", "{current_question}")
])

conversation_history = [
    HumanMessage(content="How do I prepare for a job interview?"),
    AIMessage(content="Start by researching the company and practicing common questions.")
]

formatted_messages = chat_prompt.format_messages(
    conversation=conversation_history,
    current_question="Should I send a thank-you email afterward?"
)

print(formatted_messages)
```

`MessagesPlaceholder("conversation")` 자리가 이전 대화라는 것을 `“conversation”` 이라는 텍스트로 표시해두었다.

이곳에 그대로 `conversation_history` 가 들어간다.

<br>

출력결과

```bash
[
    SystemMessage(content="You are a helpful career coach.", role="system"),
    HumanMessage(content="How do I prepare for a job interview?", role="user"),
    AIMessage(content="Start by researching the company and practicing common questions.", role="assistant"),
    HumanMessage(content="Should I send a thank-you email afterward?", role="user")
]
```

<br>

참고

- https://mirascope.com/blog/langchain-prompt-template
- https://reference.langchain.com/python/langchain_core/prompts/#langchain_core.prompts.chat.ChatPromptTemplate
