# LangChain(9) - Models (invoke, stream, batch, schema, reasoning, Ollama)

LLM은 텍스트를 해석하고 생성하는 AI 도구이다

텍스트 생성 외에도 <mark>**다른 작업도 LangChain에서 추상화해 지원**</mark>한다.

- <mark>**Tool calling**</mark>
  데이터베이스 질의나 API 호출 같은 <mark>**외부 도구**</mark>를 호출
- <mark>**Structured output**</mark>
  모델의 응답이 미리 <mark>**정의된 형식**</mark>을 따르게 함
- <mark>**Multimodality**</mark>
  텍스트 <mark>**이외의 데이터**</mark>(이미지, 오디오, 비디오)를 처리하고 반환
- <mark>**Reasoning**</mark>
  여러 단계에 거쳐 <mark>**추론해 결론**</mark>에 도달

<br>

<mark>**모델은 에이전트의 추론 엔진**</mark>이다.

에이전트의 의사결정 과정에서

- 어떤 도구를 호출할지,
- 결과를 어떻게 해석할지,
- 언제 최종 답변을 낼지를 모델이 주도한다.

<br>

모델마다 강점이 달라, <mark>**모델에 따라 에이전트의 신뢰성과 성능**</mark>에 직접적인 영향을 준다.

- 어떤 모델은 복잡한 지시사항을 더 잘 따르고
- 어떤 모델은 구조화된 추론에 강하고
- 어떤 모델은 더 큰 컨텍스트 윈도우를 지원해, 한번에 많은 정보를 처리하는데 유리하다.

<br>

LangChain은 다양한 <mark>**Provider의 모델 인터페이스를 통합**</mark>했다.

- <mark>**여러 모델을 쉽게, 빠르게 실험하며,**</mark>
- 빠르게 교체 가능해,
- <mark>**코드 복잡성이 매우 낮아진다.**</mark>

<br>

LangChain의 모델은 두 가지 방식으로 사용할 수 있다.

- 에이전트와 사용
  에이전트를 생성할때 모델을 동적으로 지정가능
- 단독 사용
  에이전트 루프 없이 모델을 직접 호출해 텍스트 생성, 분류, 추출 등 작업을 수행

<br>

두 가지방식 모두 같은 모델 인터페이스를 제공한다.

- 처음에는 단순하게 <mark>**모델을 단독으로 사용**</mark>하고
- 필요해지면, 더 <mark>**복잡한 에이전트 기반으로 확장**</mark>할 수 있도록 설계한것이다.

<br>

## 모델 생성(Initialize a model)

모든 Provider는 같은 인터페이스로 모델을 시작한다.

`init_chat_model(model name)`

```python
import os
from langchain.chat_models import init_chat_model

# OpenAI
os.environ["OPENAI_API_KEY"] = "sk-..."
model = init_chat_model("gpt-4.1")

# Anthropic
os.environ["ANTHROPIC_API_KEY"] = "sk-..."
model = init_chat_model("claude-sonnet-4-5-20250929")

# HuggingFace
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."
model = init_chat_model(
    "microsoft/Phi-3-mini-4k-instruct",
    model_provider="huggingface",
)

response = model.invoke("Why do parrots talk?")
```

<br>

## 파라미터(Parameters)

모델을 생성할때(`init_chat_model`), 모델 환경을 파라미터로 받아 설정할 수 있다.

Provider에 따라 약간씩 달라지지만, 표준적으로는 아래와 같다.

<br>

`model`

- string, required
- 사용하려는 Provider의 <mark>**모델 이름**</mark>이다.
- “:” 형식을 사용해서 모델과 Provider을 하나의 인자로 넣을 수 있다.
  예를 들어, ‘openai:o1’

<br>

`api_key`

- string
- Provider API key이다.

<br>

`temperature`

- number
- 값이 높을수록 더 창의적(변동성 있는) 응답이 나오고, 값이 낮을수록 더 결정론적(일관적인) 응답이 나오게 된다.

<br>

`max_tokens`

- number
- 응답에서 생성할 <mark>**토큰 총량을 제한**</mark>한다. 모델의 <mark>**출력 길이를 조절**</mark>한다.

<br>

`timeout`

- number
- 모델로부터 <mark>**응답을 기다릴 최대 시간(초)**</mark>이다. 이 시간을 넘기면 요청을 취소한다.

<br>

`max_retries`

- number
- 레이트 리밋 같은 이슈로 요청이 실패했을 때, 시스템이 <mark>**재전송을 시도할 최대 횟수**</mark>이다.

```python
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)
```

> Provider에 따라 모델 별로 추가 파라미터들이 있을 수 있다.

<br>

## 호출(Invocation)

모델이 응답을 생성할때, <mark>**호출 방법은 세가지**</mark>이다.

<br>

### 1. Invoke

가장 쉬운 방식은 `Invoke()`를 사용하면 된다.

입력으로 단일 메시지, 메시지 리스트를 전달한다.

```python
response = model.invoke("Why do parrots have colorful feathers?")
print(response)
```

<br>

<mark>**대화 히스토리를 표현**</mark>하려면, <mark>**메시지 리스트를 전달**</mark>하면된다.

딕셔너리 포맷

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")

```

<br>

메시지 클래스

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")

```

> 만약 `invoke()` 반환 타입이 문자열이면, 모델이 레거시 되었는지, text-completion 인지 확인해야한다.

<br>

### 2. Stream

대부분의 모델은 응답을 생성할때, 스트리밍을 할 수 있다.

긴 출력일수록 스트리밍 기능은 <mark>**사용자 경험을 크게 개선**</mark>한다.

<br>

`stream()`을 호출하면, 모델이 생성한 출력 조각(chunk)을 보여주는 <mark>**이터레이터를 반환**</mark>한다.

따라서 반복문을 돌면, 각 조각을 실시간으로 처리할 수 있다.

```python
# Stream tool calls, reasoning, and other content
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)

```

`invoke()`는 모델이 응답생성을 끝내면, 단일 AIMessage를 반환한다.

반면에 `stream()`은 <mark>**여러개의 AIMessageChunk 객체를 반환**</mark>한다.

중요한 점은, 스트림에서 나온 청크들이 <mark>**전체메시지로 다시 합쳐질수있게 설계**</mark>되었다는 점이다.

```python
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]

```

이렇게 만들어진 결과 메시지는 invoke()로 생성한 메시지랑 동일하다.

<mark>**메시지 히스토리에 합쳐**</mark>, 다시 모델에 대화 컨텍스트로 전달할 수 있게된다.

> 만약, 전체출력을 메모리에 모두 쌓은뒤 처리해야하는 애플리케이션이라면, 스트리밍 사용을 하지 않는것이 좋다.

<br>

설계의도

- UX 개선(스트리밍)과 내부처리의 일관성(청크 → 최종 메시지 결합)을 좋게했다.
- 단순히 텍스트를 조각내는것 뿐만아니라, 나중에 대화 히스토리에도 사용할 수 있게 만들어준 구조이다.

<br>

### 3. Batch

서로 독립적인 <mark>**요청을 묶어서(batch) 모델에 처리**</mark>시켜, <mark>**병렬 처리**</mark>로 성능을 올리고 비용을 줄인다.

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

<br>

기본적으로 `batch()`는 전체가 끝난 다음 최종결과를 반환한다.

만약 각 입력에 대한 출력이 생성되는 즉시 받아보고 싶으면, `batch_as_completed()`로 <mark>**결과를 스트리밍**</mark>할 수 있다.

```python
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)
```

`batch_as_completed()`는 결과가 입력 순서대로 오지 않을수있다.

결과를 반환했을때, <mark>**인덱스에 있으므로 이것을 이용해서 순서대로 재구성**</mark>하면 된다.

<br>

많은 입력을 처리할때, 한번에 처리하는 <mark>**호출수를 제한**</mark>할수도 있다.

```python
model.batch(
    list_of_inputs,
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls
    }
)
```

<br>

batch() 설계의도

- 작은 예제에서는 invoke()로 시작하고, 트래픽과 비용이 생기면 batch로 확장할 수 있게 만들었다.

<br>

## 구조화 출력(Strutured output)

모델에게 <mark>**응답을 특정 스키마에 맞춰**</mark>서 만들라고 할 수 있다.

이렇게하면 출력이 쉽게 파싱되고, 이후 <mark>**후처리단계에서 안정적**</mark>으로 사용할 수 있다.

LangChain에서는 구조화 출력을 강제 학위해서 여러 스키마 타입과 여러 메서드를 지원한다.

<br>

### Pydantic 모델

<mark>**Pydantic 모델**</mark>은 필드검증(validation), 설명(description), 중첩 구조(nested structures) 같은 기능이 풍부해서 <mark>**가장 강력한 모델**</mark>이다.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)

```

<br>

원본 반환값 출력

파싱된 구조와 함께 <mark>**원본 AIMessage 객체를 같이 반환**</mark>하는 것이 유용한 경우가 있다.

예를 들어 토큰 수 같은 메타데이터를 확인하려면 raw 메시지가 필요하다.

이때 `with_structured_output` 호출 시 `include_raw=True`로 설정한다.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("Provide details about the movie Inception")
response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }

```

<br>

중첩 스키마

<mark>**스키마를 중첩**</mark>할 수 있다.

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)

```

“현실 데이터는 보통 중첩된 객체 형태이므로, 단일 평면 JSON만으로는 표현력이 부족하다”는 점을 반영하는 것이다.

그래서 리스트, 객체의 리스트, 옵션 필드(None 허용) 같은 실전 데이터 모델링을 그대로 스키마로 표현할 수 있게 해 둔다.

<br>

## 추론(Reasoning)

결론에 도달할때 여러 단계에 걸친 추론(multi-step reasoning)을 수행할 수 있다.

복잡한 문제를 더 작고 다루기 <mark>**쉬운 단계들로 쪼개서 해결**</mark>하는 것이다.

<br>

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)

```

모델에 따라, 추론에 얼마나 많은 노력을 들일지(reasoning effort level)를 지정할 수 있는 경우도 있다.

추론을 완전히 끌수도 있다.

- 범주형 티어(tier)로 제공되는 경우: 예를 들어 'low', 'high' 같은 단계로 추론 강도를 지정한다.
- 정수 토큰 예산(token budget)으로 제공되는 경우: 추론에 사용할 수 있는 토큰 수를 숫자로 제한한다.

<br>

설계의도

1. 관측 가능성(observability)이다.

   단순히 답만 받는 것이 아니라 “어떤 중간 과정으로 답이 나왔는지”를 일부라도 확인 가능하게 해서, <mark>**디버깅과 품질 개선을 쉽게 하려는 의도**</mark>이다.

2. 비용, 지연시간, 정확도의 트레이드오프 제어이다.

   추론을 많이 하게 하면 정답률이 올라갈 수 있지만 비용과 지연이 커질 수 있고, 반대로 추론을 줄이면 빠르고 저렴하지만 실수가 늘 수 있다.

   그래서 <mark>**모델별로 추론 수준을 조절할 수 있게 설계**</mark>한다.

<br>

모델을 제공하는 Provider은 [이곳 Chat models Docs](https://docs.langchain.com/oss/python/integrations/chat)에서 더 자세히 볼 수 있다.

그중에 로컬에서 모델을 사용하는 Ollama에 대해 알아보자.

<br>

## ChatOllama

Ollama는 gpt-oss 같은 <mark>**오픈소스 대형 언어 모델(LLM)을 로컬에서 실행**</mark>할 수 있게 해주는 도구이다.

Ollama는 모델 가중치, 설정, 데이터 등을 Modelfile로 <mark>**정의된 하나의 패키지로 묶어서 배포**</mark>한다.

이 과정에서 GPU 사용을 포함한 <mark>**설치 및 설정 세부사항을 최적화**</mark>한다.

<br>

모델 특징

| [<mark>**Tool calling**</mark>](https://docs.langchain.com/oss/python/langchain/tools) | [<mark>**Structured output**</mark>](https://docs.langchain.com/oss/python/langchain/structured-output) | <mark>**JSON mode**</mark> | [<mark>**Image input**</mark>](https://docs.langchain.com/oss/python/langchain/messages#multimodal) | <mark>**Audio input**</mark> | <mark>**Video input**</mark> | [<mark>**Token-level streaming**</mark>](https://docs.langchain.com/oss/python/langchain/streaming) | <mark>**Native async**</mark> | [<mark>**Token usage**</mark>](https://docs.langchain.com/oss/python/langchain/models#token-usage) | [<mark>**Logprobs**</mark>](https://docs.langchain.com/oss/python/langchain/models#log-probabilities) |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| ✅                                                                                     | ✅                                                                                                      | ✅                         | ✅                                                                                                  | ❌                           | ❌                           | ✅                                                                                                  | ✅                            | ❌                                                                                                 | ❌                                                                                                    |

<mark>**랭체인에서 바로 사용하기 전**</mark>에,

<mark>**로컬에 Ollama를 사용하기 위해 먼저 설치해야한다.**</mark>

```bash
brew install ollama
```

<br>

시작

```bash
brew services start ollama
```

<br>

옵션1) LLM 모델 불러오기

```bash
# 예 ollama pull llama3
ollama pull <name-of-model>

# 원하는 버전의 모델을 구체적으로 사용할수있다.
ollama pull vicuna:13b-v1.5-16k-q4_0

# 모델의 종류를 볼수있다.
ollama list
```

<br>

옵션2) LLM 모델 바로 사용하기

```bash
# 모델을 불러오고 바로 사용한다.
ollama run <name-of-model>

```

모델 저장 경로

- MAC: <mark>**`~/.ollama/models`**</mark>
- Linux: <mark>**`/usr/share/ollama/.ollama/models`**</mark>

<br>

랭체인에서 Ollama 패키지 설치

```bash
pip install -qU langchain-ollama
```

<br>

### 모델 생성

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)
```

<br>

### 호출

```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
```

<br>

### 도구 호출

Ollama의 tool calling은 <mark>**OpenAI 호환 웹 서버 스펙**</mark>을 따른다고 설명한다.

LangChain에서는 `BaseChatModel.bind_tools()`로 <mark>**도구를 바인딩**</mark>하는 표준 방식으로 사용할 수 있다.

<br>

도구 호출을 지원하는 모델을 선택해야한다.

```bash
ollama pull gpt-oss:20b
```

<br>

```python
from typing import List

from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True

llm = ChatOllama(
    model="gpt-oss:20b",
    validate_model_on_init=True,
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)

if isinstance(result, AIMessage) and result.tool_calls:
    print(result.tool_calls)

```

llm이 생성한 <mark>**result에는 도구를 호출해야하는지에 대한 정보**</mark>가 포함되어있다.

`tool_calls` 리스트에 <mark>**name, args, id, type 같은 정보가 들어간 형태**</mark>이다.

<br>

### 멀티모달(Multi-modal)

Ollama는 gemma3 같은 멀티모달 LLM을 제한적으로 지원한다고 안내한다.

멀티모달을 쓰려면 Ollama 버전을 최신으로 업데이트하라고 강조한다.

이미지 처리를 위해 pillow 설치 한다.

```bash
pip install pillow
```

```python
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="bakllava", temperature=0)

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

from langchain_core.output_parsers import StrOutputParser

chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(
    {"text": "What is the Dollar-based gross retention rate?", "image": image_b64}
)

print(query_chain)

```

PIL <mark>**이미지를 base64로 바꾸는 전처리**</mark>를 한다.

이후 ChatOllama(model="bakllava") 같은 멀티모달 모델을 생성한다.

prompt_func 함수를 사용해 <mark>**HumanMessage의 content를 “이미지 파트 + 텍스트 파트”로 구성**</mark>한 뒤 체인에 연결한다.

<br>

### 추론 모델

Granite 3.2의 thinking 기능을 쓰려면 `role="control"`이고 `content="thinking"`인 메시지를 추가한다.

control은 표준 역할이 아니므로 `ChatMessage` 객체를 사용한다.

```python
from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="granite3.2:8b")

messages = [
    ChatMessage(role="control", content="thinking"),
    HumanMessage("What is 3^3?"),
]

response = llm.invoke(messages)
print(response.content)

```

<br>

Reference

- https://docs.langchain.com/oss/python/langchain/models
- https://docs.langchain.com/oss/python/integrations/chat
- [https://reference.langchain.com/python/integrations/langchain_ollama](https://reference.langchain.com/python/integrations/langchain_ollama/?_gl=1*b1iu06*_gcl_au*NjkzNTc0Mjg0LjE3NjcwNjIyMzc.*_ga*MjQ5MDY5OTYwLjE3NjcwNjIyMzc.*_ga_47WX3HKKY2*czE3Njc2MDU5NjUkbzE4JGcxJHQxNzY3NjA4MDQzJGo1NyRsMCRoMA)
- https://ollama.com/search
