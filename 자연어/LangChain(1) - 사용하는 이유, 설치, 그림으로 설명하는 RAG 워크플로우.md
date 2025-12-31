- [LangChain(1) - 사용하는 이유, 설치, 그림으로 설명하는 RAG 워크플로우](#langchain1---사용하는-이유-설치-그림으로-설명하는-rag-워크플로우)
  - [LangChain 이란?](#langchain-이란)
  - [왜 사용하는걸까?](#왜-사용하는걸까)
  - [설치](#설치)
  - [워크플로우](#워크플로우)
    - [RAG 단계](#rag-단계)

# LangChain(1) - 사용하는 이유, 설치, 그림으로 설명하는 RAG 워크플로우

<br>

## LangChain 이란?

LangChain은<span style='background-color: #fff5b1'>**RAG나 Agent를 쉽게**</span> 만들 수 있게 도와주는 오픈소스 프레임워크이다.

보통 RAG 서비스를 만들려면 모델 연결, 프롬프트 관리, 도구 호출, 검색, 상태 관리 등등 전부 직접 만들어야한다.

LangChain은 이런것들을 미리 만들어진 구조로 제공한다.

<br>

## 왜 사용하는걸까?

LangChain은 OpenAI, Anthropic, Google 같은 다양한 LLM과

검색기, 벡터DB, 파일 로더, API 같은<span style='background-color: #fff5b1'>**여러 도구를**</span>

**같은 방식으로 사용할 수 있게 통합**했다.

<br>

각각의 기업, 라이브러리들을 구현하기 위한 코드들이 제각각 다르다.

이것들을 모두 같은 코드로<span style='background-color: #fff5b1'>**추상화해서 통합해 제공**</span>한다.

<br>

그래서

- 모델을 바꾸더라도<span style='background-color: #fff5b1'>**코드 구조는 거의 그대로 유지**</span>
- 새로운 도구가 나와도<span style='background-color: #fff5b1'>**쉽게 교체하거나 추가 가능**</span>

이런 장점은 요즘 굉장히 빠르게 발전해<span style='background-color: #fff5b1'>**변화하는 AI 생태계에 적응**</span>할 수 있게한다.

<br>

## 설치

랭체인 패키지 설치

```bash
pip install -U langchain
# Requires Python 3.10+
```

<br>

랭체인은 여러 회사 LLM과 각종 도구(벡터 DB, 검색, 로더 등)를 바로 붙여 쓸수있게 패키지를 제공한다.

이 패키지들는 랭체인 본체에 다 들어 있지 않고,<span style='background-color: #fff5b1'>**회사나 기능별로 분리된 패키지를 설치**</span>해야한다.

<br>

예를들어

- LLM: OpenAI, Anthropic 같은 대형 언어 모델
- Chat models: 대화형 모델들
- Retrievers: 검색용 컴포넌트(문서 검색, RAG용)
- Vector stores: FAISS, Chroma 같은 벡터 데이터베이스
- Document loaders: PDF, 웹페이지, CSV 등을 불러오는 도구
- 그 외 각종 툴과 서비스들

<br>

아래는 OpenAI와 Anthropic 패키지 설치방법이다.

```bash
# Installing the OpenAI integration
pip install -U langchain-openai

# Installing the Anthropic integration
pip install -U langchain-anthropic
```

<br>

이런식으로 다른 도구들을 랭체인 코드로 통합할수있는 패키지들이 존재한다.

다양한 패키지들은 [**LangChain integrations packages Docs 링크**](https://docs.langchain.com/oss/python/integrations/providers/overview)를 통해 들어가면 볼 수 있다.

<br>

## 워크플로우

![image.png](<../images//LangChain(1)/1.png>)

<br>

### RAG 단계

RAG는 크게 4가지 단계를 거치게 된다.

입력 프로세싱 → 임베딩 & 스토리지 → 검색 → 생성

<br>

입력 프로세싱(Input processing)

1. Text input: 문서 텍스트
2. Document Loader: 문서를 불러오는 로더
3. Text Splitter: 로드된 문서를 규칙에 맞게 분할(청킹, Chunking)
4. Document: 분할된 문서(청크, Chunk)

<br>

임베딩 & 스토리지(Embeeding & Storage)

1. Embedding models: 문서를 임베딩할 모델
2. Vectors: 임베딩된 문서 벡터
3. Vector Store: 임베딩된 문서가 저장되는 스토어

<br>

검색(Retrieval)

1. User Query: 사용자 입력
2. Embedding Models: 사용자 입력을 임베딩할 모델
3. Query vector: 사용자 입력이 임베딩된 벡터
4. Retrievers: 사용자 입력(쿼리벡터)과 관련된 문서를 찾는 검색기
5. Vector stores: 문서가 저장된 벡터스토어에서 사용자 입력과 관련된 문서 검색
6. Relevant context: 사용자 입력과 관련된 문서

<br>

생성(Generation)

1. Chat models: 사영자 입력과 관련된 문서를 바탕으로 응답할 LLM 모델
2. AI response: LLM이 모델이 생성한 응답

<br>

**정리**

1.<span style='background-color: #fff5b1'>**Input processing**</span> – 원시 데이터를 구조화된 문서로 바꾸는 단계 2.<span style='background-color: #fff5b1'>**Embedding & storage**</span> – 텍스트를 벡터로 바꿔 저장하는 단계 3.<span style='background-color: #fff5b1'>**Retrieval**</span> – 사용자 질문과 관련된 정보를 찾는 단계 4.<span style='background-color: #fff5b1'>**Generation**</span> – AI가 답변을 생성하는 단계

<br>

랭체인은 이런 단계를 컴포넌트라고 부르는 모듈로 나누어 코드를 제공한다.

| Category                                                                                   | Purpose               | Key Components                      | Use Cases                     |
| ------------------------------------------------------------------------------------------ | --------------------- | ----------------------------------- | ----------------------------- |
| [Document Processing](https://docs.langchain.com/oss/python/integrations/document_loaders) | 데이터 수집 및 전처리 | Loaders, Splitters, Transformers    | PDF 처리, 웹 스크래핑         |
| [Models](https://docs.langchain.com/oss/python/langchain/models)                           | AI 추론 및 생성       | Chat Models, LLMs, Embedding Models | 텍스트 생성, 추론, 의미 이해  |
| [Vector Stores](https://docs.langchain.com/oss/python/integrations/vectorstores)           | 의미 기반 검색 저장소 | Chroma, Pinecone, FAISS             | 유사도 검색, 임베딩 저장      |
| [Retrievers](https://docs.langchain.com/oss/python/integrations/retrievers)                | 정보 검색             | Vector Retrievers, Web Retrievers   | RAG, 지식 베이스 검색         |
| [Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)                | 맥락 유지             | Message History, Custom State       | 대화 유지, 상태 기반 상호작용 |
| [Tools](https://docs.langchain.com/oss/python/langchain/tools)                             | 외부 기능 사용        | APIs, Databases 등                  | 웹 검색, 데이터 조회, 계산    |
| [Agents](https://docs.langchain.com/oss/python/langchain/agents)                           | 흐름 제어 및 추론     | ReAct Agents, Tool Calling Agents   | 비결정적 워크플로우, 의사결정 |

<br>

다음 글에서 위의 모듈들을 더 자세히 알아보겠다.

<br>

참고

- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langchain/component-architecture
