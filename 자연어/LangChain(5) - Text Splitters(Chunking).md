- [LangChain(5) - Text Splitters(Chunking)](#langchain5---text-splitterschunking)
  - [1. 텍스트 구조 기반 분할(Text structure-based)](#1-텍스트-구조-기반-분할text-structure-based)
  - [2. 길이 기반 분할(Length-based)](#2-길이-기반-분할length-based)
  - [3. 문서 구조 기반 분할(Document structure-based)](#3-문서-구조-기반-분할document-structure-based)

<br>

# LangChain(5) - Text Splitters(Chunking)

<br>

텍스트 스플리터는 큰 문서를 작은 조각으로 나눠서, 각 조각이 개별적으로 검색 가능하도록 만들고 모델의 컨텍스트 윈도우 제한안에 들어오도록 한다.

<br>

문서를 나누는 것을 청킹이라 부르며, 나눠진 문서는 청크라고 부른다.

청킹 방법은 여러가지 있다.

1. 텍스트 구조 기반 분할(Text structure-based)
2. 길이 기반 분할(Length-based)
3. 문서 구조 기반 분할(Document structure-based)

<br>

## 1. 텍스트 구조 기반 분할(Text structure-based)

LangChain의 패키지중 “`RecursiveCharacterTextSplitter`” 스플리터가 이 방식으로 동작한다.

기본적으로 청킹은 “`RecursiveCharacterTextSplitter`” 부터 사용하는 것이 좋다.

- 문맥을 최대한 유지
- 청크 크기를 관리하는 균형 잡힌 방식으로 동작한다.

<br>

기본 설정 만으로도 잘 동작한다.

<mark>**특정 태스크에 맞게 성능을 미세조정**</mark> 해야할때만, <mark>**설정을 변경**</mark>하는 것을 LangChain에서 권장한다.

<br>

<mark>**동작원리**</mark>를 알아보자.

문서의 텍스트는 크게 <mark>**4가지 구조**</mark>로 나눌수있다.

1. 문단: `“\n\n”`, 문단 경계
2. 문장: `“\n”`, 줄바꿈 경계
3. 단어: `“ “`, 단어 사이의 경계
4. 문자: `“”`, 문자 단위

<br>

1. <mark>**문단 나누기**</mark>

   문서 전체를 문단 단위인 `“\n\n”` 으로 나누어서 <mark>**문단 리스트를 만든다.**</mark>

2. <mark>**문단 합치기**</mark>

   중요한 포인트는 나눈 결과를 <mark>**그대로 청크로 쓰지 않는다.**</mark>

   <mark>**다시 이어 붙여서 최대한 정해둔 청크 사이즈에 가깝게 맞춘다.**</mark>

   문단1 + 문단2 + 문단3 … 이렇게 더해보다 <mark>**청크 사이즈를 넘기기 직전에 끊어서 하나의 청크를 만든다.**</mark>

3. <mark>**재귀 분할**</mark>

   만약 어떤 <mark>**문단 하나가 청크 사이즈보다 길수도있다.**</mark>

   이런 경우에 문단 <mark>**다음 단계인 문장단위**</mark> `“\n”`로 <mark>**다시 나눈다.**</mark>

   그래도 줄이 너무 길면, 단어단위(” ”), 더 작으면, 문자 단위(””)까지 내려가며 나눈다.

<br>

여기서 생기는 문제가 존재한다.

- 문서를 나눈뒤 청크를 만들때, <mark>**경계부분에서 정보가 끊길수있다.**</mark>
- 이전 청크의 마지막 부분과 다음 청크의 첫부분이 같은 정보이지만, 나뉘어져 <mark>**서로 다른 청크로 존재**</mark>한다.

<br>

<mark>**청크 오버랩(chunk overlap)**</mark>

청크 오버랩 방식을 통해 이런부분을 보완할 수 있다.

<mark>**이전 청크 마지막 부분과 다음 청크 마지막 부분을 두개의 청크가 모두 포함하게 한다.**</mark>

즉, 중복된 텍스트 조각을 가지게 한다는 것이다.

<br>

이렇게 겹치면, 청크 경계에서 갈려도 <mark>**앞, 뒤 청크 둘중 하나에 문맥이 남을확률**</mark>이 높아진다.

<br>

장점

- 문단, 문장 같은 <mark>**자연스러운 경계**</mark>를 우선으로 분할하여, <mark>**의미가 잘 유지**</mark>된다.
- <mark>**문맥이 덩어리로 보존**</mark>이 잘되는 편이라, <mark>**RAG 검색 결과의 품질이 안정적**</mark>인 경우가 많다.
- 긴 문단만 더 잘게 쪼개지는 식이라, <mark>**불필요하게 잘게 쪼개지는 것을 줄일 수 있다.**</mark>

단점

- 문서가 <mark>**문단이나 문장이 잘 구성되지 않은 경우**</mark>(데이터가 안좋음), 기대한 단위로 잘 안나뉠 수 있다.
- <mark>**모델 컨텍스트에 딱 맞추기 어려울**</mark>수있다.
- 경계 규칙과 <mark>**오버랩 설정**</mark>이 맞지 않으면, <mark>**청크간 중복이 과해져 문장 흐름이 어색**</mark>해진다.

<br>

[더 자세한 정보 링크](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)

<br>

## 2. 길이 기반 분할(Length-based)

가장 직관적인 방법은 문서를 <mark>**청크 길이 기준으로 분할**</mark>하는 것이다.

<br>

길이를 분할하는 기준은 2가지가 있다.

1. 토큰 기반 분할
   - <mark>**토큰수를 기준으로 분할**</mark>하며, 언어 모델을 사용할떄 유용하다.
2. 문자 기반 분할
   - <mark>**문자수를 기준으로 분할**</mark>하며, 텍스트 유형에 상관없이 일관성을 유지하기 쉽다.

<br>

장점

- <mark>**구현이 간단**</mark>하다.
- 청크 크기가 일정해서 <mark>**임베딩, 검색, 비용 관리가 쉽다.**</mark>
- 토큰 기반으로 하면, <mark>**다양한 모델 컨텍스트 한도에 정확하게 맞기 쉽다.**</mark>

단점

- 문장이나 문단 중간에 잘리는 일이 많아, <mark>**의미가 일관성이 떨어진다.**</mark>
- 중요한 <mark>**정의나 결론이 경계에서 분할되어 문맥 손실**</mark>이 생길수있다.
- 이를 보완하기 위해 <mark>**overlap을 키우면 중복 저장과 저장 비용이 증가**</mark>한다.

<br>

## 3. 문서 구조 기반 분할(Document structure-based)

HTML, Markdown, JSON 같이 문서 종류가 다른 경우, 문서 종류를 기준으로 분할할 수 있다.

문서 종류를 기반으로 분할하면, 의미적으로 연관된 텍스트가 묶인다.

<br>

<mark>**LangChain 문서에 각 문서 스플리터에 대해 잘 정리되어 있다.**</mark>

- [Markdown 분할 (Split Markdown)](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)
- [JSON 분할 (Split JSON)](https://docs.langchain.com/oss/python/integrations/splitters/recursive_json_splitter)
- [코드 분할 (Split code)](https://docs.langchain.com/oss/python/integrations/splitters/code_splitter)
- [HTML 분할 (Split HTML)](https://docs.langchain.com/oss/python/integrations/splitters/split_html)

<br>

장점

- 문서의 구조를 <mark>**원형 그대로 보존**</mark>된다.
- <mark>**메타데이터**</mark>를 붙이기 쉬워서 <mark>**검색 정확도**</mark>를 올리기 좋다.
- <mark>**코드, 표, 목록 등 특정 형식**</mark>이 있는 문서에서 <mark>**원형이 보존돼 효과적**</mark>이다.

단점

- 문서 형식 파싱이 필요해서 <mark>**전처리 복잡도**</mark> 올라간다
- 잘못된 문서에서는 <mark>**품질이 저하에 크게 영향**</mark>을 받는다.
- 문서가 지나치게 길면, 결국 <mark>**추가적인 길이기반, 재귀 분할과 결합**</mark>해야한다.

<br>

참고

- https://docs.langchain.com/oss/python/integrations/splitters
- https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
