- [RAG(Retrieval-Agumented Generation) Deep Dive with code](#ragretrieval-agumented-generation-deep-dive-with-code)
  - [Document Loader](#document-loader)
    - [웹 문서(WebBaseLoader)](#웹-문서webbaseloader)
    - [텍스트 문서(TextLoader)](#텍스트-문서textloader)
    - [디렉토리 폴더(DirectoryLoader)](#디렉토리-폴더directoryloader)
    - [CSV 문서(CSVLoader)](#csv-문서csvloader)
    - [PDF 문서](#pdf-문서)
  - [Text Chunking](#text-chunking)
    - [길이 단위 청킹(RecursiveCharacterTextSplitter)](#길이-단위-청킹recursivecharactertextsplitter)
    - [의미 고려 청킹(**SemanticChunker)**](#의미-고려-청킹semanticchunker)
  - [Embedding Model](#embedding-model)
    - [**OpenAIEmbeddings**](#openaiembeddings)
    - [HuggingFaceEmbeddings](#huggingfaceembeddings)
    - [**GoogleGenerativeAIEmbeddings**](#googlegenerativeaiembeddings)
  - [Vector Store](#vector-store)
    - [Chroma](#chroma)
    - [FAISS(Facebook AI Similarity Search)](#faissfacebook-ai-similarity-search)
  - [Retriever](#retriever)
    - [Vector Store Retriver](#vector-store-retriver)
    - [Multi Query Retriever](#multi-query-retriever)
    - [Contextual compression](#contextual-compression)
  - [Prompt Template](#prompt-template)
    - [PromptTemplate](#prompttemplate)
    - [ChatPromptTemplate](#chatprompttemplate)
    - [FewShotPromptTemplate](#fewshotprompttemplate)
  - [Generation](#generation)

# RAG(Retrieval-Agumented Generation) Deep Dive with code

RAG는 검색 증강 생성이라고 부른다. 대규모 언어 모델 LLM의 한계를 보완하기 위해 만들어진 기술이다.

LLM은 방대한 데이터로 학습하여 똑똑하지만, 몇가지 단점을 가지고있다.

<br>

1. 최신 정보 부족: 특정 시점의 데이터로만 학습했기 때문에 이후의 최신정보는 알지못한다.
2. 환각: 사실이 아닌 내용을 그럴듯하게 지어내는 경우가 있다.
3. 전문 분야 지식 부족: 법률, 의료, 특정 회사의 내부 문서 등 제한된 분야의 전문 지식은 부족하다.

<br>

RAG는 이러한 단점을 해결하기 위해 LLM이 답변을 생성하기 전에 다음과 같은 작업을 한다.

사용자가 질문을 입력하면 관련된 정보를 외부 데이터(ex:csv,pdf…)에서 검색(Retrieval)한뒤 해당 지식을 프롬프트에 삽입하여 모델이 활용해 답변하게 만든다.

<br>

즉,

1. 입력 질문
2. 외부 데이터 검색후 가져옴
3. (입력 질문 + 외부 데이터)인 증강된 프롬프트 생성(ex: 이 참고자료를 보고 답변해줘)
4. (입력 질문 + 외부 데이터)인 증강된 프롬프트 모델에 삽입
5. 최종 답변

<br>

RAG을 통한 LLM의 답변 워크플로우는 다음과 같다.

1. 문서로드
2. 청킹
   - 토큰 청킹
   - 문장 청킹
3. 임베딩 벡터 모델 생성
   - 쿼리 임베딩 벡터 생성
   - 청킹 임베딩 벡터 생성
4. 벡터 스토어 생성
   - 청킹 벡터 저장
5. 리트리버 생성
   - 벡터 스토어 리트리버
     - 코사인 유사도
     - MMR
   - 멀티 쿼리 리트리버
   - 컨텍스츄얼 컴프레서 리트리버
6. 리트리버 문서 검색
7. 포멧팅 컨텍스트
8. 프롬프트 템플릿 생성
   - 프롬프트 템플릿
   - 챗 프롬프트 템플릿
   - 퓨샷 프롬프트 템플릿
9. LLM 모델 생성
10. 파이프라인 체인 생성
    1. 리트리버, 포맷팅 컨텍스트
    2. 프롬프트
    3. llm
    4. 출력 파싱
11. 체인 실행

<br>

## Document Loader

랭체인에서는 다양한 소스에서 문서를 불러오고 처리할 수 있다.

- 웹 문서(WebBaseLoader)
- 텍스트 문서(TextLoader)
- 디렉토리 폴더(Directory Loader)
- CSV 문서(CSVLoader)
- PDF 문서
  - PDF 문서 페이지별 로드(PyPDFLoader)
  - 형식이 없는 PDF 문서 로드(UnstructuredPDFLoader)
  - PDF 문서의 메타 데이터를 상세하게 추출(PyMuPDFLoader)
  - 온라인 PDF 문서로드(OnlinePDFLoader)
  - 특정 폴더의 모든 PDF 문서 로드(PyPDFDirectoryLoader)

<br>

이 모두 `langchain_community` 라이브러리의 `document_loaders` 모듈에 존재한다.

<br>

### 웹 문서(WebBaseLoader)

특정 웹 페이지의 내용을 로드하고 파싱할때 사용하는 클래스이다.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# 여러 개의 url 지정 가능
url1 = "https://blog.langchain.dev/customers-replit/"
url2 = "https://blog.langchain.dev/langgraph-v0-2/"

loader = WebBaseLoader(
    web_paths=(url1, url2),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-header", "article-content")
        )
    ),
)
docs = loader.load()
len(docs)

```

`web_paths` 파라미터는 URL을 단일 문자열 또는 여러개 URL을 시퀀스 배열로 지정할 수 있다. 지금은 튜플로 2개의 URL을 넣었다.

<br>

`bs_kwargs` 파라미터는 BeautifulSoup을 사용해서 HTML을 파싱할때 사용되는 파라미터들을 딕셔너리 형태로 전달한다. `bs4.SoupStrainer`를 사용해서 특정 클래스 이름을 가진 HTML 요소만 파싱하도록 했다.

`"article-header"`, `"article-content”` 클래스를 가진 요소만 선택해서 파싱한다.

<br>

### 텍스트 문서(TextLoader)

TextLoader는 파일 경로를 파라미터로 받아 파일을 불러온다.

load 메서드를 통해 파일 내용을 담고 있는 Document 객체로 변환시킨다.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader('history.txt')
data = loader.load()
```

<br>

### 디렉토리 폴더(DirectoryLoader)

디렉토리 내의 모든 문서를 로드할 수 있다.

DirectoryLoader 인스턴스를 생성할 때 문서가 있는 디렉토리의 경로와 해당 문서를 식별할 수 있는 glob 패턴을 지정한다.

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(path='./', glob='*.txt', loader_cls=TextLoader)

data = loader.load()
```

`./` 내의 모든 `.txt` 파일을 찾는 작업을 하고 있다.

디렉토리 내의 파일들을 `TextLoader`로 가져온다.

<br>

### CSV 문서(CSVLoader)

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='한국주택금융공사_주택금융관련_지수_20160101.csv',
									 encoding='cp949')
data = loader.load()
```

file_path로 파일 경로를 입력해 csv를 로드 한다.

인코딩 방식을 cp949로하여 한국어 인코딩이 가능하게 했다.

각 CSV 샘플(행) 들이 리스트 형태로 Document객체에 들어가 있다.

<br>

### PDF 문서

**PDF 문서 페이지별 로드(PyPDFLoader)**

`pypdf` 라이브러리를 설치해야한다.

```python
!pip install -q pypdf
```

<br>

파일 경로를 넣어주면 된다.

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_filepath = '000660_SK_2023.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()
```

<br>

Document 객체가 반환된다.

metadata 속성에는 파일의 출처와 해당 페이지 번호가 있다.

<br>

**형식이 없는 PDF 문서 로드(UnstructuredPDFLoader)**

**PDF 파일에서 텍스트를 추출할때** `unstructured` 라이브러리를 사용한다.

`unstructured` 라이브러리는 PDF 파일 내의 다양한 청크를 서로다른 elements로 생성한다.

이 elements를 결합해서 하나의 텍스트 데이터로 반환한다.

<br>

설치

```python
!pip install unstructured unstructured-inference
```

<br>

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

pdf_filepath = '000660_SK_2023.pdf'

# 전체 텍스트를 단일 문서 객체로 변환
loader = UnstructuredPDFLoader(pdf_filepath)
pages = loader.load()
```

<br>

**PDF 문서의 메타 데이터를 상세하게 추출(PyMuPDFLoader)**

PDF 파일의 페이지를 로드하고 각 페이지를 개별 document 객체로 추출한다.

PDF 문서의 자세한 메타데이터를 추출할 때 사용한다.

<br>

설치

```python
!pip install pymupdf
```

<br>

```python
from langchain_community.document_loaders import PyMuPDFLoader

pdf_filepath = '000660_SK_2023.pdf'

loader = PyMuPDFLoader(pdf_filepath)
pages = loader.load()
```

<br>

다른 로더들보다 메타 데이터가 더 자세하게 포함되어있다.

```python
{'source': '000660_SK_2023.pdf',
 'file_path': '000660_SK_2023.pdf',
 'page': 0,
 'total_pages': 21,
 'format': 'PDF 1.6',
 'title': '',
 'author': '',
 'subject': '',
 'keywords': '',
 'creator': 'Adobe InDesign 16.2 (Macintosh)',
 'producer': 'Adobe PDF Library 15.0',
 'creationDate': "D:20230626161631+09'00'",
 'modDate': "D:20230626172106+09'00'",
 'trapped': ''}
```

<br>

**온라인 PDF 문서로드(OnlinePDFLoader)**

웹 사이트에 있는 PDF 문서를 로드해서 페이지 내용을 추출 할 수있다.

```python
from langchain_community.document_loaders import OnlinePDFLoader

# Transformers 논문을 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
pages = loader.load()
```

<br>

**특정 폴더의 모든 PDF 문서 로드(PyPDFDirectoryLoader)**

디렉토리 경로를 입력해서 디렉토리안의 모든 PDF 문서를 로드 한다.

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader('./')
data = loader.load()
```

<br>

## Text Chunking

입력 길이 제한 내 텍스트가 너무 길면 모델이 검토해야할 정보량이 과다해져서 답변의 정확도가 떨어진다.

RAG는 긴 입력 문서를 작은 단위로 분할해서 처리한다. 이렇게 텍스트를 자르는 것을 청킹이라고 부른다.

<br>

청킹에는 다양한 전략들이 있다. 그중 가장 대표적인 방식은 크게 두가지이다.

1. 길이 단위 청킹(RecursiveCharacterTextSplitter)
2. 의미 고려 청킹(SemanticChunker)

<br>

### 길이 단위 청킹(RecursiveCharacterTextSplitter)

랭체인에서 텍스트 분할 도구 중 가장 많이 쓰이는 것이 RecursiveCharacterTextSplitter이다.

긴 텍스트를 받아서 특정 규칙에 따라 재귀적으로 분할해서 더 짧은 단위의 청크를 만든다.

<br>

사용자가 각 청크의 최대길이를 지정할 수 있다. 예를들어 10000인 텍스트에 청크의 최대길이를 500으로 하면, 500 길이를 초과하지 않는 여러개의 청크로 분할된다.

분할 방식은 다음과 같다.

1. 가장 먼저 `\n\n`(두번의 줄바꿈)을 기준으로 텍스트를 나눈다.
2. 나눈 청크가 사용자가 원하는 길이보다 길다면, `\n`(한번의 줄바꿈)을 기준으로 다시 나눈다.
3. 여전히 나눈 청크가 사용자가 원하는 길이보다 길다면, `“ “`(공백)을 기준으로 나누는 작업을 한다.

<br>

이렇게 점진적으로 더 작은 단위로 분할해서 지정한 최대 길이와 근사한 청크를 얻는다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splited_texts = text_splitter.create_documents([file])

splited_texts[1].page_content
```

`chunk_size=500` → 청크 길이가 500을 넘지 않는다.

`chunk_overlap=0` → 청크가 얼마나 겹치는지 정한다. 0을 지정해서 내용이 겹치지 않는다.

<br>

파이썬 문자열을 분할할때 `create_documents()`을 사용한다.

출력 결과는 `Document(page_content=”텍스트”)` 형태를 가진다.

랭체인에서 텍스트를 청크들로 나눌때 나오는 형태이다.

청크에 있는 문자열을 보려면 `.page_content`을 하면된다.

<br>

**chunk_overlap**

`chunk_overlap=50` 을하면 각 청크의 앞뒤 간 길이의 50정도의 내용이 겹친다.

<br>

길이 단위 청킹은 문맥을 파악하기 위해 분할 하는 것이 아니라 오로지 길이를 맞추기 위해서이다.

따라서 청킹의 텍스트를 보면 중간 내용이 끊기게 된다.

이렇게 되면 내용의 마무리가 되어있지 않아, 모델의 성능이 저하 된다.

<br>

이러한 문제를 해결하기 위해서 문맥 단위로 분할한다.

<br>

만약 문자수가 아닌 토큰수로 분할한다면 아래와 같이하면 된다.

```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
)
```

<br>

### 의미 고려 청킹(**SemanticChunker)**

긴 길이를 짧게 만드는 청킹 기술이다.

각 문장을 임베딩 벡터로 변환한 뒤, 문장간의 유사도를 구해서 유사한 문장끼리 그룹화 하도록 동작한다.

따라서 어느정도 문맥의 의미가 고려되서 청크를 분할하게 된다.

<br>

OpenAI의 Embedding API를 사용한다.

```python
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(OpenAIEmbeddings())
texts = text_splitter.create_documents([file])
```

SemanticChunker가 문서를 분할하는 방법은 크게 세 가지가 있다.

- 백분위수(Percentile) 방식(기본값)
- 표준편차(Standard Deviation) 방식
- 사분위수(Interquartile) 방식

<br>

세가지 모두 코사인 거리라는 개념을 사용한다.

코사인 거리는 두 문장간의 의미적 차이를 나타낸다.

<br>

## Embedding Model

임베딩 모델은 사람이 이해할 수 있는 문장에서 컴퓨터가 이해할 수 있는 숫자로 이루어진 벡터 형식으로 변환해, 문장의 의미를 파악할 수있는 의미 좌표로 만든다.

<br>

임베딩 모델를 사용해서 사용자가 질문한(Query)를 임베딩 벡터 형식으로 변환시킨다.

개발자의 목표와 환경에 맞게 임베딩 모델을 제공하는 라이브러리를 선택해야한다.

- OpenAI: 사용하기 쉬운 고성능 모델을 제공하는데 중점을 둔다. `text-embedding-3-large` 와 `text-embedding-3-small` 같은 임베딩 모델이 있다. API 사용법이 간단해서 빠르게 적용하고 싶을때 적합하다.
- Google(Gemini): `text-embedding-004` 같은 Gemini 기반 임베딩 모델이 있다.
- Hugging Face: 하나의 모델이 아니라 수많은 임베딩 모델들이 있다. 직접 파인튜닝해서 성능을 향상시키기 위해서도 좋다.

<br>

### **OpenAIEmbeddings**

OpenAI API를 사용해서 각 문서를 임베딩 벡터로 변환한다.

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np
from numpy import dot
from numpy.linalg import norm

embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))
```

`embed_documents(list['문서1': str,''문서2'...])`

메서드에 각 문서를 요소로하고 있는 리스트를 전달한다.

각 문서들은 임베딩 벡터가 된다. 문서 임베딩 벡터를 담고있는 리스트가 반환된다.

OpenAI 임베딩 모델은 1536 차원이다.

shape: `(문서 수, 1536)`

<br>

`embed_query('Query(질문)': str)`

메서드에 단일 쿼리(질문) 문자열을 전달해서 임베딩 벡터를 만든다.

Query 벡터와 Document 벡터의 코사인 유사도를 구해 유사한 답변을 찾을 수 있다.

<br>

`cos_sim(문서 벡터, 쿼리 벡터)`

쿼리 벡터와와 문서 벡터 간의 코사인 유사도를 계산한다.

이 유사도 점수를 바탕으로 어떤 쿼리와 가장 관련이 있는 문서인지 찾을 수 있다.

<br>

### HuggingFaceEmbeddings

`sentence-transformers` 라이브러리를 사용해서 Hugging Face 모델에서 사용된 사전 훈련된 임베딩 모델을 다운받을 수 있다.

Hugging Face는 오픈소스이여서 OpenAI와 다르게 요금이 부과되지 않는다.

<br>

설치

```python
!pip install -U sentence-transformers
```

<br>

임베딩 모델 생성

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

embeddings_model.word_embedding_dimension
```

`model_name='모델 이름'`

사용할 임베딩 모델의 이름을 지정한다.

<br>

`model_kwargs`

모델의 환경 설정을 한다.

<br>

`encode_kwargs`

인코딩 환경설정을 한다. 임베딩을 정규화해서 모든 벡터가 같은 범위 값을 가지게한다.

유사도 계산 할때 일관성을 높인다.

<br>

`embeddings_model.word_embedding_dimension` 값을 보면 임베딩 벡터의 차원 크기를 볼 수 있다.

<br>

임베딩 적용

```python
import numpy as np
from numpy import dot
from numpy.linalg import norm

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))

# 0.5899016189601531
# 0.4182631225980652
# 0.7240604521610333
# 0.05702662997392148
# 0.4316418328113528
```

`embed_documents(list['문서1': str,''문서2'...])`

메서드에 각 문서를 요소로하고 있는 리스트를 전달한다.

각 문서들은 임베딩 벡터가 된다. 문서 임베딩 벡터를 담고있는 리스트가 반환된다.

<br>

`embed_query('Query(질문)': str)`

메서드에 단일 쿼리(질문) 문자열을 전달해서 임베딩 벡터를 만든다.

Query 벡터와 Document 벡터의 코사인 유사도를 구해 유사한 답변을 찾을 수 있다.

<br>

`cos_sim(문서 벡터, 쿼리 벡터)`

쿼리 벡터와와 문서 벡터 간의 코사인 유사도를 계산한다.

이 유사도 점수를 바탕으로 어떤 쿼리와 가장 관련이 있는 문서인지 찾을 수 있다.

<br>

### **GoogleGenerativeAIEmbeddings**

`langchain_google_genai` 라이브러리를 사용해서 문장을 임베딩 할 수 있다.

<br>

설치

```python
!pip install -q langchain_google_genai
```

<br>

Google API를 사용하기 위해 API 키를 설정한다.

```python
os.environ['GOOGLE_API_KEY'] = '구글 API 키를 입력하면된다.'
```

<br>

모델 생성

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))

```

`embed_documents(list['문서1': str,''문서2'...])`

메서드에 각 문서를 요소로하고 있는 리스트를 전달한다.

각 문서들은 임베딩 벡터가 된다. 문서 임베딩 벡터를 담고있는 리스트가 반환된다.

<br>

`embed_query('Query(질문)': str)`

메서드에 단일 쿼리(질문) 문자열을 전달해서 임베딩 벡터를 만든다.

Query 벡터와 Document 벡터의 코사인 유사도를 구해 유사한 답변을 찾을 수 있다.

<br>

`cos_sim(문서 벡터, 쿼리 벡터)`

쿼리 벡터와와 문서 벡터 간의 코사인 유사도를 계산한다.

이 유사도 점수를 바탕으로 어떤 쿼리와 가장 관련이 있는 문서인지 찾을 수 있다.

<br>

## Vector Store

벡터 저장소(vector store)는 임베딩 벡터들을 효율적으로 저장하고 검색할 수 있는 데이터베이스를 말한다.

NLP, 이미지 처리 등 다양한 고차원 벡터 데이터를 관리하기 위해 만들어졌다.

벡터 스토어의 핵심은 대규모 벡터 데이터셋에서 빠른 속도로 가장 유사한 벡터를 찾아내는 것이다.

<br>

유사도 점수를 통해 가장 유사한 벡터들을 순서대로 반환한다.

벡터 스토어는 다양한 라이브러리들이 있다.

- Chroma
- FAISS(Facebook AI Similarity Search)
- Elasticsearch
- Pinecone

<br>

### Chroma

스플리터는 텍스트를 청크 단위로 나눠준다.

**스플리터 생성**

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader('history.txt')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

texts = text_splitter.split_text(data[0].page_content)
texts[0]

```

`TextLoader('텍스트 경로/텍스트 파일 이름.txt')`

텍스트 파일을 로드한다.

<br>

`RecursiveCharacterTextSplitter.from_tiktoken_encoder()`

로드된 텍스트를 여러개의 청크로 분할한다.

- (`RecursiveCharacterTextSplitter()`만 사용하면 토큰이 아니라 문장 길이로 자름)

`chunk_size=250` 청크의 250 토큰을 넘지 않는다.

`chunk_ovelap=50` 각 청크의 앞뒤 간 길이의 50정도의 내용이 겹친다.

`encoding_name='cl100k_base'` 어떤 규칙으로 토큰으로 나눌지에 대한 토크나이저 이름이다. 사용할 모델과 동일한 토크나이저를 사용해야한다.

<br>

`text_splitter.split_text('텍스트')`

텍스트를 앞서 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` 에서 설정한 값에 따라서 텍스트를 토큰화 한다.

이후 토큰들을 청크 길이에 맞춰서 나눈다.

<br>

**임베딩 모델 및 Chroma 벡터 스토어 생성**

```python
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_texts(
		texts=texts,
		embedding=embeddings_model,
    collection_name = 'history',
    persist_directory = './db/chromadb',
    collection_metadata = {'hnsw:space': 'cosine'}, # l2 is the default
)

query = '누가 한글을 창제했나요?'
docs = db.similarity_search(query)
mmr_docs = db.max_marginal_relevance_search(query, k=4, fetch_k=10)

print(docs[0].page_content)
print(mmr_docs[0].page_content)
```

`from_texts()`

- `texts=texts`: 텍스트를 벡터로 만들때 사용할 임베딩 모델을 전달한다.
- `embedding=embeddings_model`: 임베딩 모델로 분할된 텍스트들을 임베딩해서 `Chroma` 벡터 스토어에 저장한다.
- `collection_name = 'history'`: 데이터베이스 내에서 이 데이터 묶음을 식별한 고유한 이름을 `‘history’`로 지정한다.
- `persist_directory = './db/chromadb'`: 데이터베이스를 지정할 디스크 경로이다. 이 설정 덕분에 `Chroma(persist_directory=’./db/chromadb’)` 코드로 저장된 DB를 다시 불러온다.
- `collection_metadata = {'hnsw:space': 'cosine'}`: 데이터베이서의 유사도 계산 방식을 설정한다. 코사인 유사도를 사용하겠다고 명시한다. 디폴트 값은 L2인 유클리드 거리이다.

<br>

`db.similarity_search(query)`

데이터베이스에서 쿼리와 가장 유사한 청크를 찾는다.

유사도 계산에 따라 가장 유사한 청크들을 순서대로 반환한다.

가장 유사한 청크는 `docs[0].page_content`를 통해 볼수있다.

<br>

`db.max_marginal_relevance_search(query)`

코사인 유사도와 다르게 MMR(Maximal Marginal Relevance) 검색 알고리즘을 사용한다.

우리 말로 직역하면 ‘최대 한계 관련성’이라고 부른다. 이렇게 부른 이유는 아래와 같다.

- 최대: 관련성과 다양성을 종합한 점수가 가장 높은 최적의 결과를 선택한다.
- 한계: 가장 관련성이 있는 정보를 뽑는다. 이미 뽑힌 정보와 비슷하면 점수를 깎는다.
- 관련성: 검색결과가 질문과 얼마나 관련있는지 나타낸다.

<br>

`fetch_k=10`: query와 가장 유사한 상위 10개의 유사한 문서를 가져온다.

`k=4`:

1. 가장 유사한 문서를 첫번째로 뽑는다.
2. 나머지 문서들 중 첫번째 문서랑 유사하거나 중복된것들은 MMR 점수로 계산해서 낮게 준다.
3. MMR 점수가 가장 높은 문서를 다음 문서를 선택한뒤 이과정을 k개 문서까지 반복한다.

<br>

`from_documents()`

```python
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./chroma_db"  # 벡터 스토어를 디스크에 저장
)
```

`Chroma.from_texts`는 순수 텍스트 리스트만 입력을 받는다.

`Chroma.from_documents`는 파일을 `TextSplitter`로 잘라서 `Document` 객체 리스트로 만든 직후, 텍스트 내용과 메타데이터를 함께 포함하게 한다.

<br>

### FAISS(Facebook AI Similarity Search)

Facebook AI Reasearch에 의해 개발된 라이브러리이다.

FAISS는 특히 벡터의 압축된 표현을 사용해서 메모리 사용량을 최소화 하면서 검색속도를 높인다는 특징이있다.

<br>

설치

```python
# CPU만 사용하는 버전
!pip install faiss-cpu sentence-transformers

# GPU만 사용하는 버전
!pip install faiss-gpu sentence-transformers
```

<br>

```python
# 벡터스토어 db 인스턴스를 생성
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore = FAISS.from_documents(documents=documents,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE
                                  )

query = '카카오뱅크가 중대성 평가를 통해 도출한 6가지 중대 주제는 무엇인가?'
docs = vectorstore.similarity_search(query)
```

HuggingFaceEmbeddings 클래스를 사용해서 `‘jhgan/ko-sbert-nli'` 모델을 불러온다.

문서 객체를 임베딩 벡터로 변환하여 벡터 저장소에 저장할 때, 모델의 입력 길이 제한을 고려해야 한다.

<br>

`FAISS.from_documents()`

- `documents=documents:` 파일을 `TextSplitter`로 잘라서 `Document` 객체 리스트로 만든 직후, 텍스트 내용과 메타데이터를 함께 임베딩 벡터로 변환시켜 저장한다.
- `embedding=embeddings_model`: 임베딩 모델로 분할된 텍스트들을 임베딩해서 `FAISS` 벡터 스토어에 저장한다.
- `distance_strategy = DistanceStrategy.COSINE`: 벡터 간 유사도를 측정하는 방법을 지정한다. 코사인 유사도를 사용한다.

<br>

`vectorstore.similarity_search(query)`

query와 가장 유사한 문서 순서로 정렬해서 리스트에 반환한다.

<br>

MMR 검색 기법

```python
mmr_docs = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)
```

코사인 유사도와 다르게 MMR(Maximal Marginal Relevance) 검색 알고리즘을 사용한다.

우리 말로 직역하면 ‘최대 한계 관련성’이라고 부른다. 이렇게 부른 이유는 아래와 같다.

- 최대: 관련성과 다양성을 종합한 점수가 가장 높은 최적의 결과를 선택한다.
- 한계: 가장 관련성이 있는 정보를 뽑는다. 이미 뽑힌 정보와 비슷하면 점수를 깎는다.
- 관련성: 검색결과가 질문과 얼마나 관련있는지 나타낸다.

<br>

`fetch_k=10`: query와 가장 유사한 상위 10개의 유사한 문서를 가져온다.

`k=4`:

1. 가장 유사한 문서를 첫번째로 뽑는다.
2. 나머지 문서들 중 첫번째 문서랑 유사하거나 중복된것들은 MMR 점수로 계산해서 낮게 준다.
3. MMR 점수가 가장 높은 문서를 다음 문서를 선택한뒤 이과정을 k개 문서까지 반복한다.

<br>

**FAISS DB를 로컬에 저장**

벡터 스토어 DB를 저장하고 불러오는 과정에서 생성된 벡터 인덱스를 재사용한다.

```python
# save db
vectorstore.save_local('./db/faiss')

# load db
db3 = FAISS.load_local('./db/faiss', embeddings_model)
```

`vectorstore` 상태를 `'./db/faiss'` 경로에 저장한다.

해당 경로로 불러온뒤, 벡터 스토어를 다시 생성할 임베딩 모델을 전달한다.

벡터스토어를 만들때 사용했던것과 동일한 임베딩 모델을 사용해야한다.

<br>

## Retriever

벡터 스토어 리트리버(되찾다)를 사용하면 대량의 텍스트 데이터 정보를 효율적으로 검색할 수 있다.

### Vector Store Retriver

**문서 로드 및 청킹**

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

loader = PyMuPDFLoader('323410_카카오뱅크_2023.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)
len(documents)
```

<br>

**문서 임베딩 및 벡터 스토어 저장**

```python
# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore = FAISS.from_documents(documents,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE
                                  )
```

<br>

**문서 검색**

```python
# 검색 쿼리
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

# 가장 유사도가 높은 문장을 하나만 추출
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

docs = retriever.get_relevant_documents(query)
print(len(docs))
docs[0]
```

`vectorstore.as_retriever()`

벡터 스토어의 Retriever 객체를 설정과 함께 생성한다.

<br>

`retriever.get_relevant_documents(query)`

Retiever 객체에 있는 설정 조건에 맞춰 query와 가장 유사한 문서을 검색한다.

<br>

**MMR 검색 기법**

```python
# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50, 'lambda_mult': 0.5}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
docs[0]
```

`search_type='mmr':` 검색 기법을 설정한다.

<br>

`lambda_mult` 는 유사성과 다양성을 조절한다.

이 값은 `search_type=’mmr’` 로 설정되었을 때만 가능하며, 0과 1사이의 값을 가진다.

1에 가까울 수록 유사성에 더 큰 중점을 둔다.

0에 가까울수록 다양성에 더 큰 중점을 둔다.

<br>

`lambda_mult` 값이 높을때(0.7~0.9)

- 첫번째 정보는 쿼리와 가장 높은 유사도를 가진다.
- 두번째 정보는 첫본째 정보와 0.7~0.9만큼의 유사성을 가진 정보을 가진다.
- 즉, 거의 다양성이 없고 유사성이 크다보니, 쿼리(질문)에 가장 명확한 답변을 해준다.
- 예시) 대한민국의 수도는 어딘가요? → 구체적인 질문이다 보니 유사성이큰 답변을 해주어야한다.

<br>

`lambda_mult` 값이 낮을때(0.3~0.6)

- 첫번째 정보는 쿼리와 가장 높은 유사도를 가진다.
- 두번째 정보는 첫번째 정보와 0.3~0.6만큼의 유사성을 가진 정보를 가진다.(거의 다름)
- 만약 0.0과 같은 값이 온다면, 첫번째 정보와 관련성이 거의 없는 정보를 선택한다.
- 즉, 쿼리(질문)에 대한 답보다 더 넓은 범위로 다양하게 답변을 해준다.
- 예시) 인공지능의 역사에 대해 알려줘 → 구체적인 질문이 아닌, 넓은 범위의 질문이다. 따라서 낮은 값으로 다양한 정보를 제공해주는것이 좋다.

<br>

### Multi Query Retriever

사용자가 던지는 원래 질문이 너무 애매하거나 한가지 측면만 담고 있을때가 있다.

이 질문 하나로 관련된 모든 중요한 정보를 찾아내지 못하는 경우가 있다.

이런 벡터 스토어 리트리버의 한계를 극복하기 위해 생겨난 검색 방법이다.

<br>

사용자의 하나의 질문을 받아서 그 질문의 다양한 관점을 반영한 여러개의 질문을 생성한다.

이 질문들을 사용해서 더 풍부하고 정확한 정보를 검색하는 기술이다.

```python
import logging
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOpenAI(
    model='gpt-3.5-turbo-0125',
    temperature=0,
    max_tokens=500,
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

question = "RAG의 한계와 구성요소에 대해 알려줘"
retrieved_docs = retriever.invoke(query=question)

# 결과 출력
print(f"원본 질문: {question}")
print(f"총 {len(retrieved_docs)}개의 관련 문서가 검색되었습니다.")
print("--- 검색된 문서 내용 ---")
for i, doc in enumerate(retrieved_docs):
    print(f"문서 {i+1}: {doc.page_content}")
```

<br>

```python
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
```

코드가 실행될 때 LLM이 원본 질문을 바탕으로 **어떤 새로운 질문들을 생성**했는지 로그로 직접 볼 수 있다.

<br>

출력 결과

```python
INFO:langchain.retrievers.multi_query:Generated queries:
[
    'RAG의 구성요소는 무엇인가?',
    'RAG 기술의 단점이나 한계점은 무엇인지 설명해줘.',
    'RAG 시스템의 주요 구성 요소와 그 각각의 역할은 무엇인가?'
]

원본 질문: RAG의 한계와 구성요소에 대해 알려줘

총 3개의 관련 문서가 검색되었습니다.

--- 검색된 문서 내용 ---
문서 1: RAG의 핵심 구성요소는 크게 세 가지로 나뉩니다. 첫째는 대규모 문서를 저장하는 '벡터 스토어'입니다.
둘째는 질문과 관련된 문서를 찾아내는 '리트리버'입니다. 마지막은 검색된 정보로 답변을 만드는 '생성기'입니다.

문서 2: 하지만 RAG에도 한계는 존재합니다. 검색의 품질이 낮으면, 관련 없는 정보로 인해 답변의 질이 떨어집니다.
또한, 데이터를 최신으로 유지하고 관리하는 데 지속적인 노력이 필요합니다.

문서 3: 이러한 단점에도 불구하고, RAG는 환각을 줄이는 데 매우 효과적입니다.
```

<br>

### Contextual compression

일반적인 리트리버는 질문과 유사한 문서 전체나 청크를 통째로 가져온다.

결과물에는 질문과 직접적인 관련이 없는 불필요한 문장들이 함께 포함될 가능성이 높다.

이러한 불필요한 정보는 노이즈로 작용해서 오히려 LLM의 답변 생성 과정을 방해할 수 있다.

이러한 한계를 해결하기 위해 컨텍츄얼 컴프레션 기법이 생겼다.

<br>

컨텍스츄얼 컴프레션은 이러한 문제를 해결하기 위해서

리트리버가 가져온 문서들을 최종적으로 LLM에게 보내기전에 특정 작업을 거친다.

중간에 다큐먼츠 컴프레서(Documnet Compressor, 문서 압축기)라는 필터를 한번 더 거치게 한다.

<br>

다큐먼츠 컴프레서는 여러 종류가 있다.

- LLMChainExtractor
- EmbeddingsFilter

<br>

**LLMChainExtractor**

LLMChainExtractor이라는 컴프레서는 리트리버가 찾아온 각 문서의 내용을 다시 LLM에 보여준다.

‘이 문서 내용 중에서 원래 질문과 관련된 문장들만 골라줘’ 라는 새로운 프롬프트를 실행해서 핵심 내용만을 추출한다.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 리트리버 및 압축기(Compressor) 생성
# 압축하지 않을 일반 리트리버를 생성
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 문서 내용을 압축할 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLMChainExtractor 압축기 생성
compressor = LLMChainExtractor.from_llm(llm)

# 컨텍스츄얼 컴프레션 리트리버 생성
# 기본 리트리버와 압축기를 결합
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 리트리버 실행 및 결과 비교
question = "RAG의 단점은 뭐야?"

# (비교용) 압축 전 결과
print("--- 1. 압축 전 검색 결과 (기본 리트리버) ---")
retrieved_docs_before = base_retriever.invoke(question)
for doc in retrieved_docs_before:
    print(f"[문서 내용]: {doc.page_content}\n")

# 압축 후 결과
print("\n--- 2. 압축 후 검색 결과 (Contextual Compression Retriever) ---")
retrieved_docs_after = compression_retriever.invoke(question)
for doc in retrieved_docs_after:
    print(f"[압축된 내용]: {doc.page_content}\n")
```

```python
--- 1. 압축 전 검색 결과 (기본 리트리버) ---
[문서 내용]: 하지만 RAG에도 단점은 존재합니다. 검색 단계에서 관련 없는 문서가 추출되면, 오히려 답변의 품질이 저하될 수 있습니다. 데이터베이스를 최신으로 유지하는 비용도 발생합니다.

[문서 내용]: RAG의 가장 큰 장점은 환각(hallucination)을 줄일 수 있다는 점입니다. 또한, 모델의 학습 데이터에 없는 최신 정보에 대해서도 답변이 가능합니다.

[문서 내용]: 검색 증강 생성(RAG)은 LLM의 한계를 보완하는 기술입니다. 외부 지식 베이스를 참조하여 답변의 정확성을 높입니다.

--- 2. 압축 후 검색 결과 (Contextual Compression Retriever) ---
[압축된 내용]: 하지만 RAG에도 단점은 존재합니다. 검색 단계에서 관련 없는 문서가 추출되면, 오히려 답변의 품질이 저하될 수 있습니다. 데이터베이스를 최신으로 유지하는 비용도 발생합니다.
```

`ContextualCompressionRetriever` 은 기본 리트리버가 가져온 3개 문서를 LLMChainExtractor에게 하나씩 전달한다.

`LLMChainExtractor`는 내부적으로 LLM을 호출하여 각 문서의 내용을 보고 **"이 내용이 'RAG의 단점'이라는 질문과 관련이 있는가?"**를 판단한다.

- RAG의 '장점'과 '정의'에 대한 문서는 질문과 직접적인 관련이 없다고 판단하여 **버린다.**
- '단점'에 대한 문서는 질문과 직접 관련이 있으므로 **그대로 통과시킨다.**

최종적으로 압축 후 결과에는 질문과 100% 관련된 **단 하나의 문서만** 남게 된다.

이렇게 정제된 정보는 LLM이 더 정확하고 집중된 답변을 생성하는 데 큰 도움이 된다.

<br>

**EmbeddingsFilter**

임베딩스필터는 임베딩 기반의 유사도를 기준으로 관련 없는 정보를 걸러내는 역활을 한다.

리트리버가 찾아온 문서의 **각 문장들을 다시 한번 임베딩한다.**

사용자의 원본 질문 벡터와 유사도를 개별적으로 계산한다.

그 유사도 점수가 미리 정해놓은 특정 기준치인 **임계값(Threshold)을 넘는 문장들만 통과** 시킨다.

기준에 **미달하는 문장들은 버리는 방식**으로 정보를 압축한다.

```python
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 기본 리트리버 및 압축기(Compressor) 생성
# 압축하지 않을 일반 리트리버를 생성
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 유사도 계산에 사용할 임베딩 모델을 정의합니다.
embeddings = OpenAIEmbeddings()

# EmbeddingsFilter 압축기 생성
# similarity_threshold: 이 값보다 유사도가 낮은 문서는 거른다. (0~1 사이)
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)

# 컨텍스츄얼 컴프레션 리트리버 생성
# 기본 리트리버와 압축기를 결합
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 리트리버 실행 및 결과 비교
question = "RAG의 단점은 뭐야?"

# 압축 전 결과
print("--- 1. 압축 전 검색 결과 (기본 리트리버) ---")
retrieved_docs_before = base_retriever.invoke(question)
for doc in retrieved_docs_before:
    print(f"[문서 내용]: {doc.page_content}\n")

# 압축 후 결과
print("\n--- 2. 압축 후 검색 결과 (Contextual Compression Retriever) ---")
retrieved_docs_after = compression_retriever.invoke(question)
for doc in retrieved_docs_after:
    print(f"[압축된 내용]: {doc.page_content}\n")
```

```python
--- 1. 압축 전 검색 결과 (기본 리트리버) ---
[문서 내용]: 하지만 RAG에도 단점은 존재합니다. 검색 단계에서 관련 없는 문서가 추출되면, 오히려 답변의 품질이 저하될 수 있습니다. 데이터베이스를 최신으로 유지하는 비용도 발생합니다.

[문서 내용]: RAG의 가장 큰 장점은 환각(hallucination)을 줄일 수 있다는 점입니다. 또한, 모델의 학습 데이터에 없는 최신 정보에 대해서도 답변이 가능합니다.

[문서 내용]: 검색 증강 생성(RAG)은 LLM의 한계를 보완하는 기술입니다. 외부 지식 베이스를 참조하여 답변의 정확성을 높입니다.

--- 2. 압축 후 검색 결과 (Contextual Compression Retriever) ---
[압축된 내용]: 하지만 RAG에도 단점은 존재합니다. 검색 단계에서 관련 없는 문서가 추출되면, 오히려 답변의 품질이 저하될 수 있습니다. 데이터베이스를 최신으로 유지하는 비용도 발생합니다.
```

`ContextualCompressionRetriever`는 이 3개의 문서를 `EmbeddingsFilter`에게 전달한다.

`EmbeddingsFilter`는 **LLM을 호출하지 않는다.**

대신, 가져온 **각 문서의 내용을 다시 임베딩**하여 원본 질문("RAG의 단점은 뭐야?")의 임베딩 벡터와 **유사도를 직접 계산한다.**

- 계산된 유사도 점수가 `similarity_threshold`로 설정한 `0.8`보다 낮은 문서는 버린다.
- '단점'에 대한 문서는 질문과 유사도가 매우 높아 `0.8`을 넘으므로 **통과시킨다.**

<br>

**최종적으로** `LLMChainExtractor`와 마찬가지로 질문과 가장 관련 있는 문서만 남는다.

`EmbeddingsFilter`는 LLM 호출이 없어 **더 빠르고 비용이 저렴**하지만, LLM만큼 문맥을 깊게 이해하지는 못할 수 있다.

반면 `LLMChainExtractor`는 더 정교하지만 비용과 시간이 더 소요되므로, 상황에 맞게 선택하여 사용하는 것이 좋다.

<br>

## Prompt Template

프롬프트는 모델에게 특정 작업을 수행하도록 입력하는 텍스트이다.

템플릿은 미리 프롬프트의 특정 형식이나 구조를 정의하는 것이다.

따라서 프롬프트 템플릿은 매번 프롬프트를 처음부터 작성하지 않고 안전고 쉽게 생성할 수 있게해준다.

프롬프트는 특정 태스크에 따라서 달라진다.

해당 태스크에 맞는 템플릿을 선택하고 필요한 설정만 바꿔주면 된다.

그래서 라이브러리에서 다양한 종류의 프롬프트 템플릿이 존재한다.

- ChatPromptTemplate
- FewShotPromptTemplate
- PromptTemplate

<br>

### PromptTemplate

가장 기본적인 템플릿이다. 단순 태스크들에 사용한다.

<br>

설치

```python
!pip install langchain langchain_openai
```

<br>

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# LLM 모델을 준비
# 실제 실행을 위해서는 OPENAI_API_KEY 환경 변수 설정이 필요
llm = ChatOpenAI(model="gpt-4o")

# PromptTemplate을 생성
# {subject}는 사용자의 입력을 받을 변수
prompt_template = PromptTemplate.from_template(
    "{subject}에 대한 짧은 농담 하나만 해줘."
)

# 포멧팅(formatting)
# 템플릿에 실제 값을 넣어 프롬프트를 완성
# .format() 메서드를 사용하여 변수에 값을 채움
formatted_prompt = prompt_template.format(subject="컴퓨터")

# 프롬프트 템플릿과 LLM을 연결해 체인(Chain)을 생성
chain = prompt_template | llm

# 체인을 실행
response = chain.invoke({"subject": "컴퓨터"})

# --- 결과 출력 ---

# 템플릿이 어떻게 최종 프롬프트로 변환되었는지 확인
print("--- 최종 생성된 프롬프트 ---")
print(formatted_prompt)
print("-" * 25)

# AI의 최종 답변 확인
print(f"AI가 생성한 농담: {response.content}")
```

- 템플릿 정의: `PromptTemplate.from_template()` 안에 템플릿으로 사용할 문자열을 넣는다.
- 변수 치환: `.format(subject="컴퓨터")`가 실행되면, 템플릿의 `{subject}` 부분이 "컴퓨터"라는 사용자 입력이 들어간다.
- 프롬프트 완성: 최종적으로 "컴퓨터에 대한 짧은 농담 하나만 해줘."라는 완전한 문장이 만들어져 AI 모델에 전달한다.

<br>

결과 출력

```python
--- 최종 생성된 프롬프트 ---
컴퓨터에 대한 짧은 농담 하나만 해줘.
-------------------------

AI가 생성한 농담: 왜 컴퓨터는 항상 추워할까? 창문(Windows)을 너무 많이 열어둬서!
```

<br>

### ChatPromptTemplate

대화를 하는 챗 모델에 특화된 템플릿이다.

챗 모델은 일반적인 텍스트 덩어리를 입력받지 않는다.

role(역할) 정보가 포함된 메시지들의 목록을 입력으로 받는다.

- 시스템 메시지(SystemMessage): 챗봇 모델의 역할이나 정체성, 행동 지침을 설정하는 역할이다.
- 휴먼 메시지(HumanMessage): 사용자인 사람이 입력하는 메세지이다. 챗봇에게 보내는 모든 질문이나 요청이다.
- 에이아이 메시지(AIMessage): 챗봇 모델이 이전에 했던 답변이다. 이전의 대화 기록전체를 보고 답변을 한다.

이런 역할에 맞춘 프롬프트 템플릿으로 챗봇 모델 인풋에 전달한다.

<br>

설치

```python
!pip install langchain langchain_openai
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM 모델
llm = ChatOpenAI(model="gpt-4o")

# ChatPromptTemplate을 생성

# 시스템(system) 역할, AI의 전반적인 행동 지침을 설정
# {input_language}와 {output_language}는 나중에 채워질 변수

# 사용자(human) 역할, 사용자의 실제 입력
# {text}는 사용자가 번역을 요청할 문장이 들어갈 변수
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {input_language}를 {output_language}로 번역하는 전문 번역가입니다."),
	  ("human", "{text}")
])

# 포맷팅(formatting)
# 템플릿에 변수 값을 넣어 프롬프트를 완성
# .format_messages() 메서드를 사용하여 변수에 값을 채움
formatted_messages = chat_prompt.format_messages(
    input_language="한국어",
    output_language="영어",
    text="안녕하세요, 만나서 반갑습니다."
)

# 프롬프트 템플릿과 LLM을 연결하여 체인(Chain)을 생성
chain = chat_prompt | llm

# 체인을 실행
response = chain.invoke({
    "input_language": "한국어",
    "output_language": "영어",
    "text": "안녕하세요, 만나서 반갑습니다."
})

# --- 결과 출력 ---
# 템플릿이 어떤 메시지 목록으로 변환되었는지 확인
print("--- 최종 생성된 메시지 목록 ---")
for message in formatted_messages:
    # 각 메시지의 클래스 이름(역할)과 내용(content)을 출력
    print(f"역할: {message.__class__.__name__}, 내용: {message.content}")
print("-" * 25)

# AI의 최종 답변 확인
print(f"AI의 번역 결과: {response.content}")
```

1. 역할(SystemMessage)을 부여해서 모델에게 정체성과 규칙을 알려준다. 프롬프트 가장 앞에 위치해서 전체 대화의 방향성을 결정한다.
2. 질문 전달 (HumanMessage): 사용자가 AI에게 보낸 실제 메시지, 즉 "안녕하세요, 만나서 반갑습니다."라는 번역할 문장을 담음
3. 구조화된 입력: 이렇게 생성된 `SystemMessage`와 `HumanMessage` 객체들을 하나의 리스트(list)로 묶어서 채팅 모델에 전달한다. 채팅 모델은 이 구조화된 목록을 보고 시스템 메시지의 지침에 따라 사용자의 메시지에 응답한다.

<br>

실행 결과

```python
--- 최종 생성된 프롬프트 ---
역할: SystemMessage, 내용: 당신은 한국어를 영어로 번역하는 전문 번역가입니다.
역할: HumanMessage, 내용: 안녕하세요, 만나서 반갑습니다.
-------------------------

AI의 번역 결과: Hello, it's nice to meet you.
```

<br>

### FewShotPromptTemplate

퓨샷은 아주 적은 샘플만으로 학습시키는 방법이다.

퓨샷 러닝(Few-Shot Learning)은 모델이 적은 샘플만으로 패턴을 파악해서 새로운 문제에 적용하는 기법이다.

<br>

퓨샷 러닝도 이런 아이디어로 프롬프트에 적용한 것이다.

모델에게 단순히 “이것을 해줘”라고 지시하는 것이 아니라, 몇 개의 구체적인 질문과 답변 예시를 먼저 보여준다.

이후 진짜 원하는 질문을 던지는 방식의 프롬프트를 만든다.

예시)

먼저 두개의 예시 샘플과 질문할 부분을 비워둔 질문 샘플을 하나의 프롬프트에 만들어 입력으로 전달한다.

- 예시 샘플1 → "문장: 이 영화는 정말 최고야!, 감정: 긍정"
- 예시 샘플2 →"문장: 음식이 너무 맛이 없었어., 감정: 부정"
- 질문 샘플 → "문장: 오늘 하루는 정말 멋졌어., 감정:"
- `프롬프트 = (예시 샘플1 + 예시 샘플2 + 질문 샘플)`

<br>

이 템플릿은 두 부분으로 구성된다.

1. 이그잼플 프롬프트 템플릿(example_prompt template): 예시 샘플 구조로 만드는 템플릿이다. 샘플이 이그잼플 프롬프트 템플릿에 들어가 예시 샘플로 만들어진다.
2. 이그잼플스(examples): 이그잼플 프롬프트 템플릿에 들어갈 실제 예시 데이터 목록이다.

<br>

즉, 예시 샘플 데이터를 이그잼플 프롬프트 템플릿에 하나식 적용해서 예시 샘플을 만든다.

이후 사용자의 프롬프트와 합쳐서 완성된 퓨샷 프롬프트를 만든다.

<br>

설치

```python
!pip install langchain langchain_openai
```

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

# LLM 모델
llm = ChatOpenAI(model="gpt-4o")

# 모델에게 제공할 예시 데이터
examples = [
    {
        "input": "행복",
        "output": "불행"
    },
    {
        "input": "빠른",
        "output": "느린"
    },
    {
        "input": "뜨거운",
        "output": "차가운"
    }
]

# 예시 샘플 프롬프트 템플릿 생성
# 데이터를 '입력: {input} 출력: {output}' 형식으로 변환
example_prompt = PromptTemplate(
    input_variables=["input", "output"], # 키 이름을 넣어 벨류를 참조할 수 있게함.
    template="입력: {input} 출력: {output}", # {벨류} 키 름에 해당하는 벨류 값이 들어감
)

# FewShotPromptTemplate을 생성
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                   # 예시 목록 데이터
    example_prompt=example_prompt,       # 예시 샘플 프롬프트 템플릿
    prefix="다음 단어의 반의어를 생성해 주세요.", # 전체 프롬프트의 시작 부분에 들어갈 지시문(접두사)
    suffix="입력: {user_input} 출력:",     # 전체 프롬프트의 끝 부분에 들어갈 실제 사용자 입력(접미사)
    input_variables=["user_input"],       # 사용자에게 받을 입력의 변수 이름을 정의
)

# 퓨샷 프롬프트 템플릿과 LLM을 연결하여 체인(Chain)을 생성
# LCEL(LangChain Expression Language)을 사용한 파이프라인 구성
chain = few_shot_prompt | llm

# 체인을 실행
user_word = "높은"
result = chain.invoke({"user_input": user_word})

# --- 결과 출력 ---

# 템플릿이 어떻게 최종 프롬프트로 변환되는지 확인
final_prompt = few_shot_prompt.format(user_input=user_word)
print("--- 최종 생성된 퓨샷 프롬프트 ---")
print(final_prompt)
print("------------------------")

# AI의 답변
print(f"입력 단어: {user_word}")
print(f"AI가 생성한 반의어: {result.content}")
```

<br>

실행 결과

```python
--- 최종 생성된 프롬프트 ---
다음 단어의 반의어를 생성해 주세요.

입력: 행복
출력: 불행

입력: 빠른
출력: 느린

입력: 뜨거운
출력: 차가운

입력: 높은
출력:
-------------------------

입력 단어: 높은
AI가 생성한 반의어: 낮은
```

1. `prefix`의 지시문으로 모델에 명령
2. `example` 목록에 있던 예시 데이터들이 `example_prompt`을 통해 예시 샘플로 만든다. 챗봇 모델은 ‘입력’이 주어지면 ‘출력’으로 반의어를 내놓는 패턴을 학습하게 된다.
3. 사용자의 인풋인 ‘높은’이 `suffix`가 추가되어 ‘입력: 높은’ 이라는 실제 질문으로 만든다.
4. 챗봇 모델은 학습한 패턴에 따라 `출력:` 다음에 “낮은” 이라는 단어를 생성할 확률이 높아진다.

<br>

FewShopPromptTemplate은 모델에게 구체적인 예시를 보여준다.

복잡하거나 특정한 형식의 태스크에 훨씬 안정적이고 정확한 답변을 받을 수 있게된다.

<br>

## Generation

지금 까지 만들었던 리트리버, 프롬프트, llm 을 파이프라인으로 만들어서 모델이 사용자의 질문의 답변을 생성한다.

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 리트리버 생성
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=query_gen_llm
)

# ChatPromptTemplate 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 주어진 '컨텍스트' 정보만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다.\n\n컨텍스트:\n{context}"),
    ("human", "질문: {question}")
])

# LLM 모델
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
)

# RAG 체인(Chain) 구성
# 유사도를 계산한뒤 상위 청크들을 가져와 합친다.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL을 사용한 RAG 파이프라인 구성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 체인 실행
query = "RAG가 뭐야?"
response = rag_chain.invoke(query)

print("질문:", query)
print("답변:", response)

# 결과 출력
질문: RAG가 뭐야?
답변: 검색 증강 생성(RAG)은 거대 언어 모델(LLM)의 정확성과 신뢰성을 높이기 위한 기술입니다. 이 방식은 응답을 만들기 전에 외부 지식 베이스에서 관련된 정보를 검색하여 활용합니다. 이를 통해 모델이 잘못된 정보를 생성하는 환각 현상을 줄일 수 있고, 기존 학습 데이터에 없는 새로운 주제에 대해서도 답변할 수 있게 됩니다.
```

LCEL(랭체인 표현식 언어)을 사용하여, 검색(Retrieval)과 생성(Generation)을 하나의 자동화된 파이프라인를 만든다.

<br>

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

1. **입력 처리**

`{"context": ..., "question": ...}` 딕셔너리 형태의 코드는 RunnableParallel이라고 불린다.

여러 작업을 동시에 병렬로 실행해서 다음단계로 넘긴다.

`chain.invoke(’사용자 질문’)` 이 실행되면 이부분이 가장 먼저 실행된다.

<br>

`RunnablePassthrough()`은 사용자의 질문을 그대로 전달한다.

`invoke`에 들어온 사용자 질문을 아무런 변경 없이 `question`이라는 키에 담아서 다음 단계로 전달한다.

<br>

`"context": retriever | format_docs`

invoke에 들어온 사용자 질문이 동시에 retriever에 전달된다.

`retriever` : 리트리버에서 질문과 가장 유사한 청크들을 벡터 스토어에서 검색해 Documents 객체들의 리스트를 반환한다.

`format_docs`: 리트리버가 찾아온 청크 리스트를 format_docs 함수로 넘겨준다. 이 함수는 리스트 안의 각 청크들을 하나의 긴 텍스트로 합쳐준다.

최종적으로 깔끔하게 정리된 텍스트가 `“context”`라는 키에 담긴다.

<br>

이단계가 끝나면 `{"context": "합쳐진 문서 내용...", "question": "사용자 질문"}` 딕셔너리가 만들어져 prompt 단계로 넘어간다.

<br>

1. **프롬프트 생성 `| prompt`**

앞 단게에서 만들어진 딕셔너리를 입력으로 받는다.

`ChatPromptTemplate`으로 만들어진 `prompt`는 이 딕셔너리의 값들을 자신의 템플릿에 있는 `{context}`와 `{question}` 변수 자리에 들어간다.

LLM이 이해할 수 있는 구조화된 프롬프트 객체가 만들어져 `llm` 단계로 넘어간다.

<br>

1. 답변 생성 `| llm`

prompt로 부터 완성된 프롬프트를 입력으로 받는다.

OpenAI의 `gpt-4o-mini`와 같은 거대 언어 모델(LLM)이 이 프롬프트를 기반으로 답변을 생성한다.

모델이 생성한 답변이 AIMessage와 같은 메시지 객체 형태로 `StrOutputParser` 단게로 넘어간다.

<br>

4. 출력 파싱: `| StrOutputParser()`

`StrOutputParser`는 출력 파서(Output Parser)의 한 종류이다.

`llm`이 생성한 복잡한 메시지 객체에서 사용자가 실제로 필요한 **순수 텍스트 내용(`content`)만 추출하는 역할**을 합니다.

사람이 읽을 수 있는 깔끔한 문자열(string) 형태의 최종 답변이 반환된다.

<br>

참고

- [랭체인 입문부터 응용까지](https://wikidocs.net/231154)
- [딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/288616)
- [Prompt Template 사용법 블로그](https://rudaks.tistory.com/entry/langchain-Prompt-Template-%EC%82%AC%EC%9A%A9%EB%B2%95)
