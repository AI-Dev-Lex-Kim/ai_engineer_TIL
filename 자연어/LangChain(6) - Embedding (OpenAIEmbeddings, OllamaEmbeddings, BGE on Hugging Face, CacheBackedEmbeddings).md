# LangChain(6) - Embedding (OpenAIEmbeddings, OllamaEmbeddings, BGE on Hugging Face, CacheBackedEmbeddings)

<br>

## 임베딩 모델이란?

임베딩 모델은 문장, 문단 같은 <mark>**raw data(원본 데이터)를 고정 길이의 숫자 벡터로 변환**</mark>한다.

이 벡터는 텍스트의 의미(sementic meaning)를 담고 있다.

단어의 일치 여부가 아니라 의미를 기준으로 <mark>**텍스트를 비교하고 검색**</mark>할 수 있게 된다.

<br>

### 임베딩된 벡터란?

임베딩된 벡터란, <mark>**텍스트의 의미가 벡터안에 숫자로 표현**</mark>됐다고 보면된다.

더 정확하게 보면, 이 숫자들은 <mark>**벡터 공간**</mark>에서 <mark>**자신의 의미에 따라 위치**</mark>가 하게된다.

비슷한 의미의 텍스트들은 <mark>**벡터 공간안에서 가까운 위치**</mark>에 놓이게된다.

<br>

## LangChain에서 Embedding

LangChain에서 RAG, Agent 같은 작업을 쉽게 할 수 있게 <mark>**임베딩을 추상화**</mark>했다.

<mark>**다양한 임베딩 모델**</mark>(OpenAI, Hugging Face.. 등)을 사용해도 <mark>**동일한 인터페이스(코드)**</mark>로 다룰수 있다는 점이다.

<br>

모든 <mark>**랭체인 임베딩 모델**</mark>은 두 개의 메인 메서드를 <mark>**공통적으로 사용**</mark>한다.

- `embed_documents`
  여러 문서를 임베딩
- `embed_query`
  쿼리(사용자 질문) 임베딩

<br>

핵심 설계의도 3가지

1. <mark>**쉬운 모델 교체**</mark>

   Open AI 임베딩 모델을 쓰다가 로컬 HuggingFace <mark>**임베딩 모델로 바꿔도**</mark>, 나머지 <mark>**코드는 동일하게 유지**</mark>할수있다.

   그래서 임베딩 생성을 표준 메서드 2개로 고정한것이다.

   <br>

2. <mark>**벡터스토어와 임베딩 모델 분리**</mark>

   벡터스토어(Chroma, FAISS 등)는 벡터 저장과 검색만 책임진다.

   <mark>**임베딩 모델은 “텍스트 → 벡터” 만 책임**</mark>진다.

   이 분리 메커니즘 덕분에, <mark>**Chroma를 FAISS로 바꾸거나 임베딩 모델만 바꾸는 조합이 쉬워진다.**</mark>(결합도 감소)

   <br>

3. <mark>**문서 임베딩과 쿼리 임베딩을 분리**</mark>

   많은 모델은 둘을 동일하게 처리하지만, 어떤 시스템은 <mark>**문서와 질문**</mark>에 서로 <mark>**다른 프롬프트 전처리를 적용**</mark>한다.

   그래서 `embed_documents` vs `embed_query`로 분리해둔다.

<br>

## 임베딩 캐싱(CacheBackedEmbeddings)

랭체인에서 임베딩을 보다 효율적으로 작업할수있게 <mark>**임베딩 캐싱 메서드**</mark>가 있다.

<br>

`CacheBackedEmbeddings`

- <mark>**임베딩 모델을 그대로 감싸**</mark>서, 이미 문서나 쿼리를 임베딩 했으면, <mark>**다시 계산하지 않고 캐시에서 꺼내**</mark>쓰도록 만든 래퍼이다.
  key-value 방식으로 저장하는 방식이다.

<br>

<mark>**설계의도**</mark>

1. <mark>**비용, 속도 최적화**</mark>

   임베딩 호출은 느리고 비용이 든다.

   같은 텍스트를 여러번 임베딩 하는 경우가 많아서, “<mark>**임베딩 벡터” 결과를 캐싱**</mark>해, RAG 인덱싱과 재인덱싱 시간이 크게 줄어든다.

   <br>

2. <mark>**벡터스토어와 역활 분리**</mark>

   벡터스토어는 검색과 인덱싱만 진행하고 문서 저장하는 역활을 하지 않는다.

   `CacheBackedEmbeddings`는 <mark>**임베딩 계산 결과를 재사용하는 저장소 역활이 핵심**</mark>이다.

   어떠한 벡터스토어(Chroma, FAISS)를 쓰던 임베딩 캐싱 모듈은 그대로 쓸수있다.

<br>

`CacheBackedEmbeddings`은 다음 과 같은 파라미터를 가진다.

- `from_bytes_store`
  임베딩 캐싱 코드를 `CacheBackedEmbeddings.from_bytes_store` 으로 시작한다.
  <br>

- document_embedding_cache
  임베딩된 문서 청크를 저장하는 `embed_documents`용 캐시 저장소이다.
  <br>

- batch_size
  스토어를 업데이트할때, 몇개의 문서를 캐시저장소에 저장할지 설정한다.
  기본값은 `None`이다.
  <br>

- namespace
  같은 문서 텍스트라도 모델이 다르면 벡터가 달라진다.
  namespace를 나누어서 잘못된 문서를 캐시저장소에서 가져오지 않게 방지한다.
  그래서 보통 모델명을 namespace로 둔다.
  <br>

- query_embedding_cache
  query도 캐싱할때 쓴다.
  True를 주면, <mark>**`document_embedding_cache`**</mark> 저장소에 저장한다.

<br>

<mark>**Chroma DB**</mark>와 같이 사용할때

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma

# 캐싱할 임베딩 모델
underlying_embeddings = ... # e.g., OpenAIEmbeddings(), HuggingFaceEmbeddings(), etc.

# 로컬저장소에 기존의 임베딩을 저장
# 프로덕션 단계에서는 사용하지 않지만, 로컬에서는 작업할때는 유용하다.
cache_dir = LocalFileStore("./cache/")
chroma_dir = LocalFileStore("./chroma_db") # Chroma db 폴더

# 문서
documents = [
    "파이썬은 인터프리터 언어이다",
    "자바는 JVM 위에서 동작한다",
    "LangChain은 LLM 애플리케이션 프레임워크이다",
]

# 메타데이터
metadatas = [
    {"source": "demo", "idx": 0},
    {"source": "demo", "idx": 1},
    {"source": "demo", "idx": 2},
]

# 저장소의 key로 사용할 중복되지 않을 id 생성
def make_id(text):
		h = hashlib.sha1(text.encode("utf-8")).hexdigest()
		return h[:12] # 짧게 잘라서 id로 사용		# 예) b54c82219cc8, e84eb7e8eb61

ids = [make_id(t) for t in documents]

# 캐싱 시작
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,                 # 캐싱할 임베딩
    store,                                 # 캐시 저장소
    namespace=underlying_embeddings.model, # 모델 충돌 방지
    query_embedding_cache=True             # query까지 store에 캐싱하겠다는 뜻
)

# Chroma 벡터스토어 생성
vector_store = Chroma(
    collection_name="demo_collection",      # 컬렉션 이름이다
    embedding_function=cached_embedder,     # 여기서 cached_embedder가 사용된다
    persist_directory=chroma_dir,           # Chrom db 폴더 경로
)

# 문서를 저장한다.
# 이 시점에 Chroma가 문서 임베딩을 위해서 embedding_function.embed_documents를 호출한다
# 이 시점에 CacheBackedEmbeddings는 저장되지 않은 문서만 저장한다.
vector_store.add_texts(
    texts=documents,          # 저장할 문서 텍스트이다
    metadatas=metadatas,      # 메타데이터
    ids=ids,                  # 고유 id로 중복 저장을 방지한다
)

# 사용자 질문
query = "파이썬이 뭐야?"

# 쿼리 임베딩이 필요해서 embed_query가 호출된다.
# query_embedding_cache=True이면 같은 쿼리를 반복할 때 캐시 hit가 날 수 있다
# 캐시저장소에 저장된 문서를 유사도를 검색해 빠르게 불러온다.
results = vector_store.similarity_search(query=query, k=2)

print("results", [d.page_content for d in results])

# 같은 쿼리를 한 번 더 검색한다.(관찰용)
results2 = vector_store.similarity_search(query=query, k=2)

```

<br>

## 대표적인 임베딩 모델 사용법

<mark>**OpenAIEmbeddings**</mark>

```bash
pip install -qU langchain-openai
```

```python
import getpass
import os

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("API key 입력")
```

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # `text-embedding-3` 임베딩 모델 사용
    # 임베딩 모델의 사이즈를 구체적으로 지정가능하다. 이번은 large 선택
    # 임베딩 모델을 반환한다.
    # 1024 차원
)
```

<br>

<mark>**OllamaEmbeddings**</mark>

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

```python
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(
    model="llama3",
)
```

<br>

<mark>**BGE on Hugging Face**</mark>

BGE 모델은 허깅페이스에서 <mark>**가장 성능이 좋은 오픈소스 임베딩 모델**</mark>이다.

BAAI(Beijing Academy of Artifical Intelligence) 에서 BGE 모델을 만들었다.

BAAI는 AI 연구와 개발하는 비영리단체이다.

<br>

```bash
pip install -qU  sentence_transformers
```

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
```

<br>

다양한 <mark>**다른 임베딩 모델**</mark>에 대해서는 [LangChian Docs 링크](https://docs.langchain.com/oss/python/integrations/text_embedding#top-integrations)으로 들어가면 자세히 알수있다.

<br>

참고

- https://docs.langchain.com/oss/python/integrations/text_embedding
- https://wikidocs.net/233777
- https://reference.langchain.com/python/langchain_core/embeddings
- https://docs.langchain.com/oss/python/integrations/text_embedding/openai
- https://docs.langchain.com/oss/python/integrations/text_embedding/ollama
- https://docs.langchain.com/oss/python/integrations/text_embedding/bge_huggingface
