# LangChain(7) - Vector Store(Chroma, FAISS)

![image.png](<../images/langchain(7)/1.png>)

Vector Store는 임베딩한 문서 벡터를 저장하고, 질의(query) 벡터와 유사도를 계산해 문서를 빠르게 찾아주는 모듈이다.

LangChain은 <mark>**VectorStore라는 클래스로 추상화**</mark>해서 Chroma, FAISS 같은 <mark>**벡터스토어를 바뀌어껴도 일관성 있게 동작**</mark>한다.

<br>

## 통합 인터페이스

VectorStore <mark>**인터페이스를 통일**</mark>해서 아래와 같은 <mark>**공통적인 메서드를 사용**</mark>한다.

- <mark>**add_documents**</mark>
  저장소에 문서를 추가한다.
  ```python
  vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])
  ```
- <mark>**delete**</mark>
  ID로 스토어에 저장된 문서를 제거한다.
  ```python
  vector_store.delete(ids=["id1"])
  ```
- <mark>**similarity_search**</mark>
  query 텍스트를 임베딩해서 가장 유사한 문서를 반환한다.
  ```python
  similar_docs = vector_store.similarity_search("your query here")
  ```

<br>

대표적인 벡터스토어는 Chroma, FAISS가 있다.

<br>

## Chroma DB

Chroma는 문서 검색뿐만 아니라 <mark>**저장까지도 하는 점**</mark>이 핵심이다.

데이터를 저장할때는 3가지를 같이 저장한다.

- <mark>**문서 원문**</mark>
- <mark>**임베딩 벡터**</mark>
- <mark>**메타데이터(id 포함)**</mark>

<br>

<mark>**데이터 관리**</mark>까지 맡아서 애플리케이션 <mark>**코드를 단순화하는것이 Chroma의 특징**</mark>이다.

<br>

Chroma를 선택하는 주요 특징

- <mark>**문서 + 메타데이터까지 한번에 관리**</mark>하고 싶을때
- <mark>**persist(재시작해도 유지)**</mark>가 좋음
- <mark>**메타데이터 필터링**</mark>이 있는 검색이 있을때 구현이 단순함
- 팀이 인프라가 크게 갖추지 않은 <mark>**초기 단계 경우**</mark>

<br>

설치

```python
pip install -qU "langchain-chroma"
```

<br>

### LangChain Chroma 시작

LangChain에서 추상화한 Chroma로 연결해 시작한다.

<br>

로컬 메모리에 저장(in Memory, 데이터 일시적 저장)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)
```

<br>

영속성을 가진 로컬 데이터 저장(in Persistence, 시스템이 종료 또는 재부팅되도 상태가 유지)

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
```

<br>

Chroma Server 연결

크로마 서버를 로컬에서 운영중일때(`chroma run`), `host=’localhost’` 로 연결한다.

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    host="localhost",
)
```

Chroma Cloud

Chroma 클라우드를 사용하는 경우

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)
```

<br>

### Client Chroma 시작

사용자가 Chromadb를 직접 만들고 LangChain에 주입하는 방식이다.

이 경우는 저수준의 추상화로 직접 개발자의 의도에 맞게 더욱 관리하기 쉽게할수있다.

<br>

특히 팀 프로젝트에서 DB연결, 인증, 라우팅 설정을 한곳에서 관리하는 편이 유지보수에 좋다.

<br>

로컬 메모리에 저장

```python
import chromadb

client = chromadb.Client()
```

<br>

영속성을 가진 로컬 데이터 저장

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_langchain_db")
```

<br>

Chroma Server 연결

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000, ssl=False)
```

<br>

Chroma Cloud 연결

```python
import chromadb

client = chromadb.CloudClient()
```

<br>

Vectorstore 생성

```python
vector_store_from_client = Chroma(
    client=client,
    collection_name="collection_name",
    embedding_function=embeddings,
)
```

<br>

### 문서 추가

`add_documnets` 메서드를 사용해 데이터를 벡터스토어에 추가한다.

```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```

<br>

### 문서 업데이트

`updatge_documetns` 메서드를 사용해 저장소에 있는 기존의 문서를 업데이트할수있다.

```python
updated_document_1 = Document(
    page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)

vector_store.update_document(document_id=uuids[0], document=updated_document_1)
# You can also update multiple documents at once
vector_store.update_documents(
    ids=uuids[:2], documents=[updated_document_1, updated_document_2]
)
```

<br>

### 문서 삭제

벡터 스토어에서 문서를 삭제할수있다.

```python
vector_store.delete(ids=uuids[-1])
```

<br>

### 관련된 문서 검색

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

<br>

점수 포함 관련 문서 검색

```python
results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
```

<br>

### Retriever 생성

<mark>**VectorStore**</mark>는 후보 문서를 추출 <mark>**기능 담당**</mark>

<mark>**Retriever**</mark>은 후보 문서를 어떻게 가져올지 <mark>**검색 전략 담당**</mark>

<br>

MMR 검색 전략 Retriever

```python
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
```

<br>

기준 점수 Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8},
)
```

<br>

더 다양한 검색 전략 파라미터는 [이곳 API Reference](https://reference.langchain.com/python/integrations/langchain_chroma/?_gl=1*13rjtam*_gcl_au*NjkzNTc0Mjg0LjE3NjcwNjIyMzc.*_ga*MjQ5MDY5OTYwLjE3NjcwNjIyMzc.*_ga_47WX3HKKY2*czE3Njc1OTE5NTgkbzE3JGcxJHQxNzY3NTkyMDgyJGo2MCRsMCRoMA..) 에서 찾을 수 있다.

<br>

## FAISS(Facebook AI Similarity Search)

FAISS는 <mark>**벡터 검색에 최적화된 라이브러리**</mark> 이다.

벡터들을 어떻게 <mark>**빠르게 검색할지에 집중**</mark>을 한다.

문서나 메타데이터 저장의 주 역활은 FAISS가 하기보다, <mark>**다른 별도의 저장구조와 같이 묶어서 사용**</mark>한다.

<br>

정리하면, 빠른 검색 인덱스 엔진 역활을 맡고, 문서 관리는 다른 계층이 맡는 구조로 설계했다.

<br>

[FAISS Docs](https://github.com/facebookresearch/faiss/wiki)는 굉장히 자세히 FAISS 사용법을 알려준다.

<br>

FAISS 선택하는 주요 특징

- 빠르고 가볍게 벡터 검색을 할때
- 검색만 전문으로 맡기고, 나머지 데이터 관리는 별도 시스템에 맞기는 아키텍처일때

<br>

설치

gpu를 사용하면 `faiss-gpu`로 변경하면 된다.

```bash
pip install -qU langchain-community faiss-cpu
```

<br>

모델을 자동 추척할때, LangSmith을 사용하는것이 가장 좋다.

```python
os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

<br>

### 인덱스 생성

FAISS는 <mark>**Index 객체를 중심으로 설계**</mark>되어 있다.

- 벡터들의 집합을 <mark>**캡슐화**</mark>
- 검색이 효율적으로 되도록 벡터들을 <mark>**전처리**</mark>

<br>

<mark>**인덱스 종류는 여러가지**</mark>가 있다. 다른 다양한 인덱스는 [이곳 Docs](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)에서 확인 할 수 있다.

아래 예제들은 `IndexFlatL2`을 사용한다.

- 가장 단순한 버전으로 브루트포스 방식으로 L2거리 검색을 수행한다.

<br>

모든 인덱스는 생성될때 <mark>**벡터의 차원(d)**</mark>을 알아야한다.

대부분의 인덱스는 <mark>**훈련(training) 단계가 필요**</mark>하다.

- 이 단계에서 <mark>**벡터들의 분포를 분석**</mark>함.
- IndexFlatL2는 이 과정을 건너뛸 수 있음.

<br>

인덱스가 생성된뒤 훈련이 끝나면, 인덱스에 대해 <mark>**두가지 연산**</mark>을 수행 할수있다.

- add: 문서 추가
- search: 문서 검색

<br>

인덱스는 두 가지 <mark>**상태 변수**</mark>를 확인할 수 있다.

- is_trained: <mark>**훈련이 필요한지**</mark> 여부 불리언값
- ntotal: 인덱싱이 되어있는 <mark>**벡터개수**</mark>

<br>

일부 인덱스는 각 벡터에 대응하는 <mark>**정수 ID를 함께 저장 가능**</mark>하다.

- IndexFlatL2는 해당되지 않음
- ID를 제공하지 않으면 add는 <mark>**벡터의 순번을 ID로 사용**</mark>한다.
  첫번째 벡터는0, 두번째 벡터는1, 이런식이다.

<br>

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

<br>

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# 임베딩 차원수 파라미터로 전달
# d: 차원
# faiss,IndexFlatL2(d)
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

<br>

### 문서 추가

```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```

<br>

### 문서 삭제

```python
vector_store.delete(ids=[uuids[-1]])
```

<br>

### 문서 검색

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

<br>

MongoDB query와 operation을 지원해서 Metadata 필터링을 더 고급옵션으로 설정할 수 있다.

- <mark>**`$eq`**</mark> (equals)
- <mark>**`$neq`**</mark> (not equals)
- <mark>**`$gt`**</mark> (greater than)
- <mark>**`$lt`**</mark> (less than)
- <mark>**`$gte`**</mark> (greater than or equal)
- <mark>**`$lte`**</mark> (less than or equal)
- <mark>**`$in`**</mark> (membership in list)
- <mark>**`$nin`**</mark> (not in list)
- <mark>**`$and`**</mark> (all conditions must match)
- <mark>**`$or`**</mark> (any condition must match)
- <mark>**`$not`**</mark> (negation of condition)

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": {"$eq": "tweet"}},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

<br>

점수 검색

```python
results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
```

<br>

### Retriever 생성

```python
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
```

<br>

### 벡터 스토어 저장 및 로딩

FAISS index를 저장하고 로드해서, 매번 벡터스토어를 다시 생성하지 않아도 된다.

```python
vector_store.save_local("faiss_index")

new_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

docs = new_vector_store.similarity_search("qux")
```

<br>

### Merging

두개의 FAISS 벡터스토어를 합칠 수 있다.

```python
db1 = FAISS.from_texts(["foo"], embeddings)
db2 = FAISS.from_texts(["bar"], embeddings)

db1.merge_from(db2)
```

<br>

## FAISS vs Chroma 선택 기준

핵심 차이

- FAISS
  - <mark>**유사도 검색, 클러스터링**</mark>을 위한 라이브러리
  - 정확 검색, IVF, PQ, HNSW 등 <mark>**다양한 인덱스 검색**</mark>
  - 속도, 정확도, 메모리 트레이드오프를 <mark>**직접 고를수있음**</mark>
  - <mark>**GPU 인덱스**</mark> 제공
- Chroma
  - 컬렉션 단위로 <mark>**문서, 임베딩과 메타데이터를 같이 관리**</mark>
  - 쿼리시 메타데이터 기반 필터링 같은 <mark>**DB 스러운 기능 제공**</mark>

<br>

선택 기준

- <mark>**성능, 규모 최우선(FAISS)**</mark>
  - FAISS가 하드코어 성능, 대규모에 강점
- <mark>**개발 속도, 제품 기능(Chroma)**</mark>
  - 메타데이터 필터, CRUD, 로컬 지속성은 Chroma가 빠르게 DB 처럼 구현할 수 있음
- 운영 형태
  - 인덱스 , 스토리지를 직접 설계, <mark>**운영 의지 있으면 FAISS**</mark>
  - 앱 내부에 임베딩 저장소를 <mark>**간단히 포함**</mark> 하고 싶으면 <mark>**Chroma**</mark>

<br>

참고

- https://docs.langchain.com/oss/python/integrations/vectorstores
- https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
- https://github.com/facebookresearch/faiss/wiki
- https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
