# LangChain(9) - Reranker (BAAI/bge-reranker-v2-m3)

## Reranker란?

일반적으로 <mark>**사용자의 질문에 관련된 문서를 리트리버**</mark>가 가져오게된다.

이때, 임베딩 모델로 질문과 문서를 각각 따로 임베딩해서 유사도를 비교한뒤 관련된 문서를 가져온다.

<br>

이렇게 <mark>**질문과 문서를 따로 임베딩**</mark>하는 모델을 <mark>**Bi-encoder**</mark> 모델이라고 한다.

- A와 B를 각각 <mark>**독립적으로 인코딩**</mark> 한후, 유사도를 비교한다.
- 각자 독립적으로 인코딩을해, <mark>**속도가 매우 빠르다.**</mark>

<br>

Reranker는 <mark>**Cross-encoder**</mark> 모델로 동작한다.

- <mark>**A와 B를 한번에 인코딩**</mark> 한후, 모델 내부 Attention으로 <mark>**두 관계를 계산**</mark>한다.
- 두 문장을 곧 바로 비교하므로 <mark>**정확도가 매우 높다.**</mark>
- 하지만 매번 계산해야하므로 <mark>**연산 비용이 크다.**</mark>

<br>

그래서 2 step으로 더욱 정확성을 높이는 것이 Reranker이다.

1. 1단계로 bi-encoder인 리트리버로 수만~수백만 문서 중 <mark>**상위 k개를 빠르게 추린다.**</mark>
2. 2단계로 추려진 소수 문서중 cross encoder을 적용해, <mark>**가장 관련성 높은 결과를 재정렬한다.**</mark>

<br>

## HuggingFace Reranker

일반적으로 `[CLS] Query [SEP] Document [SEP]` 형태로 입력한다.

[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)는 다국어를 지원해주는 리랭커 모델이다.

```python
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 질문–문서 쌍을 동시에 입력받아 관련성 점수를 계산하는
# cross encoder 기반 리랭커 모델을 로드한다
# 설계 의도: 문서 임베딩 유사도가 아니라, 질문에 대한 직접적인 관련성을 정밀하게 평가한다
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# cross encoder 리랭커를 생성한다
# base_retriever가 가져온 문서들을 재정렬한 뒤, 상위 3개만 남긴다
# 설계 의도: 1차 검색의 recall은 유지하고, precision을 이 단계에서 끌어올린다
compressor = CrossEncoderReranker(model=model, top_n=3)

# ContextualCompressionRetriever를 생성한다
# 내부적으로는
# 1) base_retriever로 1차 검색을 수행하고
# 2) 검색 결과 문서를 cross encoder 리랭커로 재정렬·축소한다
# 설계 의도: 검색 + 리랭킹 과정을 하나의 retriever 인터페이스로 감싸기 위함이다
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 문서 압축 검색기를 호출한다.
# 실제 동작은 "다시 검색"이 아니라,
# 1차 검색 결과를 리랭킹해 상위 문서만 반환하는 과정이다
compressed_docs = compression_retriever.invoke("What is the plan for the economy?")

# 리랭킹 및 압축 이후 최종 선택된 문서를 출력한다
pretty_print_docs(compressed_docs)

```

<br>

Reference

- https://github.com/langchain-ai/docs/blob/main/src/oss/python/integrations/document_transformers/cross_encoder_reranker.mdx
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://wikidocs.net/253836
