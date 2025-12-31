- [LangChain(4) - PDF Document Loaders(PyPDF, Lazy Load, OCR)](#langchain4---pdf-document-loaderspypdf-lazy-load-ocr)
  - [PDF](#pdf)
  - [PyPDFLoader](#pypdfloader)
    - [Lazy Load](#lazy-load)
  - [PDF에서 이미지 추출](#pdf에서-이미지-추출)
    - [rapidOCR](#rapidocr)
    - [**tessearct**](#tessearct)
    - [**Multimodal model**](#multimodal-model)

# LangChain(4) - PDF Document Loaders(PyPDF, Lazy Load, OCR)

Documetn loader는 다양한 데이터를 읽어와서 LangChain의 Document 형식으로 변환한다.

<mark>**다양한 데이터를 Document 형식으로 변환**</mark>해 출처와 상관없이 <mark>**일관된 방식으로 데이터를 처리**</mark>한다.

<br>

Documetn Loader는 형식에 따라 다양하게 존재하지만, 공통적인 파라미터가 있다.

- `load()`
  - 모든 문서를 한번에 로드한다
- `lazy_load()`
  - 문서를 지연 로딩 방식으로 스트리밍한다.
  - 대용량 데이터셋을 다룰때 유용하다.

대표적으로 많이 사용하는 로더에 대해 알아보자.

<br>

## PDF

PDF는 Raw Data 중에서도 <mark>**가장 많이 사용되는 파일형식**</mark>이이다.

PDF는 RAG에서 데이터를 추출할때 난이도가 있는 편이다.

텍스트 뿐만아니라 <mark>**이미지나 테이블 형태도 포함**</mark>하고 있기 때문이다.

<br>

따라서 OCR과 테이블을 구조화형태로 불러올 수 있는 라이브러리를 결합해야 온전하게 PDF를 추출할 수 있다고 볼수있다.

<br>

대표적인 PDF 라이브러리를 알아보자.

<br>

## PyPDFLoader

설치

```bash
pip install langchain-community pypdf
```

<br>

사용방법

```bash
from langchain_community.document_loaders import PyPDFLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
docs[0]

import pprint

pprint.pp(docs[0].metadata)
```

<br>

결과

```bash
{'producer': 'pdfTeX-1.40.21',
 'creator': 'LaTeX with hyperref',
 'creationdate': '2021-06-22T01:27:10+00:00',
 'author': '',
 'keywords': '',
 'moddate': '2021-06-22T01:27:10+00:00',
 'ptex.fullbanner': 'This is pdfTeX, Version 3.14159265-2.6-1.40.21 (TeX Live '
                    '2020) kpathsea version 6.3.2',
 'subject': '',
 'title': '',
 'trapped': '/False',
 'source': './example_data/layout-parser-paper.pdf',
 'total_pages': 16,
 'page': 0,
 'page_label': '1'}
```

<br>

### Lazy Load

한번에 로드 하지않고, 하나씩 스트리밍 처럼 받아온다.

대용량 데이터에서 <mark>**메모리를 아끼고 중간중간 인덱싱 같은 작업**</mark>을 할때 유용하다.

```bash
pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:
        # 10개를 로드하면, 인덱싱을해 저장한다.
        index.upsert(pages) # 예시

        # 비운뒤 다음 10개를 로드한다.
        pages = []

# 마지막에 남은 것 처리 (필수)
if pages:
    index.upsert(pages)
```

<br>

## PDF에서 이미지 추출

한글 텍스트 추출하는 능력은 뛰어나지만 이미지 데이터는 추출하지 못한다.

보통 <mark>**이미지를 추출하는 방법**</mark>은 3가지 이다.

- <mark>**rapidOCR**</mark>
  가볍고 빠른 광학 문자 인식(OCR) 도구
- <mark>**Tesseract**</mark>
  높은 정확도를 제공하는 OCR 도구
- <mark>**멀티모달 언어 모델**</mark>
  이미지와 텍스트를 함께 처리할 수 있는 언어 모델

<br>

추출된 이미지 출력 형식을 HTML, Markdown, 텍스트 중에서 선택할수있다.

<br>

### rapidOCR

```bash
pip install -qU rapidocr-onnxruntime
```

```bash
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

print(docs[5].page_content)
```

<br>

### <mark>**tessearct**</mark>

```bash
pip install -qU pytesseract
```

한글을 인식하게 하기위해 <mark>**한국어 언어를 다운로드**</mark> 해야한다.

<mark>**kor.traineddata을 다운로드해 tessdata 폴더에 넣어준다.**</mark>

https://github.com/tesseract-ocr/tessdata

<br>

```bash
text = pytesseract.image_to_string('image.png', lang=lang, config=f'--tessdata-dir "{korean_data_path}"')

```

<br>

```bash
from langchain_community.document_loaders.parsers import TesseractBlobParser

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="html-img",
    images_parser=TesseractBlobParser(langs='kor'),
)
docs = loader.load()
print(docs[5].page_content)
```

`langs='kor'` 파라미터를 넣어주어야한다.

### <mark>**Multimodal model**</mark>

이미지를 읽을수있는 <mark>**모델을 불러와 인식**</mark>한다.

예를 들어 OpenAI 모델을 불러 OCR 작업을 하게 할 수있다.

```bash
%pip install -qU langchain-openai
```

<br>

```python
import os
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key =")
```

<br>

```python
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI

loader = PyPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
)
docs = loader.load()
print(docs[5].page_content)
```

<br>

참고

- https://docs.langchain.com/oss/python/integrations/document_loaders
- https://reference.langchain.com/python/langchain_core/document_loaders/?_gl=1*18cs8il*_gcl_au*NjkzNTc0Mjg0LjE3NjcwNjIyMzc.*_ga*MjQ5MDY5OTYwLjE3NjcwNjIyMzc.*_ga_47WX3HKKY2*czE3NjcwODU0MDMkbzQkZzEkdDE3NjcwOTE1MDgkajMxJGwwJGgw
- [https://velog.io/@autorag/PDF-한글-텍스트-추출-실험](https://velog.io/@autorag/PDF-%ED%95%9C%EA%B8%80-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%B6%94%EC%B6%9C-%EC%8B%A4%ED%97%98)
- https://reference.langchain.com/v0.3/python/community/document_loaders/langchain_community.document_loaders.parsers.images.TesseractBlobParser.html
