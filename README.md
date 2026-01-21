# AI Engineer TIL

<br>

- [AI Engineer TIL](#ai-engineer-til)
  - [1. 논문 리뷰](#1-논문-리뷰)
  - [2. NLP(Natural Language Processing)](#2-nlpnatural-language-processing)
    - [2.1 LangChain](#21-langchain)
    - [2.2 Agent(LangChain)](#22-agentlangchain)
    - [2.3 OpenAI](#23-openai)
    - [2.4 HuggingFace](#24-huggingface)
    - [2.5 이론공부](#25-이론공부)
  - [3. AI Basic](#3-ai-basic)

<br>

## 1. 논문 리뷰

- [[NLP 논문 리뷰] Attention is all you need(Transformer)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0/%5BNLP%20%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0%5D%20Attention%20is%20all%20you%20need(Transformer).md>)
- [[CV 논문 리뷰]What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0/%5BCV%20%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0%5DWhat%20is%20YOLOv8%3A%20An%20In-Depth%20Exploration%20of%20the%20Internal%20Features%20of%20the%20Next-Generation%20Object%20Detector.md)
- [[CV 논문 리뷰]You Only Look Once: Unified, Real-Time Object Detection](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0/%5BCV%20%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0%5DYou%20Only%20Look%20Once%3A%20Unified%2C%20Real-Time%20Object%20Detection.md)
- [[CV 논문 리뷰]Squeeze-and-Excitation Networks](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0/%5BCV%20%EB%85%BC%EB%AC%B8%20%EB%A6%AC%EB%B7%B0%5DSqueeze-and-Excitation%20Networks.md)

<br>

## 2. NLP(Natural Language Processing)

### 2.1 LangChain

- [LangChain(1) - 사용하는 이유, RAG 워크플로우](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(1)%20-%20%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94%20%EC%9D%B4%EC%9C%A0%2C%20%EC%84%A4%EC%B9%98%2C%20%EA%B7%B8%EB%A6%BC%EC%9C%BC%EB%A1%9C%20%EC%84%A4%EB%AA%85%ED%95%98%EB%8A%94%20RAG%20%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C%EC%9A%B0.md>)
- [LangChain(2) - Message (SystemMessage, HumanMessage, AIMessage)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangaChain(2)%20-%20SystemMessage%2C%20HumanMessage%2C%20AIMessage.md>)
- [LangChain(3) - Prompt Template (PromptTemplate, ChatPromptTemplate, MessagesPromptTemplate)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(3)%20-%20PromptTemplate%2C%20ChatPromptTemplate%2C%20MessagesPromptTemplate.md>)
- [LangChain(4) - Document Loaders (PyPDF, Lazy Load, OCR)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(4)%20-%20PDF%20Document%20Loaders(PyPDF%2C%20Lazy%20Load%2C%20OCR).md>)
- [LangChain(5) - Text Splitters (Chunking)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(5)%20-%20Text%20Splitters(Chunking).md>)
- [LangChain(6) - Embedding (OpenAIEmbeddings, OllamaEmbeddings, BGE on Hugging Face)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(6)%20-%20Embedding%20(OpenAIEmbeddings%2C%20OllamaEmbeddings%2C%20BGE%20on%20Hugging%20Face%2C%20CacheBackedEmbeddings).md>)
- [LangChain(7) - Vector Store(Chroma, FAISS)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(7)%20-%20Vector%20Store(Chroma%2C%20FAISS).md>)
- [LangChain(8) - Retriever (MultiQueryRetriever, MultiVectorRetriever, Long-Context Reorder)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(8)%20-%20Retriever%20(MultiQueryRetriever%2C%20MultiVectorRetriever%2C%20Long-Context%20Reorder).md>)
- [LangChain(9) - Models (invoke, stream, batch, schema, reasoning, Ollama)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(9)%20-%20Models%20(invoke%2C%20stream%2C%20batch%2C%20schema%2C%20reasoning%2C%20Ollama).md>)
- [LangChain(10) - Reranker (BAAI/bge-reranker-v2-m3)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LangChain(10)%20-%20Reranker%20(bge-reranker-v2-m3).md>)

<br>

### 2.2 Agent(LangChain)

- [Agent(1) - Structured output (ProviderStrategy, ToolStrategy)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Agent(1)%20-%20Structured%20output.md>)
- [Agent(2) - Tools (ToolRuntime)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Agent(2)%20-%20Tools%20(ToolRuntime).md>)
- [Agent(3) - Short-term memory (Delete, Trim, Summarize)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Agent(3)%20-%20Short-term%20memory%20(Delete,%20Trim,%20Summarize).md>)
- [Agent(4) - MCP (MultiServerMCPClient, FastMCP)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Agent(5)%20-%20MCP%20(MultiServerMCPClient%2C%20FastMCP).md>)

<br>

### 2.3 OpenAI

- [OpenAI(1) - Responses vs Chat Completions](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Responses%20vs%20Chat%20Completions.md)
- [OpenAI(2) - Responses API](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Responses%20API.md)
- [OpenAI(3) - Conversation state(대화 상태 관리)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Conversation%20state(%EB%8C%80%ED%99%94%20%EC%83%81%ED%83%9C%20%EA%B4%80%EB%A6%AC).md>)
- [OpenAI(4) - LLM Supervised Fine-Tuing](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20LLM%20Supervised%20Fine-Tuing%20Search.md)
- [OpenAI(5) - Fine-Tuing 성능을 더 높이는 방법](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Fine-Tuing%20%EC%84%B1%EB%8A%A5%EC%9D%84%20%EB%8D%94%20%EB%86%92%EC%9D%B4%EB%8A%94%20%EB%B0%A9%EB%B2%95.md)
- [OpenAI(6) - LLM Fine-Tuning을 위한 데이터 준비 및 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20LLM%20Fine-Tuning%EC%9D%84%20%EC%9C%84%ED%95%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A4%80%EB%B9%84%20%EB%B0%8F%20%EB%B6%84%EC%84%9D.md)
- [OpenAI(7) - Fine-Tuning API](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Fine-Tuning%20API.md)
- [OpenAI(8) - 프롬프트 엔지니어링(Prompt Template)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81(Prompt%20Template).md>)
- [OpenAI(9) - 프롬프트 엔지니어링 체크리스트](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81%20%EC%B2%B4%ED%81%AC%EB%A6%AC%EC%8A%A4%ED%8A%B8.md)
- [OpenAI(10) - Evaluating model performance](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Evaluating%20model%20performance.md)
- [OpenAI(11) - Evaluating model code](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Evaluating%20model%20code.md)

<br>

### 2.4 HuggingFace

- [Hugging Face(1) - T5 모델 구현](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/hugging%20face%20T5%20%EC%82%AC%EC%9A%A9%EB%B2%95.md)
- [Hugging Face(2) - load_dataset](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/dataset%20load_dataset().md>)
- [Hugging Face(3) - Dataset.map](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Hugging%20Face%20Dataset%20map.md)
- [Hugging Face(4) - local download to Cache dir](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Hugging%20Face%20local%20download%20to%20Cache%20dir.md)
- [Hugging Face(5) - nn.Module로 커스텀 모델 구현 방법](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/nn.Module%EB%A1%9C%20Hugging%20Face%20%EB%AA%A8%EB%8D%B8%20%EC%BB%A4%EC%8A%A4%ED%85%80.md)

<br>

### 2.5 이론공부

- [CLM(Causal Language Modeling)이란? (인과적 언어 모델링): 데이터셋/학습/추론](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Causal%20language%20modeling%EC%9D%B4%EB%9E%80%3F%20(%EC%9D%B8%EA%B3%BC%EC%A0%81%20%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8%EB%A7%81)%3A%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B_%ED%95%99%EC%8A%B5_%EC%B6%94%EB%A1%A0.md>)
- [MLM(Masked Language Modeling)이란? (마스크 언어 모델링): 데이터셋/학습/추론](<https://github.com/AI-Dev-LILex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Masked%20Language%20Modeling%EC%9D%B4%EB%9E%80%3F%20(%EB%A7%88%EC%8A%A4%ED%81%AC%20%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8%EB%A7%81)%3A%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B_%ED%95%99%EC%8A%B5_%EC%B6%94%EB%A1%A0.md>)
- [LoRA(Low Rank Adaptation of LLM)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LoRA(Low%20Rank%20Adaptation%20of%20LLM).md>)
- [MCP(Model Context Protocol)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/MCP(Model%20Context%20Protocol).md>)
- [Self-Attention 구현 코드 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Self-Attention%2C%20Mult-Head-Attention.md)
- [Bahdanau Attention Mechanism 구현 코드 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Bahdanau%20Attention%20Mechanism.md)
- [BERT(1) - 주요 특징](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/BERT%20%EC%A3%BC%EC%9A%94%20%ED%8A%B9%EC%A7%95.md)
- [BERT(2) - [CLS] 토큰 이해하기](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/BERT%EC%9D%98%20%5BCLS%5D%20%ED%86%A0%ED%81%B0%20%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0.md)
- [BERT(3) - 모델 코드 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/BERT%20%EC%A0%84%EC%B2%B4%20%EA%B3%BC%EC%A0%95.md)
- [BERT(4) - skt koBERT Full Fine-Tuning LoRA 감정분석 모델 구현](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/skt%20koBERT%20Full%20Fine-Tuning%20LoRA%20%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EB%AA%A8%EB%8D%B8%20%EA%B5%AC%ED%98%84.md)

<br>

## 3. AI Basic

- [Precision, Recall, F1-Score](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/Precision%2C%20Recall%2C%20F1-Score%20.md)
- [Backpropagation Deep Dive](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/Backpropagation%2C%20%EA%B2%BD%EC%82%AC%20%ED%95%98%EA%B0%95%EB%B2%95%20%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8%20Deep%20Dive.md)
- [nn.BatchNorm2d 코드 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/Pytorch/nn_BatchNorm2d.md)
- [Pytorch DataLoader 코드 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/Pytorch/Pytorch%20DataLoader%20DeepDive.md)
- [기초통계와 데이터 시각화(1) - 데이터 종류, pandas 시각화, 분산과 표준편차](<https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EA%B8%B0%EC%B4%88%ED%86%B5%EA%B3%84%EC%99%80%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94(1)%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A2%85%EB%A5%98%2C%20pandas%20%EC%8B%9C%EA%B0%81%ED%99%94%2C%20%EB%B6%84%EC%82%B0%EA%B3%BC%20%ED%91%9C%EC%A4%80%ED%8E%B8%EC%B0%A8.md>)
- [기초통계와 데이터 시각화(2) - 데이터 그래프 종류 및 시각화 with searborn](<https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EA%B8%B0%EC%B4%88%ED%86%B5%EA%B3%84%EC%99%80%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94(2)%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B7%B8%EB%9E%98%ED%94%84%20%EC%A2%85%EB%A5%98%20%EB%B0%8F%20%EC%8B%9C%EA%B0%81%ED%99%94%20with%20searborn.md>)
- [Pandas 총정리](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/Pandas%20%EC%B4%9D%EC%A0%95%EB%A6%AC.md)
- [Pandas 총정리(간단)](<https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/Pandas%20%EC%B4%9D%EC%A0%95%EB%A6%AC(%EA%B0%84%EB%8B%A8).md>)
