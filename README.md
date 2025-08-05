# ai_engineer_TIL

<br>

- [ai\_engineer\_TIL](#ai_engineer_til)
  - [NLP(Natural Language Processing)](#nlpnatural-language-processing)
    - [Hugging Face](#hugging-face)
  - [Image Generation](#image-generation)
  - [Pytorch](#pytorch)
  - [Pandas](#pandas)
  - [Numpy](#numpy)
  - [AI Basic](#ai-basic)
  - [Python](#python)
  - [etc](#etc)

<br>

## NLP(Natural Language Processing)

- [Bahdanau Attention Mechanism](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Bahdanau%20Attention%20Mechanism.md)
- [Bahdanau Attention 코드 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Bahdanau%20Attention%20%EC%BD%94%EB%93%9C%20%EB%B6%84%EC%84%9D.md)
- [Self-Attention, Mult-Head-Attention](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Self-Attention%2C%20Mult-Head-Attention.md)
- [BERT 주요 특징](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/BERT%20%EC%A3%BC%EC%9A%94%20%ED%8A%B9%EC%A7%95.md)
- [BERT 전체 과정](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/BERT%20%EC%A0%84%EC%B2%B4%20%EA%B3%BC%EC%A0%95.md)
- [LoRA(Low Rank Adaptation of LLM)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/LoRA(Low%20Rank%20Adaptation%20of%20LLM).md>)
- [skt koBERT Full Fine-Tuning LoRA 감정분석 모델 구현](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/skt%20koBERT%20Full%20Fine-Tuning%20LoRA%20%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EB%AA%A8%EB%8D%B8%20%EA%B5%AC%ED%98%84.md)
- [RAG Deep Dive with code](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/RAG(Retrieval-Agumented%20Generation)%20Deep%20Dive%20with%20code.md>)
- [MCP(Model Context Protocol)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/MCP(Model%20Context%20Protocol).md>)
- [OpenAI Responses vs Chat Completions](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Responses%20vs%20Chat%20Completions.md)
- [OpenAI Responses API](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Responses%20API.md)
- [OpenAI Conversation state(대화 상태 관리)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Conversation%20state(%EB%8C%80%ED%99%94%20%EC%83%81%ED%83%9C%20%EA%B4%80%EB%A6%AC).md>)
- [OpenAI LLM Supervised Fine-Tuing](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20LLM%20Supervised%20Fine-Tuing%20Search.md)
- [OpenAI Fine-Tuing 성능을 더 높이는 방법](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Fine-Tuing%20%EC%84%B1%EB%8A%A5%EC%9D%84%20%EB%8D%94%20%EB%86%92%EC%9D%B4%EB%8A%94%20%EB%B0%A9%EB%B2%95.md)
- [OpenAI LLM Fine-Tuning을 위한 데이터 준비 및 분석](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20LLM%20Fine-Tuning%EC%9D%84%20%EC%9C%84%ED%95%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A4%80%EB%B9%84%20%EB%B0%8F%20%EB%B6%84%EC%84%9D.md)
- [OpenAI Fine-Tuning API](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20Fine-Tuning%20API.md)
- [OpenAI 프롬프트 엔지니어링(Prompt Template)](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/OpenAI%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81(Prompt%20Template).md>)

<br>

### Hugging Face

- [Hugging Face load_dataset](<https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/dataset%20load_dataset().md>)
- [Hugging Face 사용법 with T5](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/hugging%20face%20T5%20%EC%82%AC%EC%9A%A9%EB%B2%95.md)
- [Hugging Face Dataset.map](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/%EC%9E%90%EC%97%B0%EC%96%B4/Hugging%20Face%20Dataset%20map.md)

<br>

## Image Generation

- [cGAN](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/Image%20Generation/cGAN.md)

<br>

## Pytorch

- [nn.BatchNorm2d](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/Pytorch/nn_BatchNorm2d.md)
- [Pytorch DataLoader DeepDive](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/Pytorch/Pytorch%20DataLoader%20DeepDive.md)

<br>

## Pandas

- [DataFrame 생성, 불러오기, 저장, 데이터 일부 처리](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/DataFrame%20%EC%83%9D%EC%84%B1%2C%20%EB%B6%88%EB%9F%AC%EC%98%A4%EA%B8%B0%2C%20%EC%A0%80%EC%9E%A5%2C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%9D%BC%EB%B6%80%20%EC%B2%98%EB%A6%AC.md)
- [datetime 타입으로 변경하는 방법 with pandas](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/datetime%20%ED%83%80%EC%9E%85%EC%9C%BC%EB%A1%9C%20%EB%B3%80%EA%B2%BD%ED%95%98%EB%8A%94%20%EB%B0%A9%EB%B2%95%20with%20pandas.md)
- [날짜와 시간 데이터 인덱싱 및 슬라이싱 방법](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/%EB%82%A0%EC%A7%9C%EC%99%80%20%EC%8B%9C%EA%B0%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%9D%B8%EB%8D%B1%EC%8B%B1%20%EB%B0%8F%20%EC%8A%AC%EB%9D%BC%EC%9D%B4%EC%8B%B1%20%EB%B0%A9%EB%B2%95.md)
- [Pandas 총정리](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/Pandas%20%EC%B4%9D%EC%A0%95%EB%A6%AC.md)
- [Pandas 총정리(간단)](<https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Pandas/Pandas%20%EC%B4%9D%EC%A0%95%EB%A6%AC(%EA%B0%84%EB%8B%A8).md>)

<br>

## Numpy

- [Numpy 기초문법](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/Numpy/Numpy%20%EA%B8%B0%EC%B4%88%EB%AC%B8%EB%B2%95.md)

<br>

## AI Basic

- [Precision, Recall, F1-Score](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/Precision%2C%20Recall%2C%20F1-Score%20.md)
- [Backpropagation, 경사 하강법 업데이트 Deep Dive](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/Backpropagation%2C%20%EA%B2%BD%EC%82%AC%20%ED%95%98%EA%B0%95%EB%B2%95%20%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8%20Deep%20Dive.md)
- [데이터 사이언스란?](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4%EB%9E%80%3F.md)
- [기초통계와 데이터 시각화(1) - 데이터 종류, pandas 시각화, 분산과 표준편차](<https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EA%B8%B0%EC%B4%88%ED%86%B5%EA%B3%84%EC%99%80%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94(1)%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A2%85%EB%A5%98%2C%20pandas%20%EC%8B%9C%EA%B0%81%ED%99%94%2C%20%EB%B6%84%EC%82%B0%EA%B3%BC%20%ED%91%9C%EC%A4%80%ED%8E%B8%EC%B0%A8.md>)
- [기초통계와 데이터 시각화(2) - 데이터 그래프 종류 및 시각화 with searborn](<https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EA%B8%B0%EC%B4%88%ED%86%B5%EA%B3%84%EC%99%80%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94(2)%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B7%B8%EB%9E%98%ED%94%84%20%EC%A2%85%EB%A5%98%20%EB%B0%8F%20%EC%8B%9C%EA%B0%81%ED%99%94%20with%20searborn.md>)
- [결정 트리 장단점, 부스팅 기법, 차원 축소 기법](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EA%B2%B0%EC%A0%95%20%ED%8A%B8%EB%A6%AC%20%EC%9E%A5%EB%8B%A8%EC%A0%90%2C%20%EB%B6%80%EC%8A%A4%ED%8C%85%20%EA%B8%B0%EB%B2%95%2C%20%EC%B0%A8%EC%9B%90%20%EC%B6%95%EC%86%8C%20%EA%B8%B0%EB%B2%95.md)
- [딥러닝-머신러닝관계 , 성능향상을 위한 하이퍼파라미터 종류](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/AI%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%20%EA%B8%B0%EC%B4%88%20%EC%A7%80%EC%8B%9D/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EA%B4%80%EA%B3%84%20%2C%20%EC%84%B1%EB%8A%A5%ED%96%A5%EC%83%81%EC%9D%84%20%EC%9C%84%ED%95%9C%20%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%20%EC%A2%85%EB%A5%98.md)

<br>

## Python

- [datetime](https://github.com/FE-Lex-Kim/ai_engineer_TIL/blob/main/python/datetime.md)

<br>

## etc

- [if \_\_name\_\_ == "\_\_main\_\_"](https://github.com/AI-Dev-Lex-Kim/ai_engineer_TIL/blob/main/etc/if%20__name__%20%3D%3D%20__main__.md)

<br>
