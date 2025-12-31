# OpenAI <mark>**Evaluating model performance**</mark>

## 평가지표 종류

<mark>**Generation 평가지표**</mark>

- <mark>**충실성 (Faithfulness)**</mark>: 생성된 답변이 검색된 <mark>**문맥(Context)에 얼마나 충실한지를 평가**</mark>한다. 즉, 답변이 검색된 문서에 근거하여 생성되었는지, 환각(Hallucination) 없이 사실에 부합하는지를 본다.
- <mark>**답변 관련성 (Answer Relevancy)**</mark>: 생성된 답변이 사용자 <mark>**질문과 얼마나 관련성이 높은지를 평가**</mark>한다. 질문의 의도를 잘 파악하여 적절한 답변을 생성했는지를 본다.
- <mark>**독립성 (Groundness)**</mark>: 생성된 답변이 오직 제공된 문맥(retrieved context)에만 기반하여 생성되었는지, 외부 지식이나 환각이 없는지를 평가한다. 충실성과 유사한 개념이다.
- <mark>**언어 모델 기반 평가 (LLM-as-a-judge)**</mark>: LLM 자체를 평가 도구로 사용하여 생성된 답변의 품질(정확성, 유용성, 자연스러움 등)을 평가하는 방식이다.
- <mark>**ROUGE, BLEU, METEOR**</mark>: 요약, 기계 번역 등에서 사용되는 전통적인 텍스트 유사성 지표로, 생성된 답변과 참조 답변(Ground Truth) 간의 단어 또는 구문 일치도를 측정한다.

<br>

<mark>**종합 시스템 평가 지표 및 프레임워크**</mark>

Retrival 과 Generation 두 요소의 시너지를 종합적으로 평가하는 지표들이다.

- <mark>**전체적인 유용성 (Overall Utility)**</mark>: 시스템이 사용자 질문에 대해 최종적으로 얼마나 도움이 되는 답변을 제공했는지 평가한다. 이는 주로 사람의 평가(Human Evaluation)를 통해 이루어진다.
- <mark>**RAGAS (Retrieval Augmented Generation Assessment)**</mark>: RAG 시스템의 정량적 평가를 지원하는 오픈소스 프레임워크이다. 위에서 언급된 Faithfulness, Answer Relevancy, Context Precision, Context Recall 등의 여러 지표를 통합하여 RAG 파이프라인의 각 구성 요소를 세밀하게 분석할 수 있도록 돕는다.
- <mark>**Hit Rate**</mark>: RAG 시스템이 사용자가 찾던 정답(혹은 근접한 정답)을 포함하는 답변을 얼마나 자주 제공했는지 측정한다.
- <mark>**시스템 성능 (System Performance)**</mark>: 응답 속도(Latency), 토큰 사용량(Token Consumption) 등 시스템의 효율성과 비용 측면도 중요한 평가 지표가 된다.

---

<br>

# 모델 성능 평가하기

평가를 통해 모델 출력을 테스트하고 개선한다.

평가(종종 <mark>**evals**</mark>라고도 불린다)는 모델 출력이 <mark>**사용자가 지정하는 스타일 및 콘텐츠 기준을 충족하는지 테스트**</mark>한다.

특히 모델을 업그레이드하거나 새로운 모델을 시도할 때, LLM 애플리케이션의 성능이 기대에 부응하는지 파악하기 위해 <mark>**평가를 작성하는 것은 신뢰할 수 있는 애플리케이션을 구축하는 데 필수적인 요소**</mark>이다.

이 가이드에서는 [\*\*Evals API](https://platform.openai.com/docs/api-reference/evals)를 사용\*\*하여 평가를 프로그래밍 방식으로 구성하는 데 중점을 둔다.

원한다면 OpenAI 대시보드에서도 평가를 구성할 수 있다.

<br>

대략적으로 LLM 애플리케이션에 대한 평가를 구축하고 실행하는 데에는 세 가지 단계가 있다.

1. 수행할 작업을 <mark>**평가로써 설명**</mark>하기
2. <mark>**테스트 입력(프롬프트와 입력 데이터)으로 평가를 실행**</mark>하기
3. 결과를 분석하고 <mark>**프롬프트를 반복하여 개선**</mark>하기

이 과정은 시스템을 구현하고 테스트하기 전에 <mark>**시스템이 어떻게 동작해야 하는지 먼저 명시하는 BDD(behavior-driven development)와 다소 유사**</mark>하다.

<mark>**Evals API**</mark>를 사용하여 위 각 단계를 어떻게 완료하는지 알아보자.

<br>

## 작업에 대한 평가 생성하기

평가를 생성하는 것은 모델이 수행할 작업을 설명하는 것부터 시작한다.

모델을 사용하여 IT 지원 티켓의 내용을 <mark>**Hardware**</mark>, <mark>**Software**</mark>, 또는 <mark>**Other**</mark> 세 가지 범주 중 하나로 분류하고 싶다고 가정해 보자.

이 사용 사례를 구현하기 위해 <mark>**Chat Completions API**</mark> 또는 <mark>**Responses API**</mark>를 사용할 수 있다. 아래 두 예시는 모두 개발자 메시지와 지원 티켓 텍스트가 포함된 사용자 메시지를 결합한다.

<br>

IT 지원 티켓 분류하기

```python
from openai import OpenAI

client = OpenAI()

instructions = """You are an expert in categorizing IT support tickets. Given the supportticket below, categorize the request into one of "Hardware", "Software",or "Other". Respond with only one of those words."""
ticket = "My monitor won't turn on - help!"

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "developer", "content": instructions},
        {"role": "user", "content": ticket},
    ],
)

print(response.output_text)
```

이 동작을 API를 통해 테스트하기 위한 평가를 설정해 보자.

평가에는 두 가지 핵심 요소가 필요하다.

- `data_source_config`: 평가와 함께 사용할 <mark>**테스트 데이터에 대한 스키마.**</mark>
- `testing_criteria`: 모델 출력이 올바른지 여부를 결정하는 [<mark>**그레이더**</mark>.](https://platform.openai.com/docs/guides/graders)
  - 그레이더(grader)는 Evals API에서 <mark>**모델의 출력이 올바른지 여부를 평가하는 핵심 컴포넌트**</mark>이다.
  - 이는 모델의 <mark>**응답과 '정답'(ground truth)을 비교**</mark>하여 점수나 합격/불합격 여부를 결정한다.
  - 그레이더의 종류에는 <mark>**문자열 일치, 정규 표현식, 특정 키워드 포함 여부 등을 검사하는 방법**</mark>이 있다.

<br>

평가 생성하기

```python
from openai import OpenAI

client = OpenAI()

# evals.create 함수를 호출하여 새로운 평가(evaluation) 작업을 생성한다.
eval_obj = client.evals.create(
    name="IT Ticket Categorization",  # string | 평가 작업의 고유하고 사람이 읽을 수 있는 이름이다.
    data_source_config={
        "type": "custom",  # string | 데이터 소스의 유형을 지정한다. 'custom'은 사용자가 직접 데이터를 제공하는 방식이다.
        "item_schema": {
            "type": "object",  # string | 각 데이터 항목이 객체(object)임을 나타낸다.
            "properties": {
                # 'ticket_text'라는 키에 문자열 타입의 값을 가진다.
                # 평가할 모델의 입력으로 사용되는 IT 티켓 텍스트이다.
                "ticket_text": {"type": "string"},
                # 'correct_label'라는 키에 문자열 타입의 값을 가진다.
                # 'ticket_text'에 대한 정답 레이블이다. 모델의 출력과 비교된다.
                "correct_label": {"type": "string"},
            },
            # 각 데이터 항목에 'ticket_text'와 'correct_label' 필드가 반드시 포함되어야 함을 명시한다.
            "required": ["ticket_text", "correct_label"],
        },
        # boolean | True로 설정 시, 데이터 소스 설정에 샘플 스키마가 포함된다.
        "include_sample_schema": True,
    },
    # list | 평가 작업의 성공/실패 여부를 판단하는 기준(criteria) 목록이다.
    testing_criteria=[
        {
            "type": "string_check",  # string | 평가 기준의 유형을 지정한다. 'string_check'는 문자열 비교를 수행한다.
            "name": "Match output to human label",  # string | 평가 기준의 이름이다.
            # string | 평가할 대상 문자열이다.
            # '{{ sample.output_text }}'는 모델의 출력 텍스트를 나타내는 템플릿 변수이다.
            "input": "{{ sample.output_text }}",
            "operation": "eq",  # string | 수행할 비교 연산이다. 'eq'는 'equal'을 의미하며, 값이 동일한지 확인한다.
            # string | 비교 대상이 되는 문자열이다.
            # '{{ item.correct_label }}'는 데이터 소스에서 제공된 정답 레이블을 나타내는 템플릿 변수이다.
            "reference": "{{ item.correct_label }}",
        }
    ],
)

print(eval_obj)  # 생성된 평가 객체(eval_obj)의 세부 정보를 출력한다.
```

파라미터

<mark>**설명: `data_source_config` 매개변수**</mark>

이 평가를 실행하려면 프롬프트가 작동할 것으로 예상되는 <mark>**데이터 유형을 나타내는 테스트 데이터 세트가 필요하다**</mark> (테스트 데이터 세트 생성에 대한 자세한 내용은 이 가이드의 뒷부분에서 다룬다).

우리의 `data_source_config` 매개변수에서 우리는 데이터 세트의 각 `item`이 <mark>**두 가지 속성을 가진 JSON 스키마를 따르도록 지정**</mark>한다.

- `ticket_text`: 사용자 질문
- `correct_label`: 모델이 일치시켜야 할 <mark>**"정답" 출력으로, 사람이 제공**</mark>한다.

프롬프트가 주어졌을 때 모델이 생성하는 출력을 참조할 것이므로, 출력 스키마 또한 `include_sample_schema`를 `true`로 설정한다.

```json
{
  "type": "custom", // string | 데이터 소스의 유형을 지정한다. 'custom'은 사용자가 직접 데이터를 제공하는 방식임을 의미한다.
  "item_schema": {
    "type": "object", // string | 각 데이터 항목이 객체(JSON object) 형태로 구성됨을 나타낸다.
    "properties": {
      // 'ticket'이라는 키에 문자열 타입의 값이 들어간다.
      // 이는 모델 평가에 사용될 입력 데이터(예: IT 티켓 내용)이다.
      "ticket": { "type": "string" },
      // 'category'라는 키에 문자열 타입의 값이 들어간다.
      // 이는 'ticket'에 대한 정답 레이블(예: 티켓의 올바른 분류)이다.
      "category": { "type": "string" }
    },
    // list | 각 데이터 항목에 'ticket'과 'category' 필드가 반드시 포함되어야 함을 명시한다.
    "required": ["ticket", "category"]
  },
  "include_sample_schema": true // boolean | True로 설정 시, 데이터 소스 설정에 샘플 스키마가 포함된다.
}
```

<br>

<mark>**설명: `testing_criteria` 매개변수**</mark>

- `testing_criteria`는 모델의 답변을 평가하는 규칙을 정하는 것이다.
- 이 규칙에 따라 <mark>**`string_check`라는 그레이더를 사용**</mark>한다.
- `string_check` 그레이더는 <mark>**모델의 출력과 정답 라벨이 정확히 일치하는지 확인**</mark>한다.
- 정답 라벨은 테스트 데이터에 포함된 <mark>**`correct_label` 필드의 값**</mark>이다.
- 결론적으로, 모델의 출력이 <mark>**`correct_label`과 같으면**</mark> <mark>**통과, 다르면 실패로 판정**</mark>한다.
  - `ticket_text`와 `correct_label`은 원하는 대로 바꿀 수 있음
  - 데이터 파일에 IT 티켓 텍스트가 `request_description`이라는 키로 저장되어 있다면, `ticket_text` 대신 `request_description`으로 변경
  - 정답 레이블 키가 `answer_category`라면, `correct_label` 대신 `answer_category`로 변경
  - `item_schema`에 정의된 필드 이름이 <mark>**실제 데이터 파일의 키 이름과 정확히 일치**</mark>
- `{{ }}`는 평가에 동적 데이터를 삽입하는 <mark>**템플릿 구문**</mark>이다
  - <mark>**`{{ item.correct_label }}`**</mark>: 테스트 데이터에 있는 <mark>**정답 값**</mark>을 가리킨다.
  - <mark>**`{{ sample.output_text }}`**</mark>: 모델이 <mark>**생성한 출력**</mark>을 의미한다.
  - Evals API는 이 <mark>**두 값을 비교하여 모델의 성능을 평가**</mark>한다.
  - 이 템플릿 구문은 <mark>**평가를 실행할 때 자동으로 데이터가 채워진다.**</mark>

```json
{
  "type": "string_check", // string | 평가 기준의 유형을 지정한다. 'string_check'는 두 문자열을 비교하여 일치 여부를 판단한다.
  "name": "Category string match", // string | 이 평가 기준의 이름이다. 사람이 이해하기 쉽게 작성한다.
  "input": "{{ sample.output_text }}", // string | 평가 대상이 되는 문자열이다. 템플릿 변수인 '{{ sample.output_text }}'는 모델의 출력 텍스트를 참조한다.
  "operation": "eq", // string | 수행할 비교 연산이다. 'eq'는 'equal'을 의미하며, 'input'과 'reference'가 정확히 일치하는지 확인한다.
  "reference": "{{ item.category }}" // string | 비교 기준이 되는 문자열이다. 템플릿 변수인 '{{ item.category }}'는 데이터 소스에서 제공된 정답 카테고리 레이블을 참조한다.
}
```

<br>

Graders

[Text similarity grader](https://platform.openai.com/docs/guides/graders#text-similarity-grader)

텍스트 유사성 평가기는 모델이 생성한 텍스트가 참조 텍스트와 얼마나 가까운지 점수를 매기는 도구이다.

이는 주로 정해진 답이 없는 개방형 텍스트 응답을 평가할 때 유용하다.

예를 들어, 전문가가 작성한 문단 형태의 참조 답변이 있을 때, 모델이 생성한 답변이 참조 답변과 얼마나 유사한지 수치로 확인할 수 있다.

다양한 평가 프레임워크를 활용하여 텍스트 유사도를 측정한다.

이를 통해 모델의 성능을 정량적으로 평가할 수 있다.

```json
{
    "type": "text_similarity",
    "name": string,
    "input": string,
    "reference": string,
    "pass_threshold": number,
    "evaluation_metric": "fuzzy_match" | "bleu" | "gleu" | "meteor" | "cosine" | "rouge_1" | "rouge_2" | "rouge_3" | "rouge_4" | "rouge_5" | "rouge_l"
}
```

- <mark>**`fuzzy_match`**</mark>: 입력 텍스트와 참조 텍스트 간의 퍼지 문자열 매칭을 수행한다. `rapidfuzz` 라이브러리를 사용한다.
- <mark>**`bleu`**</mark>: 입력 텍스트와 참조 텍스트 간의 <mark>**BLEU**</mark> 점수를 계산한다.
- <mark>**`gleu`**</mark>: 입력 텍스트와 참조 텍스트 간의 <mark>**Google BLEU**</mark> 점수를 계산한다.
- <mark>**`meteor`**</mark>: 입력 텍스트와 참조 텍스트 간의 <mark>**METEOR**</mark> 점수를 계산한다.
- <mark>**`cosine`**</mark>: 입력 텍스트와 참조 텍스트의 임베딩 벡터 간 코사인 유사도를 계산한다. `text-embedding-3-large` 모델을 사용하며, 평가 목적으로만 사용할 수 있다.
- <mark>**`rouge-*`**</mark>: 입력 텍스트와 참조 텍스트 간의 <mark>**ROUGE**</mark> 점수를 계산한다.

<br>

평가를 생성한 후, 고유 식별자인 UUID가 할당된다. 이 UUID는 나중에 실행을 시작할 때 해당 평가를 참조하는 데 필요하다.

```json
{
  "object": "eval", // string | 이 객체의 유형을 나타낸다. "eval"은 평가 작업 객체임을 의미한다.
  "id": "eval_67e321d23b54819096e6bfe140161184", // string | 평가 작업의 고유 식별자(ID)이다.
  "data_source_config": {
    "type": "custom", // string | 데이터 소스의 유형을 나타낸다. "custom"은 사용자가 직접 제공한 데이터임을 의미한다.
    "schema": "{ ... omitted for brevity... }" // object | 데이터 소스의 스키마 정의이다. 여기서는 가독성을 위해 일부 내용이 생략되었다.
  },
  "testing_criteria": [
    {
      "name": "Match output to human label", // string | 평가 기준의 이름이다.
      "id": "Match output to human label-c4fdf789-2fa5-407f-8a41-a6f4f9afd482", // string | 평가 기준의 고유 식별자(ID)이다.
      "type": "string_check", // string | 평가 기준의 유형이다. "string_check"는 문자열을 비교하여 일치 여부를 판단한다.
      "input": "{{ sample.output_text }}", // string | 평가 대상 문자열이다. 모델의 출력 텍스트를 나타내는 템플릿 변수이다.
      "reference": "{{ item.correct_label }}", // string | 비교 기준이 되는 문자열이다. 데이터 소스의 정답 레이블을 나타내는 템플릿 변수이다.
      "operation": "eq" // string | 수행할 비교 연산이다. "eq"는 두 문자열이 동일한지 확인하는 "equal" 연산이다.
    }
  ],
  "name": "IT Ticket Categorization", // string | 평가 작업의 이름이다.
  "created_at": 1742938578, // integer | 작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
  "metadata": {} // object | 사용자가 정의한 추가 메타데이터이다. 비어 있는 경우 {}로 표시된다.
}
```

우리는 애플리케이션의 원하는 동작을 설명하는 평가를 생성했으므로, 이제 테스트 데이터 세트를 사용하여 프롬프트를 테스트해 보겠다.

<br>

## 평가로 프롬프트 테스트하기

이제 앱이 <mark>**어떻게 동작하기를 원하는지 평가에 정의했으므로,**</mark>

테스트 데이터 샘플을 올바르게 생성하는 프롬프트를 구성해 보자.

<br>

<mark>**테스트 데이터 업로드하기**</mark>

<mark>**평가용 테스트 데이터 제공 방법**</mark>은 여러 가지가 있다.

그중 <mark>**JSONL 파일 업로드 방식**</mark>은 편리하고 효율적인 방법이다.

<mark>**JSONL은 'JSON Lines'의 약어**</mark>이며, 한 줄에 하나의 JSON 객체를 담는 파일 형식이다.

이 방식은 평가를 위해 <mark>**미리 정의된 데이터 스키마(데이터 구조)에 맞춰 데이터를 생성**</mark>하여 한 번에 업로드할 수 있게 한다.

이를 통해 <mark>**대량의 데이터를 일관된 형식으로**</mark> 시스템에 전달하여 평가를 실행할 수 있다.

<br>

우리가 설정한 스키마를 준수하는 샘플 JSONL 파일은 다음과 같다.

```json
{ "item":
	{
		"ticket_text": "My monitor won't turn on!",
		"correct_label": "Hardware"
	}
}
{ "item":
	{
		"ticket_text": "I'm in vim and I can't quit!",
		"correct_label": "Software"
	}
}
{ "item":
	{
		"ticket_text": "Best restaurants in Cleveland?",
		"correct_label": "Other"
	}
}
```

- <mark>**테스트 데이터 세트**</mark>는 모델 출력과 <mark>**비교할 입력 및 정답 레이블을 모두 포함**</mark>한다.
- 이 데이터 파일을 <mark>**OpenAI 플랫폼**</mark>에 <mark>**업로드하여 나중에 참조**</mark>할 수 있도록 해야 한다.
- <mark>**파일 업로드**</mark>는 <mark>**대시보드**</mark>를 통해 수동으로 진행하거나, <mark>**API를 활용하여 프로그래밍 방식으로 가능**</mark>하다.
- API를 사용하는 경우, `'tickets.jsonl'`와 같은 <mark>**형식의 파일을 저장한 디렉터리에서 명령을 실행**</mark>해야 한다.
- 이를 통해 모델의 성능을 체계적으로 평가하고 관리할 수 있다.

<br>

테스트 데이터 파일 업로드하기

```python
from openai import OpenAI

client = OpenAI()

file = client.files.create(
		    file=open("tickets.jsonl", "rb"),
		    purpose="evals"
		   )

print(file)
```

- <mark>**`client.files.create`를 호출**</mark>하여 로컬의 `"tickets.jsonl`" 파일을 <mark>**OpenAI 플랫폼에 올린다.**</mark>
- 업로드 시 `purpose`를 `"evals"`로 지정하여 파일의 용도를 명확히 한다.

<br>

```json
{
  "object": "file", // 이 객체의 유형을 나타낸다. 'file'은 파일 객체임을 의미한다.
  "id": "file-CwHg45Fo7YXwkWRPUkLNHW", // 파일의 고유 식별자(ID)이다.
  "purpose": "evals", // 파일의 용도를 나타낸다. 'evals'는 평가(evaluation) 작업에 사용됨을 의미한다.
  "filename": "tickets.jsonl", // 업로드된 파일의 이름이다.
  "bytes": 208, // 파일의 크기를 바이트(byte) 단위로 나타낸다.
  "created_at": 1742834798, // 파일이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
  "expires_at": null, // 파일이 만료되는 시간이다. 'null'은 만료되지 않음을 의미한다.
  "status": "processed", // 파일의 현재 상태. 'processed'는 파일 처리가 완료되어 사용할 준비가 되었음을 의미한다.
  "status_details": null // 파일 상태에 대한 추가적인 세부 정보이다. 현재는 'null'로 표시된다.
}
```

- 파일 업로드 성공 시, <mark>**응답 페이로드에 있는 고유한 id를 반드시 기록**</mark>해야 한다.
- 이 <mark>**id**</mark>는 나중에 <mark>**업로드된 파일을 참조할 때 중요한 역**</mark>할을 한다.

<br>

<mark>**평가 실행 생성하기**</mark>

- 준비된 <mark>**테스트 데이터**</mark>를 활용해 프롬프트의 성능을 평가한다.
- 이 평가는 <mark>**API를 통해 평가 실행을 생성하여 진행**</mark>할 수 있다.
- 이때, <mark>**`YOUR_EVAL_ID`와 `YOUR_FILE_ID`**</mark>를 위 단계에서 얻은 <mark>**고유 ID로 대체**</mark>해야 한다.
- `YOUR_EVAL_ID`는 <mark>**평가 구성에 대한 ID**</mark>이며, `YOUR_FILE_ID`는 업로드된 <mark>**테스트 데이터 파일의 ID**</mark>이다.
- 이 과정을 통해 프롬프트가 설정한 <mark>**테스트 기준**</mark>에 얼마나 잘 부합하는지 확인할 수 있다.

<br>

평가 실행 생성하기

```python
from openai import OpenAI

client = OpenAI()

# run 객체 생성 및 반환
# eval 작업의 실행(run)을 생성한다.
run = client.evals.runs.create(
    "YOUR_EVAL_ID",  # string | 평가(eval) 작업의 고유 ID이다.
    name="Categorization text run",  # string | 생성될 실행(run)의 이름을 지정한다.
    data_source={  # object | 평가에 사용될 데이터 소스 설정
        "type": "responses",  # string | 데이터 소스의 유형. 'responses'는 모델의 응답을 평가함을 의미한다.
        "model": "gpt-4.1",  # string | 평가 대상이 되는 모델의 이름이다.
        "input_messages": {  # object | 모델에 제공될 입력 메시지 템플릿
            "type": "template",  # string | 입력 메시지의 유형. 'template'은 미리 정의된 템플릿을 사용함을 의미한다.
            "template": [  # array | 역할(role)과 내용(content)을 포함하는 메시지 템플릿 목록
                {
                    "role": "developer",  # string | 메시지를 보내는 주체의 역할이다.
                    "content": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of 'Hardware', 'Software', or 'Other'. Respond with only one of those words."  # string | 역할에 해당하는 메시지 내용이다.
                },
                {
                    "role": "user",  # string | 사용자 역할을 정의한다.
                    "content": "{{ item.ticket_text }}"  # string | 'ticket_text' 필드의 데이터를 동적으로 삽입하는 템플릿 문법이다.
                },
            ],
        },
        "source": {  # object | 실제 평가 데이터가 포함된 파일 소스 정보
            "type": "file_id",  # string | 소스의 유형. 'file_id'는 파일 ID를 통해 데이터를 참조함을 의미한다.
            "id": "YOUR_FILE_ID"  # string | 평가 데이터가 저장된 파일의 고유 식별자(ID)이다.
        },
    },
)

print(run)  # run 객체의 내용을 출력하여 생성된 실행 정보를 확인한다.
```

<mark>**이중 중괄호 구문**</mark>을 사용하여 `item.ticket_text`와 같은 <mark>**동적 변수를 템플릿에 삽입**</mark>할 수 있다.

- 테스트 데이터 파일에 있는 필드의 이름이 `"ticket_text"`이기 때문이다.
- 만약 테스트 데이터 파일내에 `{"id": "...", "support_request": "..."}`와 같은 구조로 되어 있다면, 프롬프트 템플릿에서는 `{{ item.support_request }}`를 사용해야 한다.
- <mark>**`item`은 고정된 객체 이름입니다.**</mark> `item`은 OpenAI Evals 프레임워크에서 테스트 데이터의 개별 항목을 참조할 때 사용하는 예약어이므로 변경할 수 없다.

이 과정을 통해 각 <mark>**테스트 항목의 데이터가 프롬프트에 자동으로 적용**</mark>되어 모델의 응답을 얻게 된다.

이를 통해 수동 작업 없이 전체 데이터 세트에 대한 평가를 진행할 수 있다.

<br>

코드 설명

- `client.evals.runs.create` 함수를 사용하여 `YOUR_EVAL_ID`와 `YOUR_FILE_ID`를 기반으로 <mark>**`Categorization text run`이라는 이름의 실행**</mark>을 만든다.
- 이 실행은 `gpt-4.1` 모델을 대상으로, 제공된 <mark>**프롬프트 템플릿**</mark>에 따라 IT 지원 티켓을 'Hardware', 'Software', 또는 'Other'로 분류하도록 지시한다.
- 프롬프트 템플릿은 <mark>**이중 중괄호 구문을 사용**</mark>하여 테스트 데이터 파일에서 가져온 `item.ticket_text` <mark>**데이터를 동적으로 삽입**</mark>한다.
- 이렇게 생성된 `run` <mark>**객체를 통해 모든 테스트 데이터에 대한 모델의 응답을 생성하고 평가**</mark>할 수 있다.

<br>

평가 실행이 성공적으로 생성되면 다음과 같은 API 응답을 받게 된다.

```json
{
    "object": "eval.run",  // 이 객체의 유형을 나타낸다. 'eval.run'은 평가(evaluation) 작업의 실행을 의미한다.
    "id": "evalrun_67e44c73eb6481909f79a457749222c7",  // 평가 실행의 고유 식별자(ID)이다.
    "eval_id": "eval_67e44c5becec81909704be0318146157",  // 이 실행이 속한 평가 작업의 고유 ID이다.
    "report_url": "https://platform.openai.com/evaluations/abc123",  // 평가 결과 보고서에 접근할 수 있는 URL이다.
    "status": "queued",  // 평가 실행의 현재 상태. 'queued'는 실행 대기 중임을 의미한다.
    "model": "gpt-4.1",  // 이 평가에서 사용된 모델의 이름이다.
    "name": "Categorization text run",  // 평가 실행에 할당된 이름이다.
    "created_at": 1743015028,  // 평가 실행이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
    "result_counts": { ... },  // 평가 결과에 대한 요약 정보를 포함하는 객체이다. (예: 통과, 실패, 오류 수 등)
    "per_model_usage": null,  // 모델별 사용량에 대한 정보이다. 현재는 'null'로 표시된다.
    "per_testing_criteria_results": null,  // 각 평가 기준에 따른 결과에 대한 정보이다. 현재는 'null'로 표시된다.
    "data_source": {  // 평가에 사용된 데이터 소스에 대한 상세 정보
        "type": "responses",  // 데이터 소스의 유형. 'responses'는 모델의 응답을 평가함을 의미한다.
        "source": {  // 실제 데이터 파일에 대한 정보
            "type": "file_id",  // 소스의 유형. 'file_id'는 파일 ID를 통해 데이터를 참조함을 의미한다.
            "id": "file-J7MoX9ToHXp2TutMEeYnwj"  // 평가 데이터가 저장된 파일의 고유 식별자(ID)이다.
        },
        "input_messages": {  // 모델에 제공될 입력 메시지 템플릿 정보
            "type": "template",  // 입력 메시지의 유형. 'template'은 미리 정의된 템플릿을 사용함을 의미한다.
            "template": [  // 역할(role)과 내용(content)을 포함하는 메시지 템플릿 목록
                {
                    "type": "message",  // 이 객체의 유형. 'message'는 메시지 객체임을 나타낸다.
                    "role": "developer",  // 메시지를 보내는 주체의 역할이다.
                    "content": {  // 메시지의 내용 객체
                        "type": "input_text",  // 내용의 유형. 'input_text'는 텍스트 입력임을 나타낸다.
                        "text": "You are an expert in...."  // 역할에 해당하는 메시지 내용이다.
                    }
                },
                {
                    "type": "message",  // 메시지 객체
                    "role": "user",  // 사용자 역할을 정의한다.
                    "content": {  // 메시지의 내용 객체
                        "type": "input_text",  // 텍스트 입력
                        "text": "{{item.ticket_text}}"  // 'ticket_text' 필드의 데이터를 동적으로 삽입하는 템플릿 문법이다.
                    }
                }
            ]
        },
        "model": "gpt-4.1",  // 이 데이터 소스 평가에 사용된 모델의 이름이다.
        "sampling_params": null  // 샘플링 파라미터에 대한 정보이다. 현재는 'null'로 표시된다.
    },
    "error": null,  // 평가 실행 중 발생한 오류 정보이다. 오류가 없으므로 'null'이다.
    "metadata": {}  // 사용자가 정의한 추가 메타데이터가 포함된 객체이다. 현재는 비어 있다.
}
```

이제 평가 실행이 대기열에 추가되었다.

지정한 <mark>**프롬프트와 모델로 테스트용 응답을 생성하면서 비동기적으로 실행**</mark>될 것이다.

<br>

## 결과 분석하기

- <mark>**웹훅을 활용**</mark>하면 평가 실행의 성공, 실패, 취소 상태에 대한 <mark>**알림을 받을 수 있다.**</mark>
- 웹훅을 사용하려면 <mark>**엔드포인트를 만들고**</mark> `eval.run.succeeded`, `eval.run.failed`, `eval.run.canceled` <mark>**이벤트에 구독**</mark>해야 한다.
- 평가 실행 완료까지는 데이터 세트 크기에 따라 시간이 소요될 수 있다.
- <mark>**실행 상태**</mark>는 <mark>**OpenAI 대시보드나 API를 통해 실시간으로 확인**</mark>할 수 있다.

<br>

평가 실행 상태 조회하기

```python
from openai import OpenAI

client = OpenAI()

run = client.evals.runs.retrieve("YOUR_EVAL_ID", "YOUR_RUN_ID")

print(run)
```

상태를 조회하려면 평가 및 <mark>**평가 실행의 UUID가 모두 필요**</mark>하다.

조회하면 다음과 같은 평가 실행 데이터를 볼 수 있다.

```json
{
  "object": "eval.run", // 이 객체의 유형을 나타낸다. 'eval.run'은 평가(evaluation) 작업의 실행을 의미한다.
  "id": "evalrun_67e44c73eb6481909f79a457749222c7", // 평가 실행의 고유 식별자(ID)이다.
  "eval_id": "eval_67e44c5becec81909704be0318146157", // 이 실행이 속한 평가 작업의 고유 ID이다.
  "report_url": "https://platform.openai.com/evaluations/xxx", // 평가 결과 보고서에 접근할 수 있는 URL이다.
  "status": "completed", // 평가 실행의 현재 상태. 'completed'는 성공적으로 완료되었음을 의미한다.
  "model": "gpt-4.1", // 이 평가에서 사용된 모델의 이름이다.
  "name": "Categorization text run", // 평가 실행에 할당된 이름이다.
  "created_at": 1743015028, // 평가 실행이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
  "result_counts": {
    // 평가 결과에 대한 요약 정보
    "total": 3, // 총 평가 항목 수이다.
    "errored": 0, // 평가 중 오류가 발생한 항목 수이다.
    "failed": 0, // 평가 기준을 통과하지 못한 항목 수이다.
    "passed": 3 // 평가 기준을 성공적으로 통과한 항목 수이다.
  },
  "per_model_usage": [
    // 모델별 사용량에 대한 정보
    {
      "model_name": "gpt-4o-2024-08-06", // 사용된 모델의 구체적인 버전 이름이다.
      "invocation_count": 3, // 모델이 호출된 횟수이다.
      "prompt_tokens": 166, // 프롬프트에 사용된 총 토큰 수이다.
      "completion_tokens": 6, // 모델 응답에 사용된 총 토큰 수이다.
      "total_tokens": 172, // 프롬프트와 응답을 합한 총 토큰 수이다.
      "cached_tokens": 0 // 캐시된 토큰 수이다.
    }
  ],
  "per_testing_criteria_results": [
    // 각 평가 기준에 따른 결과
    {
      "testing_criteria": "Match output to human label-40d67441-5000-4754-ab8c-181c125803ce", // 평가에 사용된 기준의 이름. 여기서는 '인간 라벨과 출력 일치'를 평가했다.
      "passed": 3, // 해당 기준을 통과한 항목 수이다.
      "failed": 0 // 해당 기준을 통과하지 못한 항목 수이다.
    }
  ],
  "data_source": {
    // 평가에 사용된 데이터 소스에 대한 상세 정보
    "type": "responses", // 데이터 소스의 유형. 'responses'는 모델의 응답을 평가함을 의미한다.
    "source": {
      // 실제 데이터 파일에 대한 정보
      "type": "file_id", // 소스의 유형. 'file_id'는 파일 ID를 통해 데이터를 참조함을 의미한다.
      "id": "file-J7MoX9ToHXp2TutMEeYnwj" // 평가 데이터가 저장된 파일의 고유 식별자(ID)이다.
    },
    "input_messages": {
      // 모델에 제공될 입력 메시지 템플릿 정보
      "type": "template", // 입력 메시지의 유형. 'template'은 미리 정의된 템플릿을 사용함을 의미한다.
      "template": [
        // 역할(role)과 내용(content)을 포함하는 메시지 템플릿 목록
        {
          "type": "message", // 메시지 객체
          "role": "developer", // 메시지를 보내는 주체의 역할이다.
          "content": {
            // 메시지의 내용 객체
            "type": "input_text", // 내용의 유형. 'input_text'는 텍스트 입력임을 나타낸다.
            "text": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of Hardware, Software, or Other. Respond with only one of those words." // 역할에 해당하는 메시지 내용이다.
          }
        },
        {
          "type": "message", // 메시지 객체
          "role": "user", // 사용자 역할을 정의한다.
          "content": {
            // 메시지의 내용 객체
            "type": "input_text", // 텍스트 입력
            "text": "{{item.ticket_text}}" // 'ticket_text' 필드의 데이터를 동적으로 삽입하는 템플릿 문법이다.
          }
        }
      ]
    },
    "model": "gpt-4.1", // 이 데이터 소스 평가에 사용된 모델의 이름이다.
    "sampling_params": null // 샘플링 파라미터에 대한 정보이다. 현재는 'null'로 표시된다.
  },
  "error": null, // 평가 실행 중 발생한 오류 정보이다. 오류가 없으므로 'null'이다.
  "metadata": {} // 사용자가 정의한 추가 메타데이터가 포함된 객체이다. 현재는 비어 있다.
}
```

- <mark>**API 응답**</mark>에는 <mark>**평가 결과, API 사용량**</mark>, 그리고 대시보드에서 결과를 시각적으로 탐색할수 있는 <mark>**리포트 페이지로 연결되는**</mark> `report_url`이 포함된다.
- 이를 통해 사용자는 <mark>**테스트 기준 결과와 모델 응답 생성에 사용된 비용을 확인**</mark>할 수 있다.
- 예시로 든 테스트에서는 모델이 <mark>**작은 데이터 샘플에 대해 안정적인 성능**</mark>을 보였다.
- 실제 상황에서는 <mark>**다양한 기준, 프롬프트, 데이터 세트로 평가를 반복적으로 수행**</mark>해야 한다.
- 이 과정을 통해 <mark>**LLM 애플리케이션 평가를 위한 필수적인 도구와 방법을 습득**</mark>할 수 있다.

<br>

참고

- [OpenAI Evaluating model performance Docs](https://platform.openai.com/docs/guides/evals?api-mode=responses)
