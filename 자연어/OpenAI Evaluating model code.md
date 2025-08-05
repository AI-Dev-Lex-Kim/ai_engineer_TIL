# OpenAI Evaluating model code

## 1. 평가 환경 설정

```python
from openai import OpenAI

client = OpenAI()

# evals.create 함수를 호출하여 새로운 평가(evaluation) 작업을 생성한다.
eval_obj = client.evals.create(
    name=name,  # 평가 작업 이름
    data_source_config={
        "type": "custom",  # 데이터 소스의 유형을 지정한. 'custom'은 사용자가 직접 데이터를 제공하는 방식.
        "item_schema": {
            "type": "object",  # 각 데이터 항목이 객체(object)
            "properties": {
                "user_query": {"type": "string"},
                "label": {"type": "string"},
            },
            # 각 데이터 항목에 'prompt'와 'label' 필드가 반드시 포함되어야 함을 명시
            "required": ["user_query", "label"],
        },
        # True로 설정 시, 데이터 소스 설정에 샘플 스키마가 포함
        "include_sample_schema": True,
    },
    # 평가 작업의 성공/실패 여부를 판단하는 기준 목록이다.
    testing_criteria=[
            {
            "type": "text_similarity",
            "name": 'meteor grader',
            "input": "{{  sample.output_text  }}",
            "reference": "{{  item.label  }}",
            "evaluation_metric": "meteor" # "fuzzy_match" | "bleu" | "gleu" | "meteor" | "cosine" | "rouge_1" | "rouge_2" | "rouge_3" | "rouge_4" | "rouge_5" | "rouge_l"
        }
    ],
)

print(eval_obj)  # 생성된 평가 객체(eval_obj)의 세부 정보를 출력한다.
"""
{
  "object": "eval",  // string | 이 객체의 유형을 나타낸다. "eval"은 평가 작업 객체임을 의미한다.
  "id": "eval_67e321d23b54819096e6bfe140161184",  // string | 평가 작업의 고유 식별자(ID)이다.
  "data_source_config": {
    "type": "custom",  // string | 데이터 소스의 유형을 나타낸다. "custom"은 사용자가 직접 제공한 데이터임을 의미한다.
    "schema": "{ ... omitted for brevity... }"  // object | 데이터 소스의 스키마 정의이다. 여기서는 가독성을 위해 일부 내용이 생략되었다.
  },
  "testing_criteria": [
    {
      "name": "Match output to human label",  // string | 평가 기준의 이름이다.
      "id": "Match output to human label-c4fdf789-2fa5-407f-8a41-a6f4f9afd482",  // string | 평가 기준의 고유 식별자(ID)이다.
      "type": "string_check",  // string | 평가 기준의 유형이다. "string_check"는 문자열을 비교하여 일치 여부를 판단한다.
      "input": "{{ sample.output_text }}",  // string | 평가 대상 문자열이다. 모델의 출력 텍스트를 나타내는 템플릿 변수이다.
      "reference": "{{ item.correct_label }}",  // string | 비교 기준이 되는 문자열이다. 데이터 소스의 정답 레이블을 나타내는 템플릿 변수이다.
      "operation": "eq"  // string | 수행할 비교 연산이다. "eq"는 두 문자열이 동일한지 확인하는 "equal" 연산이다.
    }
  ],
  "name": "IT Ticket Categorization",  // string | 평가 작업의 이름이다.
  "created_at": 1742938578,  // integer | 작업이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
  "metadata": {}  // object | 사용자가 정의한 추가 메타데이터이다. 비어 있는 경우 {}로 표시된다.
}
"""
```

## 2. 테스트 데이터 파일 생성

```json
{ "item":
	{
		"user_query": "My monitor won't turn on!",
		"label": "Hardware"
	}
}
{ "item":
	{
		"user_query": "I'm in vim and I can't quit!",
		"label": "Software"
	}
}
{ "item":
	{
		"user_query": "Best restaurants in Cleveland?",
		"label": "Other"
	}
}
```

## 3. 테스트 데이터 업로드

```python
from openai import OpenAI

client = OpenAI()

file = client.files.create(
			    file=open("tickets.jsonl", "rb"),
			    purpose="evals"
		   )

print(file)

""" file
{
    "object": "file",  // 이 객체의 유형을 나타낸다. 'file'은 파일 객체임을 의미한다.
    "id": "file-CwHg45Fo7YXwkWRPUkLNHW",  // 파일의 고유 식별자(ID)이다.
    "purpose": "evals",  // 파일의 용도를 나타낸다. 'evals'는 평가(evaluation) 작업에 사용됨을 의미한다.
    "filename": "tickets.jsonl",  // 업로드된 파일의 이름이다.
    "bytes": 208,  // 파일의 크기를 바이트(byte) 단위로 나타낸다.
    "created_at": 1742834798,  // 파일이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
    "expires_at": null,  // 파일이 만료되는 시간이다. 'null'은 만료되지 않음을 의미한다.
    "status": "processed",  // 파일의 현재 상태. 'processed'는 파일 처리가 완료되어 사용할 준비가 되었음을 의미한다.
    "status_details": null  // 파일 상태에 대한 추가적인 세부 정보이다. 현재는 'null'로 표시된다.
}
"""
```

## 4. 평가 실행 생성

```python
from openai import OpenAI

client = OpenAI()
run = client.evals.runs.create(
    eval_obj.id,  # string | 평가(eval) 작업의 고유 ID이다.
    name=name,  # string | 생성될 실행(run)의 이름을 지정한다.
    data_source={  # object | 평가에 사용될 데이터 소스 설정
        "type": "responses",  # string | 데이터 소스의 유형. 'responses'는 모델의 응답을 평가함을 의미한다.
        "model": model,  # string | 평가 대상이 되는 모델의 이름이다.
        "input_messages": {  # object | 모델에 제공될 입력 메시지 템플릿
            "type": "template",  # string | 입력 메시지의 유형. 'template'은 미리 정의된 템플릿을 사용함을 의미한다.
            "template": [  # array | 역할(role)과 내용(content)을 포함하는 메시지 템플릿 목록
                {
                    "role": "user",  # string | 사용자 역할을 정의한다.
                    "content": f"""
<prompt>
    <identity>
        너는 데이터 분석가이다. 너의 유일한 임무는 컨설턴트가 고객사에게 적합한 입찰 기회를 놓치지 않도록, 매일 쏟아지는 RFP의 핵심 정보를 가장 효율적으로 처리하여 제공하는 것이다. 너는 사용자의 질문에 대해, RFP 문서에서 객관적인 사실(Fact)만을 추출하여 한눈에 파악하기 쉬운 형태로 요약하고 구조화한다. 너의 목표는 컨설턴트가 RFP의 전체 내용을 읽지 않고도, 단 몇 분 안에 해당 입찰의 참여 여부를 판단할 수 있는 핵심 근거를 제공하는 것이다.
    </identity>
    <instructions>
        ## 목표
        - 사용자의 질문에 대해 주어진 RFP 문서 내용에만 근거하여 명확하고 사실적인 답변을 생성한다.

        ## 핵심 규칙
        1.  사실 기반 응답: 오직 아래 <context>에 제공된 RFP 문서 내용만을 사용하여 답변해야 한다. 추론, 가정, 또는 외부 지식을 절대 사용해서는 안 된다.
        2.  출처 명시: 모든 주장의 근거는 반드시 문서에서 인용하여 신뢰성을 높인다.
        3.  종합적 요약: 관련된 여러 문서 조각이 있다면, 이를 논리적으로 종합하여 하나의 일관된 답변으로 재구성한다. 사용자가 전체 맥락을 파악할 수 있도록 핵심 정보를 요약하여 제공한다.
        4.  쉬운 언어 사용: RFP의 전문 용어나 복잡한 문장은 사용자가 이해하기 쉬운 평이한 표현으로 바꾸어 설명한다.

        ## 답변 형식
        - '예/아니오' 질문: 질문이 '예/아니오'로 답변될 수 있는 경우, 문서 내용을 근거로 '예' 또는 '아니오'로 명확하게 먼저 답변하고, 그 이유가 되는 문장을 함께 제시한다.
        - 비교 질문: 두 개 이상의 RFP 문서를 비교해달라는 요청에는, 지정된 기준에 따라 각 문서의 핵심 내용을 표(Table) 형식으로 요약하여 명확하게 비교한다.
        - 전문가적 조언: 입찰 컨설턴트로서, RFP 문서 내용을 바탕으로 전문적인 조언을 포함하여 답변을 구성한다.
    </instructions>
    <context>
        <retrieved_rfp_documents>
            {retrieved_rfp_text}
        </retrieved_rfp_documents>

        <user_question>
            {{{{item.user_query}}}}
        </user_question>
    </context>
</prompt>
""" # string | 'user_query' 필드의 데이터를 동적으로 삽입하는 템플릿 문법이다.
                },
            ],
        },
        "source": {  # object | 실제 평가 데이터가 포함된 파일 소스 정보
            "type": "file_id",  # string | 소스의 유형. 'file_id'는 파일 ID를 통해 데이터를 참조함을 의미한다.
            "id": file_obj.id  # string | 평가 데이터가 저장된 파일의 고유 식별자(ID)이다.
        },
    },
)

print(run)  # run 객체의 내용을 출력하여 생성된 실행 정보를 확인한다.
""" run
{
    "object": "eval.run",  // 이 객체의 유형을 나타낸다. 'eval.run'은 평가(evaluation) 작업의 실행을 의미한다.
    "id": "evalrun_67e44c73eb6481909f79a457749222c7",  // 평가 실행의 고유 식별자(ID)이다.
    "eval_id": "eval_67e44c5becec81909704be0318146157",  // 이 실행이 속한 평가 작업의 고유 ID이다.
    "report_url": "https://platform.openai.com/evaluations/xxx",  // 평가 결과 보고서에 접근할 수 있는 URL이다.
    "status": "completed",  // 평가 실행의 현재 상태. 'completed'는 성공적으로 완료되었음을 의미한다.
    "model": "gpt-4.1",  // 이 평가에서 사용된 모델의 이름이다.
    "name": "Categorization text run",  // 평가 실행에 할당된 이름이다.
    "created_at": 1743015028,  // 평가 실행이 생성된 시간을 나타내는 유닉스 타임스탬프(Unix timestamp)이다.
    "result_counts": {  // 평가 결과에 대한 요약 정보
        "total": 3,  // 총 평가 항목 수이다.
        "errored": 0,  // 평가 중 오류가 발생한 항목 수이다.
        "failed": 0,  // 평가 기준을 통과하지 못한 항목 수이다.
        "passed": 3  // 평가 기준을 성공적으로 통과한 항목 수이다.
    },
    "per_model_usage": [  // 모델별 사용량에 대한 정보
        {
            "model_name": "gpt-4o-2024-08-06",  // 사용된 모델의 구체적인 버전 이름이다.
            "invocation_count": 3,  // 모델이 호출된 횟수이다.
            "prompt_tokens": 166,  // 프롬프트에 사용된 총 토큰 수이다.
            "completion_tokens": 6,  // 모델 응답에 사용된 총 토큰 수이다.
            "total_tokens": 172,  // 프롬프트와 응답을 합한 총 토큰 수이다.
            "cached_tokens": 0  // 캐시된 토큰 수이다.
        }
    ],
    "per_testing_criteria_results": [  // 각 평가 기준에 따른 결과
        {
            "testing_criteria": "Match output to human label-40d67441-5000-4754-ab8c-181c125803ce",  // 평가에 사용된 기준의 이름. 여기서는 '인간 라벨과 출력 일치'를 평가했다.
            "passed": 3,  // 해당 기준을 통과한 항목 수이다.
            "failed": 0  // 해당 기준을 통과하지 못한 항목 수이다.
        }
    ],
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
                    "type": "message",  // 메시지 객체
                    "role": "developer",  // 메시지를 보내는 주체의 역할이다.
                    "content": {  // 메시지의 내용 객체
                        "type": "input_text",  // 내용의 유형. 'input_text'는 텍스트 입력임을 나타낸다.
                        "text": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of Hardware, Software, or Other. Respond with only one of those words."  // 역할에 해당하는 메시지 내용이다.
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
"""
```

## 5. 평가 실행 상태 조회

```python
eval_name = "RFP_Evaluation"

client = OpenAI()

eval_obj = create_eval_obj(client, eval_name)
file = upload_data(client, file_path)
run = run_eval(client, eval_obj, file, eval_name, model, retrieved_rfp_text)

print("평가 작업이 제출되었습니다. 완료될 때까지 대기합니다.")
while True:
    # 매 루프마다 평가 실행의 최신 상태를 가져옴
    result = client.evals.runs.retrieve(eval_id=eval_obj.id, run_id=run.id)

    # 현재 상태를 출력하여 진행 상황을 보여줌
    print(f"현재 상태: {result.status}... 확인 중...")

    # 상태가 'completed'이면 루프를 종료
    if result.status == 'completed':
        print("평가가 성공적으로 완료되었습니다.")
        break
    # 실패하거나 취소된 경우에도 루프를 종료
    elif result.status in ['failed', 'cancelled', 'errored']:
        print(f"평가가 {result.status} 상태로 종료되었습니다.")
        break

    # 아직 진행 중이면 10초간 대기한 후 다시 확인 (API 요청 제한을 피하기 위함)
    time.sleep(10)

# 루프가 종료된 후, 최종 결과를 출력
print("\n--- 최종 평가 결과 ---")
output_items = client.evals.runs.output_items.list(eval_id=eval_obj.id, run_id=run.id)
output_items = output_items.model_dump()
print('output_items: ', output_items)

# score, response 출력
for v in output_items['data']:
    score = v['results'][0]['score']
    response = v['sample']['output'][0]['content']
    print('score:', score)
    print('response:', response)
```
