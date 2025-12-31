# nn.Module로 Hugging Face 모델 커스텀

## AutoModel

특정 head 없이 body만 있는 기본 모델이다.

body만 있는 기본 모델이므로 특정 task를 수행하고 싶을때, 사전학습된 AutoModel을 Body로 사용해서 Head를 추가 하면된다.

<br>

예를 들어보자.

만약 긍정적이거나 부정적인 감정으로 금융 보고서를 분류하는 작은 데이터셋을 가지고 있다고 가정해 보자. Hugging Face로 확인해보니 많은 모델들이 금융 관련 질의응답(QA)을 위해 훈련되었다는 것을 발견했다. 우리와 동일한 작업은 아니지만 동일한 도메인 내에 있을 경우는 사전 훈련된 지식을 바탕으로 Body 레이어 만 가져오면된다.

<br>

두번째 예시를 들어보자

특정 도메인 전용 모델이 훈련한 대규모 데이터셋에서 텍스트를 5개의 카테고리로 분류하는 모델이 Hugging Face에 있다고 하자.

동일한 도메인의 완전히 다른 데이터셋을 사용하며, 5개의 카테고리 대신 2개의 카테고리로 데이터를 분류하고 싶다고 가정해 보자.

다시 모델의 몸체를 사용하고 우리의 2개 카테고리로 분류하는 태스크 헤드를 추가하여 우리 작업에 대한 도메인 전용 지식을 증강하려고 시도할 수 있다.

<br>

### 커스텀 분류기 레이어 추가

<mark>**BERT의 핵심 구조**</mark>만 담고 있는 `AutoModel`을 불러온다.

이렇게 하면 모델의 강력한 언어 이해 능력은 유지하면서, 분류기와 같은 최종 레이어를 직접 추가할 수 있다.

`AutoModel`의 출력값 위에 직접 만든 분류기 레이어(Classification Head)를 추가한다.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CustomModelWithHead(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        # AutoModel로 기본 모델을 로드합니다. 이 모델은 헤드가 없는 원시 임베딩을 출력
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)

        # 커스텀 분류 헤드를 정의합니다.
        # base_model의 마지막 히든 상태(hidden_size)를 입력으로 받습니다.
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1) # 필요에 따라 드롭아웃 추가

    def forward(self, input_ids, attention_mask):
        # 기본 모델을 통과시켜 히든 상태를 얻는다.
        # output.last_hidden_state는 시퀀스의 각 토큰에 대한 히든 상태를 포함
        # output.pooler_output는 토큰의 처리된 최종 히든 상태
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

				# 분류를 위해 토큰의 출력을 사용
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 사용 예시
model_name = "bert-base-uncased"
num_classes = 3 # 예를 들어, 3개의 클래스를 분류하는 작업
custom_model = CustomModelWithHead(model_name, num_classes)

# 모델 입력 (실제 학습에서는 데이터 로더를 사용합니다)
# outputs = custom_model(dummy_inputs, dummy_inputs)

```

클래스를 만들어 사전 학습된 `base_model`과 새롭게 추가된 <mark>**선형(Linear) 분류기 레이어**</mark>를 결합했다.

`custom_model`을 사용하여 여러분의 데이터에 맞게 학습을 진행할 수 있다.

<br>

참고

- https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd/
- https://chanmuzi.tistory.com/243
- https://chanmuzi.tistory.com/266
