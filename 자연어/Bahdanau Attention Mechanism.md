- [Bahdanau Attention Mechanism](#bahdanau-attention-mechanism)
  - [Seq2Seq RNN 문제점](#seq2seq-rnn-문제점)
  - [기본 개념](#기본-개념)
  - [동작 과정 간단 정리](#동작-과정-간단-정리)
  - [동작 방식](#동작-방식)
    - [유사도 점수 계산](#유사도-점수-계산)
    - [컨텍스트 벡터](#컨텍스트-벡터)
  - [코드 구현](#코드-구현)

# Bahdanau Attention Mechanism

## Seq2Seq RNN 문제점

RNN은 문제가 있다.

인코더가 긴 문장을 정해진 크기의 벡터에 모든 정보를 압축하려고 하니 정보 손실이 발생한다.

입력 시퀀스가 길어지면 그만큼 정보 손실이 더 커진다.

이런 정확도가 떨어진것을 보완하기 위해 Attention 기법이 등장했다

<br>

## 기본 개념

어텐션은 디코더에서 타임 스텝(매 시점)마다 출력 단어를 예측할때, 입력 시퀀스 전체를 다시 한번 참고 한다.

이때, 입력 시퀀스 전체 중에서 중요한 부분을 강조해서 참고한다.

해당 타임 스텝에 예측하려는 단어와 연관 될만한 입력 시퀀스의 단어들에 집중해서 참고한다.

<br>

## 동작 과정 간단 정리

1. 디코더가 인코더 입력 시퀀스 전체에서 중요한 부분이 어디인지 알기위해 유사도 점수를 구한다.
2. 유사도 점수를 참고해서 인코더 입력 시퀀스의 요약 정보인 컨텍스트 벡터를 구한다.
3. 현재 타임스텝에서 “컨텍스트 벡터”, “이전 디코더 hidden state”, “현재 입력 단어 인덱스” 3가지가 준비되어있다.
4. 현재 입력 단어 인덱스로 임베딩 벡터를 어휘사전에서 가져온다.
5. “컨텍스트 벡터”와 “현재 단어 임베딩 벡터”를 concat 해준다.
6. GRU 또는 LSTM에 concat된 벡터와 hidden state를 입력해준다.
   - “현재 단어 + 입력 전체 중 중요한 정보”를 함께 보고 RNN이 더 정확한 다음 hidden state를 만들 수 있게 된다.
7. 새로운 hidden state가 출력 된다.
8. 각 단어가 출력될 확률을 계산하기 위해 새로운 hidden state를 어휘 사전 크기만큼의 벡터로 변환해준다.
   - `self.out = nn.Linear(hidden_size, vocab_size)`
     - nn.Linear는 가장 마지막 차원만 변환해줌
     - $y = x \cdot W^T + b$
   - `logits = self.out(hidden)`
   - shape: `(batch, hidden_size)` → shape: `(batch, vocab_size)`
9. 디코더 linear에서 나온 logits를 softmax에 적용해 확률을 구한다.
   - `probs  = F.softmax(logits, dim=1)`
   - 모든 단어 중에서 가장 가능성 높은 단어를 고르기 위해 확률로 바꿔준다.
10. 단어를 선택해 출력한다.
    - 보통 두가지 방법으로 단어를 선택한다.
    1. greedy decoding(argmax)
       - 가장 확률이 높은 단어를 선택
       - 일반적인 번역, 요약일때 사용한다.
    2. sampling(multinomial)
       - 확률 기반으로 랜덤 선택
       - 예: `[0.1, 0.1, 0.8]` → 0번 인덱스 선택 확률 10%, 1번 10%, 2번 80%
       - 보통 2번 인덱스가 선택되지만 가끔 0이나 1도 선택된다.
       - 창의적 생성, 다양성이 필요할때 사용한다(대화, 소설)
    - `top1 = torch.argmax(probs, dim=1)`

<br>

## 동작 방식

### 유사도 점수 계산

디코더 내부에서 각 시점마다 단어를 예측 하기전에,

인코더 시퀀스 전체 중에서 “예측 하려는 단어”와 “연관 될 만한 중요한 단어”을 찾기위해(집중하기 위해 attend) 유사도 점수를 계산한다.

<br>

디코더의 query와 인코더의 keys간의 유사도를 계산한다.

- `query`: 디코더의 현재 hiddens state
- `keys`: 인코더의 시퀀스의 모든 단어의 각 타임스텝별 hidden state들
  - 인코더 입력 시퀀스가 `"나는 밥을 먹었다"`
  - 인코더가 4개의 RNN 스텝을 거쳐 hidden state를 만듭니다
  - `keys = [h_1, h_2, h_3, h_4]`

<br>

1. query, keys를 Wa Linear layer, Ua Linear layer을 통해 각각 선형변환 한다.
   - `Wa`, `Ua`: Linear layer안에 있는 가중치 행렬(처음엔 랜덤 초기화 값)
   - `Wa(query)`: `Wa * query` → shape: `(batch, 1, hidden_size)`
   - `Ua(keys)`: `Ua * key` → shape: `(batch, seq_len, hidden_size)`
   - 선형변환: $y = Wx + b$ → W: 가중치, x: query or keys, b: 편향
2. 선형변환한 두 가중치를 더한다
   - `Wa(query) + Ua(keys)`
   - shape: `(batch, seq_len, hidden_size)`
3. tanh로 비선형 처리한다.
   - `tanh(Wa(query) + Ua(keys))`
   - shape: `(batch, seq_len, hidden_size)`
4. 또 다른 Linear layer 가중치 Va를 적용시켜준다.
   - `Va(tanh(Wa(query) + Ua(keys)))`
   - 각 시점의 query-key 조합을 스칼라 점수(score)로 바꾸어준다.
   - attention weight을 만들기 위해 각 key에 대한 1개의 유사도 점수를 얻는다.
   - `shape: (batch, seq_len, 1)`
5. Va를 통과해 나온 스칼라 점수를 squeeze를 적용시킨다.
   - `scores.squeeze(2)`
   - score를 softmax에 넣기 전, 마지막 차원(=1)을 제거해서 텐서를 `(batch, seq_len)`로 만든다.
   - softmax는 일반적으로 `(batch, seq_len)` 형태의 2D 텐서에 적용하기 때문이다.
   - shape: `(batch, seq_len)`
6. softmax를 통해 점수들을 확률 값으로 바꾸어 어텐션 가중치를 구한다.
   - `weights = F.softmax(scores, dim=-1)`
   - 각 인코더 hidden state에 대한 점수(score)를 확률값으로 변환
   - 모든 스코어를 0~1 사이 값으로 정규화 시키고 합이 1이 되도록 만든다.
   - `shape: (batch, seq_len)`

`score = vᵀ * tanh(Wa * query + Ua * key)`

<br>

### 컨텍스트 벡터

디코더가 예측을 할때, 입력 시퀀스에서 어디에 집중할지를 반영한 정보인 어텐션 가중치를 사용한다.

“어텐션 가중치”를 “`encoder hidden states`"에 곱해서 batch-wise 행렬곱(torch.bmm)으로 계산한다.

디코더가 다음 단어를 예측할때 참고하는 요약 정보이다.

<br>

1. 행렬곱을 가능하기 위해 텐서 차원을 변경해준다.
   - `weights = attention_weights.unsqueeze(1)`
   - shape: `(batch, 1, seq_len)`
   - 각 인코더 타임스텝에 대한 어텐션 확률 (softmax 결과)
   - 예: `[0.1, 0.2, 0.3, 0.4]` (1개의 시퀀스 길이 4)
2. 각 배치마다 어텐션 가중치를 인코더 hidden state에 가중합해서 요약정보를 생성한다.

   - `context = torch.bmm(weights, encoder_hidden_states)`
   - 가중합: 각 항목을 곱해서 더한 값
   - 어텐션 가중치는 시퀀스 단어의 유사도 점수이다. 따라서 시퀀스 단어 개수만큼 존재한다.
     - `α₁, α₂, α₃, α₄`→ 각 hidden state에 대응하는 softmax된 score
   - 인코더 hidden state도 각 시퀀스 단어 마다 존재한다.
     - `h₁, h₂, h₃, h₄` → 각 단어마다 1개씩
   - 어텐션 가중치와 인코더 hidden state는 각 단어마다 1:1로 대응된다. 1:1 대응되는 것을 가중합 해준다.

     - $\text{context} = \alpha_1 h_1 + \alpha_2 h_2 + \alpha_3 h_3 + \alpha_4 h_4$

     ```python
     context =
     0.2 * [1.0, 2.0] +
     0.3 * [0.0, 1.0] +
     0.5 * [3.0, 4.0]

     = [0.2, 0.4] +
       [0.0, 0.3] +
       [1.5, 2.0]

     = [1.7, 2.7]

     ```

   | 변수명                  | shape                           | 의미                                                       |
   | ----------------------- | ------------------------------- | ---------------------------------------------------------- |
   | `weights`               | `(batch, 1, seq_len)`           | 디코더가 각 인코더 타임스텝에 대해 할당한 attention weight |
   | `encoder_hidden_states` | `(batch, seq_len, hidden_size)` | 인코더의 각 타임스텝에서 생성된 hidden state들             |
   | `context`               | `(batch, 1, hidden_size)`       | 인코더 전체를 요약한 벡터 (디코더의 다음 예측에 사용됨)    |

<br>

## 코드 구현

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).init()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).init()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batchsize, 1, dtype=torch.long, device=device).fill(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing 포함: 목표를 다음 입력으로 전달
                decoder_input = targettensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                , topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # 입력으로 사용할 부분을 히스토리에서 분리

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

		def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
```
