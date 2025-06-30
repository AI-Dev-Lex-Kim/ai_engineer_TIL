- [Bahdanau Attention 코드 분석](#bahdanau-attention-코드-분석)
  - [Class BahdanauAttention](#class-bahdanauattention)
    - [**init**](#init)
    - [forward](#forward)
  - [class AttnDecoderRNN](#class-attndecoderrnn)
    - [**init**](#init-1)
    - [forward](#forward-1)
    - [디코딩 루프](#디코딩-루프)
    - [루프 종료 후](#루프-종료-후)
    - [def forward\_step](#def-forward_step)

# Bahdanau Attention 코드 분석

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

<br>

## Class BahdanauAttention

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
```

<br>

### **init**

```python
def init(self, hidden_size):
    super(BahdanauAttention, self).init()
    self.Wa = nn.Linear(hidden_size, hidden_size)
    self.Ua = nn.Linear(hidden_size, hidden_size)
    self.Va = nn.Linear(hidden_size, 1)
```

`Wa`: 디코더 현재 hiddens state에 곱할 가중치 weight

`Ua`: 인코더 hiddens state(keys)에 곱할 가중치 weight

`Va`: attention score 유사도 점수를 1차원 스칼라로 줄이기 위한 weight

<br>

### forward

```python
def forward(self, query, keys):
    scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
    scores = scores.squeeze(2)

    weights = F.softmax(scores, dim=-1)
    weights = weights.unsqueeze(1)
    context = torch.bmm(weights, keys)

    return context, weights
```

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
   - `squeeze는 해당 차원이 1이여야만 줄일수있음.`
   - score를 softmax에 넣기 전, 마지막 차원(=1)을 제거해서 텐서를 `(batch, seq_len)`로 만든다.
   - softmax는 일반적으로 `(batch, seq_len)` 형태의 2D 텐서에 적용하기 때문이다.
   - shape: `(batch, seq_len)`
6. softmax를 통해 점수들을 확률 값으로 바꾸어 어텐션 가중치를 구한다.
   - `weights = F.softmax(scores, dim=-1)`
   - 각 인코더 hidden state에 대한 점수(score)를 확률값으로 변환
   - 모든 스코어를 0~1 사이 값으로 정규화 시키고 합이 1이 되도록 만든다.
   - `shape: (batch, seq_len)`
7. 각 배치마다 어텐션 가중치를 인코더 hidden state에 가중합해서 요약정보를 생성한다.
   - `context = torch.bmm(weights, keys)`
   - 가중합: 각 항목을 곱해서 더한 값
   - 어텐션 가중치는 시퀀스 단어의 유사도 점수이다. 따라서 시퀀스 단어 개수만큼 존재한다.
     - `α₁, α₂, α₃, α₄`→ 각 hidden state에 대응하는 softmax된 score
   - 인코더 hidden state도 각 시퀀스 단어 마다 존재한다.
     - `h₁, h₂, h₃, h₄` → 각 단어마다 1개씩

<br>

## class AttnDecoderRNN

```python
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

<br>

### **init**

```python
def __init__(self, hidden_size, output_size, dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.attention = BahdanauAttention(hidden_size)
    self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
    self.out = nn.Linear(hidden_size, output_size)
    self.dropout = nn.Dropout(dropout_p)
```

`self.embedding`

디코더 입력 토큰을 hidden size 차원의 벡터로 변환

<br>

`self.attention`

bahadanau 어텐션 방식 정의

인코더 출력 전체를 보고 디코더가 어느 부분에 집중할지 학습하도록 한다.

<br>

`self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)`

입력크기를 2 \* hidden_size로 정한다.

단어 임베딩 벡터와 컨텍스트 벡터를 concat 했어서 2배가 된다.

현재 단어 임베딩 벡터와 어텐션으로 구한 컨텍스트 벡터를 함께 받아 다음 hidden state 계산한다.

<br>

`self.out = nn.Linear(hidden_size, output_size)`

hidden_size 차원 크기를 output_size 차원으로 만들어준다.

마지막 출력에서 softmax를 하기 위해서 어휘 사전의 어휘 개수에 맞게 맞춰준다.

<br>

`self.dropout = nn.Dropout(dropout_p)`

임베딩 또는 RNN 이후에 dropout 정의한다.

과적합 방지 및 일반화 성능을 향상시킨다.

<br>

### forward

```python
def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
    batch_size = encoder_outputs.size(0)
    decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill(SOS_token)
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
```

디코더에서 다음 단어를 한스텝씩 예측하여 전체 문장을 생성한다.

<br>

`batch_size = encoder_outputs.size(0)`

입력 배치 크기를 참조한다.

<br>

`decoder_input = torch.empty(batchsize, 1, dtype=torch.long, device=device).fill(SOS_token)`

디코더의 첫부분을 시작 토큰인 SOS 토큰으로 채운다.

<br>

`decoder_hidden = encoder_hidden`

디코더의 시작 hidden state를 인코더의 마지막 hidden state로 사용한다.

인코더가 입력 문장의 정보를 요약한 것이므로 디코더 시작점으로 적절하다.

<br>

`decoder_outputs = []`, `attentions = []`

생성한 디코더 출력과 어텐션 가중치를 저장할 리스트를 초기화 해준다.

각 타임스텝 결과를 누적해서 나중에 시퀀스로 반환한다.

<br>

### 디코딩 루프

```python
for i in range(MAX_LENGTH):
        decoder_output, decoder_hidden, attn_weights = self.forward_step(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_outputs.append(decoder_output)
        attentions.append(attn_weights)

        if target_tensor is not None:
            # Teacher forcing 포함: 목표를 다음 입력으로 전달
            decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
        else:
            # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()  # 입력으로 사용할 부분을 히스토리에서 분리

```

`decoder_output, decoder_hidden, attn_weights = self.forward_step(...)`

타임스텝의 디코딩을 실행한다.

임베딩 벡터 + 어텐션 가중치 + GRU를 한 사이클로 처리한다.

<br>

`decoder_outputs.append(decoder_output)`

`attentions.append(attn_weights)`

각 스텝에서 나온 디코더 예측 결과와 어텐션 가중치를 리스트에 저장한다.

<br>

```python
if target_tensor is not None:
decoder_input = targettensor[:, i].unsqueeze(1)
```

학습시:

학습 중에는 정답 시퀀스에서 i번째 토큰을 다음 디코더에 입력으로 사용한다.(Teacher Forcing)

Teacher Forcing: 각 단계 후에 관찰된 시퀀스 값을 RNN에 다시 공급하여 RNN이 실측 시퀀스에 가깝게 유지되도록 하는 작업

<br>

정답 단어를 다음 입력으로 넣지 않고 예측 결과 단어를 다음 입력으로 넣는다면 보다 정확도가 떨어진다.

모델이 초반에 틀린 단어를 예측하면, 그 틀린 단어를 다음 입력으로 또 넣게된다.

오류가 누적되게 된다.

<br>

학습 단계에서 A 다음 B가 아닌 F를 예측하더라도 Teacher forcing으로 교정이 가능하다.

테스트 단계에서는 A 다음 F로 잘못예측 했다면, B라는 답을 알려줄수 없다.

A→F 라는 생소한 시퀀스에 대하여 다음 단어를 예측해야하는 상황이 온다.

A→F 뒤에 예측된 틀린 단어를 포함한 문장에 대해서도 또 다음 단어를 입력으로 넣게 된다.

전체 시퀀스가 틀리게 될 수 있다.

이런것을 exposure bias라고 한다.

현재 언어 모델에서 시퀀스 생성 모델에 대해 근본적으로 해결해야하는 문제로 꼽힌다.

<br>

보완 방법으로는 아래가 있다.

- 학습 중 일정 확률로만 Teacher Forcing을 사용한다.
- teacher_forching_ration를 점점 줄이는 방식으로 보완한다.
  → 초반에는 안정적으로 학습시키고 후반에는 예측을 하게한다.

<br>

```python
else:
	topv, topi = decoder_output.topk(1)
	decoder_input = topi.squeeze(-1).detach()
```

추론시:

현재 디코더 출력에서 가장 높은 확률 단어를 골라 다음 입력으로 사용한다.

추론시에는 예측한 결과를 다음 입력으로 넣어야 실제 문장을 구성하기 때문이다.

<br>

`tensor.topk(k)`

텐서에서 가장 큰 k개 값을 반환한다.

vocab(logits)에서 가장 확률 높은 단어를 하나 고른다.

<br>

softmax을 안거쳐도 되는이유

→어차피 가장 큰 logit을 고르면 가장 높은 확률 단어랑 같기 때문이다.

<br>

### 루프 종료 후

```python
decoder_outputs = torch.cat(decoder_outputs, dim=1)
decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
attentions = torch.cat(attentions, dim=1)

return decoder_outputs, decoder_hidden, attentions
```

`decoder_outputs = torch.cat(decoder_outputs, dim=1)`

전체 시퀀스 결과로 반환하기 위해 `[batch, 1, vocab]` 리스트를 `[batch, seq_len, vocab]` 텐서로 합친다.

`dim=1` 방향으로 이어 붙임

```python
Decoder outputs per time step:
Step 0: [B, 1, V]
Step 1: [B, 1, V]
Step 2: [B, 1, V]
...
Step T: [B, 1, V]

↓ torch.cat(..., dim=1)

Result: [B, T, V]

```

<br>

`decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)`

각 타임스텝의 예측 단어에 log softmax를 적용한다.

loss 계산을 위해 log-probability가 필요하다.

<br>

`attentions = torch.cat(attentions, dim=1)`

어텐션 시각화, 분석 등을 사용하기 위해 각 스텝의 어텐션 가중치를 하나로 합친다.

`[batch, seq_len, encoder_seq_len]`

<br>

### def forward_step

```python
def forward_step(self, input, hidden, encoder_outputs):
    embedded =  self.dropout(self.embedding(input))

    query = hidden.permute(1, 0, 2)
    context, attn_weights = self.attention(query, encoder_outputs)
    input_gru = torch.cat((embedded, context), dim=2)

    output, hidden = self.gru(input_gru, hidden)
    output = self.out(output)

    return output, hidden, attn_weights
```

디코더에서 타임스텝의 입력을 받아 바다나우 어텐션을 적용한다. 이후 GRU를 통해 hidden state과 logits을 생성한다.

<br>

`embedded =  self.dropout(self.embedding(input))`
입력 토큰 인덱스를 임베딩 벡터로 변환하고 dropout을 적용한다.

<br>

`query = hidden.permute(1, 0, 2)`
hidden state 차원을 `[1, batch, hidden]` → `[batch, 1, hidden]` 으로 바꿔준다.

<br>

`context, attn_weights = self.attention(query, encoder_outputs)`

현재 디코더 hidden state(query)와 인코더 hidden states(encoder_outputs)를 이용해서 컨텍스트 벡터와 어텐션 가중치를 계산한다.

<br>

`input_gru = torch.cat((embedded, context), dim=2)`

임베딩 벡터와 컨텍스트 벡터를 마지막 차원에서 결합해준다.

“현재 단어 + “인코더 시퀀스 문맥 요약 정보”를 함께 보고 RNN이 더 정확한 다음 hidden state를 만들 수 있게 된다.

shape: `[batch, 1, 2*hidden]`

<br>

`output, hidden = self.gru(input_gru, hidden)`

- `output`: 전체 시퀀스의 모든 time step 시점의 hidden state출력 (각 시점의 h_t)
- `hidden`: 각 입력 시퀀스 **마지막 time step** hidden state만 모아둠

<br>

`output = self.out(output)`

각 단어가 출력될 확률을 계산하기 위해 새로운 `output`를 어휘 사전 크기만큼의 벡터로 변환해준다.

- shape: `(batch, hidden_size)` → shape: `(batch, vocab_size)`
