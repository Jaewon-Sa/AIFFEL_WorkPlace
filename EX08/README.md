# AIFFEL Campus Online 4th Code Peer Review
----  
- 코더 : 사재원
- 리뷰어 : Donggyu Kim

## **PRT(PeerReviewTemplate)**  
------------------  
- [o] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

|Task|Contents|isDone|
|---|---|---|
|1. A learning dataset was constructed through Korean language preprocessing.|The processes of handling spaces and special characters, tokenizing, and building parallel data were properly carried out.|o|
|2. By implementing the transformer model, the Korean chatbot model was trained normally.|The implemented transformer model converged stably when learning Korean parallel data.|o|
|3. Implemented a function that responds in Korean to Korean input sentences.|Returned an answer in Korean appropriate to the context of the Korean input sentence.|o|

Evidences
1-1. spaces and special characters
```python
# 전처리 함수
def preprocess_sentence(sentence):
  # 입력받은 sentence를 소문자로 변경하고 양쪽 공백을 제거
    sentence = sentence.lower().strip() # [[YOUR CODE]]

  # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
  # 예를 들어서 "I am a student." => "I am a student ."와 같이
  # student와 온점 사이에 거리를 만듭니다.
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

  # (한글 ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다. 
    sentence = re.sub('[^ㄱ-ㅎ가-힣.?!,]+', " ", sentence)
    sentence = sentence.strip()
    return sentence
```
1-2. tokenize
```python
#단어 정수인코딩,패딩, 단어길이 초과시 삭제
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
  
    for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 15 이하인 경우에만 데이터셋으로 허용
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
  
  # 최대 길이 15로 모든 데이터셋을 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
    return tokenized_inputs, tokenized_outputs
```

1-3. building parallel data
```python
#배치사이즈 설정
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더는 이전의 target을 다음의 input으로 사용합니다.
# 이에 따라 outputs에서는 START_TOKEN을 제거하겠습니다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```

2-1. Train was done without any errors.
```python
EPOCHS = 50
model.fit(dataset, epochs=EPOCHS, verbose=1)

'''
Output:
Epoch 50/50
168/168 [==============================] - 5s 31ms/step - loss: 0.0200 - accuracy: 0.5764
'''
```

2-2. There is no diagrams to proof  loss and accuracy.

3-1. Implemented a function to inference about Korean sentence.
```python
def sentence_generation(sentence):
  # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
    prediction = decoder_inference(sentence)

  # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
    predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

    print('입력 : {}'.format(sentence))
    print('출력 : {}'.format(predicted_sentence))

    return predicted_sentence
    
sentence_generation('안 녕 하 세 요')      
```

3-2. Got an acceptable responses.
```
(1, 3)
입력 : 안녕하세요
출력 : 안녕하세요 .
(1, 5)
입력 : 퇴근하고 뭐해?
출력 : 몸과 마음이 좀 쉬어야 할 것 같아요 .
(1, 6)
입력 : 하이하이
출력 : 제가 놀아드리고 싶네요 .
(1, 10)
입력 : 안 녕 하 세 요
출력 : 네 말씀하세요 .
```

- [] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  
- [] **3. 코드가 에러를 유발할 가능성이 있나요?**
  
- [] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  

- [] **5. 코드가 간결한가요?**  

    
## ** 모델 학습 코드 리뷰 **  
------------------  
         
