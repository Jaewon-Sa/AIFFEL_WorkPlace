# AIFFEL Campus Online 4th Code Peer Review
----  
- 코더 : 사재원
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  
  
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  
```python
data.drop_duplicates(subset = ['text'], inplace=True)# inplace=True 는 따로 반환 받지않아도됨
#headlines 는 타겟이기에 중복되도 상관없음
print('전체 샘플수 :', (len(data)))
```

```python
import json

contractions={}
#정규화 시킬 사전 내용이 길어서 따로 json 파일로 저장하고 불러왔습니다
with open("contractions.json", "r") as json_file:
    contractions=json.load(json_file)
print("정규화 사전의 수: ", len(contractions))
```

```python
# 데이터 전처리 함수
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","", sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```
  
확인하는 내용과 코드 작성 이유, 함수의 역할과 입출력에 대한 자세한 설명이 있어 이해가 수월했습니다. 

  
- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
  
```python
clean_text = []

for s in data['text']:
    clean_text.append(preprocess_sentence(s))
# 전처리 후 출력
print("Text 전처리 후 결과: ", clean_text[:0])
print("Text 전처리 후 결과: ", len(clean_text))
```
  
```python
# 잘 변환되었는지 확인
print('input')
print('input ',decoder_input_train[:5])
print('target')
print('decoder ',decoder_target_train[:5])
    
```
진행 과정 사이사이마다 확인을 통해 에러를 방지하셨습니다.

  
- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
```python
src_vocab=20000 #전체 사전에 4퍼센트 정도는없어도 될 거라 판단하여 집합크기를 20000만개로 설정   
```

```python
tar_vocab=10000 #전체 사전에 4퍼센트 정도는없어도 될 거라 판단하여 집합크기를 10000만개로 설정
```
  
빈도 비율을 확인하고 적절한 집합 크기를 설정하셨습니다.  
  
| ---      | Abstractive                      | Extractive   |
|------    |-------                           |------        |
|문법완성도   |심각함                             |대체적으로 괜찮음  |
|요약내용의 질|해당 요약문만 보고는 이해하기 어려울거라 판단|대체로 파악이 가능함|
|문장길이    |대체로 굉장히 짧은 편                  |1~2문장         |
  
다음과 같이 비교 분석을 해주셨습니다.
  
    
- [x] **5. 코드가 간결한가요?**  
  
```python
data['text'] = data['text'].apply(lambda x : np.nan if len(x.split())>text_max_len else x)
data['headlines'] = data['headlines'].apply(lambda x : np.nan if len(x.split())>headlines_max_len else x)
data.dropna(axis=0, inplace=True)
```
  
```python
data['decoder_input'] = data['headlines'].apply(lambda x : 'sostoken '+ x)
data['decoder_target'] = data['headlines'].apply(lambda x : x + ' eostoken')
```
  
lambda식과 inplace=True를 활용해 간결하게 작성해주셨습니다.
 
  
  
    
## **참고링크 및 코드 개선 여부**  
------------------  
- Abstractive summarization validation으로 검색해보다가 비교가 잘 되어있는 것 같아보여서 공유합니다 :D  
[How to Validate OpenAI GPT Model Performance with Text Summarization]  
https://towardsdatascience.com/how-to-validate-openai-gpt-model-performance-with-text-summarization-298978fea764
