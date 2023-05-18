아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 사재원
- 리뷰어 : 임지혜

----------------------------------------------

PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```
#예제1. 
mse = loss(X_test, w, b, y_test)
mse    #2870.6  -> 3000미만

#예제2.
RMSE valuse = 140.6618692885105   #150미만
```

- [O] 주석을 보고 작성자의 코드가 이해되었나요?
```
def gradient(x, w, b, y):
    #dw = (loss(x, w + 0.0001, b, y) - loss(x, w, b, y)) / 0.0001
    #db = (loss(x, w, b + 0.0001, y) - loss(x, w, b, y)) / 0.0001

#gradient계산함수 작성시 주석으로 식을 적어주셔서 이해하기 좋았음 
```
- ['X'] 코드가 에러를 유발할 가능성이 있나요?
```
#에러없이 코드 잘 돌아감 
```
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
```
w = np.random.rand(len(X_train[0,:]))
#w의 수를 x_train의 feature수로 바로 연결될 수 있도록 코딩

def model(w,b,x):
    return x@w+b
#모델의 흐름을 이해하고 @를 사용해 간결하게 표현 -> 모르는 연산자?라 질문드림!
```
- [O] 코드가 간결한가요?
```
def loss(x,w,b,y):
    pred=model(w,b,x)
    return ((y - pred) ** 2).mean()

#loss함수에 MSE함수가 바로 적용될 수 있도록 간단하게 합침 

_, ax = plt.subplots(ncols=3, nrows=2)
sns.countplot(x='year',data=data, ax=ax[0,0])
#countplot을 이용해 subplot보다 간결하게 그래프 표현
```

----------------------------------------------

참고 링크 및 코드 개선
- 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
- 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
