

# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 사재원
- 리뷰어 : 장승우

## **PRT(PeerReviewTemplate)**  
------------------  
- [O] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
```python
네~ 이미지에 박스랑 스티커가 정상적으로 출력되고 있습니다.
```

- [O] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
```python
네~ 주석을 잘 달아주셔서 이해하기 쉬웠어요~
def sticker_photo(img, list_landmarks, rects):
    for rect, landmark in zip(rects, list_landmarks): # 얼굴 영역을 저장하고 있는 값과 68개의 랜드마크를 저장하고 있는 값으로 반복문 실행
    #34_index는 코끝  52_index는 윗입술 가운데
        x = landmark[33][0] # 얼굴의 인중 x좌표
        y = landmark[33][1] + (landmark[51][0]-landmark[34][0])//2 # 얼굴의 인중 y좌표
        w = h = int(round(rect[2]-rect[0])*0.8) # 얼굴 영역의 크기
        #print(x)
        sticker = cv2.resize(img_sticker, (w,h))# 얼굴 크기만큼 이미지 리사이징
        c_pos_x, c_pos_y=w//2,h//2 #스티커 중심좌표
```

- [XX] **3. 코드가 에러를 유발할 가능성이 있나요?**
```python
validation dataset 관련 에러가 있으나, 실제 코드에선 사용하지 않아서 전체 동작에선 에러발생하지 않아요~
#나중에 꼭 확인할 것 valid_dataset 테스트용
data = valid_dataset.take(1)
for a,b in data:
    print(b.shape)
```

  
- [O] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
```python
네~ 질문한 것에 대해 답변도 잘 해주시고 코드에 해당하는 값도 주석에 달아주셨어요~
        img[y:y2,x:x2] =cv2.addWeighted(sticker_area, 0.3, np.where(sticker==0,sticker,sticker_area).astype(np.uint8), 0.7,0)
        #(599,800,3) (599,800,3) (450,754,3) #살짝 반투명으로 적용하기
```

- [O] **5. 코드가 간결한가요?**  
```python
네~ 중복값이나 중심좌표 설정할때 한 줄로 표현했어요~
        x = landmark[33][0] # 얼굴의 인중 x좌표
        y = landmark[33][1] + (landmark[51][0]-landmark[34][0])//2 # 얼굴의 인중 y좌표
        w = h = int(round(rect[2]-rect[0])*0.8) # 얼굴 영역의 크기
        #print(x)
        sticker = cv2.resize(img_sticker, (w,h))# 얼굴 크기만큼 이미지 리사이징
        c_pos_x, c_pos_y=w//2,h//2 #스티커 중심좌표
```

