# AIFFEL Campus Online 4th Code Peer Review
----  
- 코더 : 사재원
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  
```python
segvalues, output = model.segmentAsPascalvoc(img_path) 
# segmentAsPascalvoc()함수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴, 
#입력으로는 경로를 입력해야함
#분할 은 pacalvoc 데이터로 학습된 모델을 이용
# segvalues=output 설명, output=이미지
```
함수의 역할과 입출력에 대한 자세한 설명이 있어 이해가 수월했습니다. 

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
  
```python
def image_segment(img, path, model, label_num=0):# 특정 클래스 한 종류만 검출하는 방식으로 진행
    segvalues, output = model.segmentAsPascalvoc(img_path) 
    print(f"출력이미지 크기={output.shape}")
    
    for class_id in segvalues['class_ids']: #검출된 클래스들
        print(LABEL_NAMES[class_id])
        
    seg_color = (colormap[label_num][2], colormap[label_num][1], colormap[label_num][0]) # rgb -> bgr
    seg_map = np.all(output==seg_color, axis=-1) 
     # 특정 축 기준 output 이미지가 seg_color(label 색상)랑 같을 때 true false 반환

    img_mask = seg_map.astype(np.uint8) * 255#img_mask 그레이스케일
    # 255와 0을 적당한 색상으로 바꿔봅니다
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)#3채널로 변환
    img_color_seg = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0.0)#색상으로 클래스 구분

    
    img_orig_blur = cv2.blur(img, (13,13))#13*13 전체이미지에 마스크로 블러 효과 주기

    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)#
    img_bg_mask = cv2.bitwise_not(img_mask_color)# cv2.bitwise_not(): 이미지 반전(2진수변환 후 비트연산)

    # cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
    # 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
    # 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다. 결국 사람이 사라지고 배경만 남아요!
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask) #겹치는 부분 반환
    img_concat = np.where(img_bg_mask==0,img,img_bg_blur)
    
    return output, color_mask, img_color_seg, img_concat # 전체 세그멘테이션 이미지, 지정한 클래스 검색 이미지 2장, 최종블러이미지
    
```
함수로의 간결한 정리와 함께 중간 결과를 print하여 확인하면서 에러를 방지습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
```python
def image_segment(img, path, model, label_num=0):# 특정 클래스 한 종류만 검출하는 방식으로 진행
    segvalues, output = model.segmentAsPascalvoc(img_path) 
    print(f"출력이미지 크기={output.shape}")
    
    for class_id in segvalues['class_ids']: #검출된 클래스들
        print(LABEL_NAMES[class_id])
        
    seg_color = (colormap[label_num][2], colormap[label_num][1], colormap[label_num][0]) # rgb -> bgr
    seg_map = np.all(output==seg_color, axis=-1) 
     # 특정 축 기준 output 이미지가 seg_color(label 색상)랑 같을 때 true false 반환

    img_mask = seg_map.astype(np.uint8) * 255#img_mask 그레이스케일
    # 255와 0을 적당한 색상으로 바꿔봅니다
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)#3채널로 변환
    img_color_seg = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0.0)#색상으로 클래스 구분

    
    img_orig_blur = cv2.blur(img, (13,13))#13*13 전체이미지에 마스크로 블러 효과 주기

    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)#
    img_bg_mask = cv2.bitwise_not(img_mask_color)# cv2.bitwise_not(): 이미지 반전(2진수변환 후 비트연산)

    # cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
    # 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
    # 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다. 결국 사람이 사라지고 배경만 남아요!
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask) #겹치는 부분 반환
    img_concat = np.where(img_bg_mask==0,img,img_bg_blur)
    
    return output, color_mask, img_color_seg, img_concat # 전체 세그멘테이션 이미지, 지정한 클래스 검색 이미지 2장, 최종블러이미지
    
    
```
과정을 이해하고 이를 함수로 정리 및 구현하셨습니다.

```python
output, color_mask, img_color_seg, img_concat = image_segment(sofa, img_path, model, 3)
```
검출하려는 클래스에 해당하는 라벨의 숫자를 활용하셨습니다.

- [x] **5. 코드가 간결한가요?**  
  
```python

fig=plt.figure(figsize=(20,10))
title=["output","color_mask","img_color_seg","img_concat"]
img=[output, color_mask, img_color_seg, img_concat]
for i in range(4):
    ax=fig.add_subplot(1, 4, i+1)
    if i==0:
        ax.imshow(img[i])
        ax.set_title(title[i])
    else:
        ax.imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
        ax.set_title(title[i])
    ax.axis("off")
        
plt.show()
```

정리를 통해 결과물을 한번에 확인할 수 있었습니다.



## **참고링크 및 코드 개선 여부**  
------------------  
- 내용에서 말씀해주신 depth 관련하여 (노드에 있던) 링크를 첨부합니다.  
 https://sites.google.com/view/struct2depth    
 https://ai.googleblog.com/2020/04/udepth-real-time-3d-depth-sensing-on.html
