# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 사재원
- 리뷰어 : 김용석


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  > 위 항목에 대한 근거 작성 필수
  > [피드백] 간결하게 작성되었으며, 마크다운으로 설명을 명시하여 이해하기 쉽게 작성되었습니다. 
  > [피드백] 컴퓨터 비전에 대한 지식이 있어 개선할 방향에 대해 이해하고 있었습니다. 
  > [피드백] 노드에서도 더 중요한 내용을 압축하고, 세세한 세팅 값, 함수의 움직임에 대한 주석처리가 너무도 잘되어있었습니다. 


- [o] 3.코드가 에러를 유발할 가능성이 있나요?
  > 이미지 생성 -> landmarke 추출 -> 특징점 발굴 -> 이미지 합성 -> 개선점 발굴까지 깔끔한 정리
  > 코드가 복잡하고 난해하면 에러발생 가능성이 높지만 간결하게 작성되어 발생가능성이 낮고, 이해하기 쉽게 만들어짐.
  
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 6장 학습한 노드에 대한 기본적인 내용이 충실하게 반영되어 있어서, 리뷰어 입장에서도 이해하기가 좋았음.


- [o] 5.코드가 간결한가요?
  > 매우 간결했습니다. 

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.



[ 주요 코드에 대한 리뷰 내용 ]
img_bgr = cv2.imread(my_image_path+'hangain.jpg')  
### cv2.imread() 함수 사용 bgr 형태로 읽음

print(img_bgr.shape)

img_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
### 현재 불러온 이미지 출력
### 현재 블러온 이미지를 bgr형태를 rgb형태로 변형


detector_hog = dlib.get_frontal_face_detector() # 기본 얼굴 감지기를 반환
### dlib.get_frontal_face_detector() 함수를 사용하여 감지기 반환시킴


def get_landmark(img):               
    dlib_rects = detector_hog(img, 1)
    list_landmarks = []
    # 얼굴 영역 박스 마다 face landmark를 찾아냅니다
    # face landmark 좌표를 저장해둡니다
    
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img, dlib_rect)
        #검출된 얼굴에게 landmark 모델 적용
        # 모든 landmark의 위치정보를 points 변수에 저장
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        # 각각의 landmark 위치정보를 (x,y) 형태로 변환하여 list_points 리스트로 저장
        list_landmarks.append(list_points)
        # list_landmarks에 랜드마크 리스트를 저장
        
    return list_landmarks, dlib_rects

list_landmarks, dlib_rects = get_landmark(img)
print(len(list_landmarks[0]))#한개의 얼굴당 68개의 특징점을 가짐
### 전체적으로 얼굴검출 모델과 특징점 68개 점을 통해서 찾는 실행함 수가 적절하게 사용되었습니다. 


def sticker_photo(img, list_landmarks, dlib_rects):
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks): 
        
        # 얼굴 영역을 저장하고 있는 값과 68개의 랜드마크를 저장하고 있는 값으로 반복문 실행
        # 34_index는 코끝  52_index는 윗입술 가운데

        x = landmark[33][0] 
        # 얼굴의 인중 x좌표
        y = landmark[33][1] + (landmark[51][0]-landmark[34][0])//2 
        # 얼굴의 인중 y좌표
        w = h = int(round(dlib_rect.width())*0.8) 
        # 얼굴 영역의 크기

        ### landmark의 주석을 통해 어느위치에 존재하는지를 명확하게 알수있게 잘 설명되었음.
        
        #print(x)
        sticker = cv2.resize(img_sticker, (w,h))
        # 얼굴 크기만큼 이미지 리사이징
        c_pos_x, c_pos_y=w//2,h//2 
        #스티커 중심좌표


        #스티커가 
        #스티커의 중심이 인중으로 오게하기위해
        x-=c_pos_x
        if  x<0:
            sticker=sticker[:,-x:] #넘어간 부분 잘라내기
            x=0
            
        y-=c_pos_y
        if y<0:
            sticker=sticker[-y:,:] #넘어간 부분 잘라내기
            y=0

        ### -음수로 넘어간 부분은 제외함으로써 위치게 올 수 있도록 함수 사용
            
            
        #print(min(y+h,img_show.shape[0]),min(x+h,img_show.shape[1]))

        y2=min(y+sticker.shape[0],img.shape[0])
        x2=min(x+sticker.shape[1],img.shape[1])
        
        ### 옆면으로 늘어나는 스티커나 영역밖으로 벗어날 경우 오류 발생을 방지

        sticker_area = img[y:y2,x:x2]
        #print(sticker_area.shape,sticker.shape)
        img[y:y2,x:x2] =cv2.addWeighted(sticker_area, 0.3, np.where(sticker==0,sticker,sticker_area).astype(np.uint8), 0.7,0)
        #(599,800,3) (599,800,3) (450,754,3) #살짝 반투명으로 적용하기
        
        ### addweighted() 0.3은 원본이미지, 0.7은 스티커 
        ### 흰색배경은 np.where로 제거가 되고 (조건조절로 가능)
        
    plt.imshow(img)
    plt.show()
    return
