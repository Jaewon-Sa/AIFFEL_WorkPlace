
# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 사재원
- 리뷰어 : 장승우

## **PRT(PeerReviewTemplate)**  
------------------  
- [O] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

네 결과 이미지와 테스트가 정상적으로 나왔습니다~


- [O] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  

네네~
```python
class MJDatasetSequence(Sequence):
    # 객체를 초기화 할 때 lmdb를 열어 env에 준비해둡니다
    # 또, lmdb에 있는 데이터 수를 미리 파악해둡니다
    def __init__(self, 
                 dataset_path,
                 label_converter,
                 batch_size=1,
                 img_size=(100,32),
                 max_text_len=22,
                 is_train=False,
                 character='') :
        
        self.label_converter = label_converter
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.character = character
        self.is_train = is_train
        self.divide_length = 100

        self.env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))
            self.index_list = [index + 1 for index in range(self.num_samples)]
        

    def __len__(self):
        return math.ceil(self.num_samples/self.batch_size/self.divide_length)
    
    # index에 해당하는 image와 label을 읽어옵니다
    # 위에서 사용한 코드와 매우 유사합니다
    # label을 조금 더 다듬는 것이 약간 다릅니다
```

- [XX] **3. 코드가 에러를 유발할 가능성이 있나요?**
경고 정도만 있으나 크게 상관은 없어보여요~
```python
# 모델을 컴파일 합니다
optimizer = tf.keras.optimizers.Adadelta(lr=0.1, clipnorm=5)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
/opt/conda/lib/python3.9/site-packages/keras/optimizer_v2/optimizer_v2.py:355: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  warnings.warn(
```

- [O] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
네네, 물어보는 질문에 대답도 잘 해주시고 직접 코드도 작성했습니다.
```python
def detect_text(img_path):
    # 배치 크기를 위해서 dimension을 확장해주고 kera-ocr의 입력 차원에 맞게 H,W,C로 변경합니다.
    img = Image.open(img_path)
    img = img.convert('RGB')
    np_img = np.array(img)
    #np_img = np.transpose(np_img, (1, 0, 2))
    img_pil = copy.deepcopy(img)
    
    img_draw = ImageDraw.Draw(img_pil)
    np_img = np.expand_dims(np_img, 0)
    # 배치의 첫 번째 결과만 가져옵니다.
    detector_img = detector.detect(np_img)
    ocr_result = detector_img[0]
    result_img = img_pil
    
    cropped_imgs = []
    for text_result in ocr_result:
        img_draw.polygon(text_result, outline='red')
        x_min = text_result[:,0].min() - 5
        x_max = text_result[:,0].max() + 5
        y_min = text_result[:,1].min() - 5
        y_max = text_result[:,1].max() + 5
        word_box = [x_min, y_min, x_max, y_max]
        cropped_imgs.append(img_pil.crop(word_box))

    return result_img, cropped_imgs
```

- [O] **5. 코드가 간결한가요?**  
네네, 순서에 맞게 잘 작성했습니다.
```python
def recognize_img(pil_img, input_img_size=(100,32)):
    # TODO: 잘려진 단어 이미지를 인식하는 코드를 작성하세요!
    pil_img = pil_img.resize(input_img_size)
    np_img = np.array(pil_img)
    np_img = np.transpose(np_img, (1, 0, 2))
    np_img = np_img[np.newaxis, :, :, :]
    output = model_pred.predict(np_img)
    result = decode_predict_ctc(output, chars="-"+TARGET_CHARACTERS)[0].replace('-','')
    return result
```  
