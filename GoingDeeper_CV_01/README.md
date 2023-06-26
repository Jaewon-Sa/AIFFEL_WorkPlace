# AIFFEL Campus Online 4th Code Peer Review(6/26)
### coder : 사재원
### Reviewer : 김신성
-----------------------------------------------------------------------
## I review project 5 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Project Going Depper 1
- Dataset : Tensorflow official cats_vs_dogs dataset
- Problem : Ablation study between ResNet and Plain Net

### Data, Pre-trained weight load
```python
# 데이터 처리 에러가 나서 임시방편으로 새로 만듬 원인불명..
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 데이터 전처리 및 증강
image_size = (224, 224)  # 이미지 크기 설정
batch_size = 32

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rescale=1./255,validation_split=0.8  # 이미지 스케일 조정
)

# 훈련 데이터셋 생성
train_dataset = datagen.flow_from_directory(
    Path,
    subset="training",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

datagen = ImageDataGenerator(
    rescale=1./255,validation_split=0.002  # 이미지 스케일 조정
)

# 훈련 데이터셋 생성
validation_dataset = datagen.flow_from_directory(
    Path,
    subset='validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

print(train_dataset)
```
"It's easy now :)"

### Model define
```python
def residual_Box(inputs, is_resnet,is_34):
    if is_34:
        for block_num,(repeat_count,filters) in enumerate(zip([3,4,6,3],[64,128,256,512])):
            for i in range(repeat_count):
                if block_num!=0 and i==0:
                    x = keras.layers.Conv2D(filters, 3, strides=2, padding='same',
                                                name="block_stage{0}_step{1}".format(block_num,i))(inputs)
                else:
                    x = keras.layers.Conv2D(filters, 3, padding='same',
                                                name="block_stage{0}_step{1}".format(block_num,i))(inputs)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.ReLU()(x)
                x = keras.layers.Conv2D(filters, 3, padding='same')(x)
                x = keras.layers.BatchNormalization()(x)
                if is_resnet:
                    if inputs.shape[1] != x.shape[1]: # 이미지 크기 다른경우
                        inputs = tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same')(inputs)
                        inputs = tf.keras.layers.BatchNormalization()(inputs)
                        
                    x = tf.keras.layers.Add()([x, inputs])
                    
                inputs = keras.layers.ReLU()(x)
            
    else: #50
        for block_num,(repeat_count,filters) in enumerate(zip([3,4,6,3],[64,128,256,512])):
            for i in range(repeat_count):
                if block_num!=0 and i==0:
                    x = keras.layers.Conv2D(filters, 1, strides=2, padding='same',
                                                name="block_stage{0}_step{1}".format(block_num,i))(inputs)
                else:
                    x = keras.layers.Conv2D(filters, 1, padding='same',
                                                name="block_stage{0}_step{1}".format(block_num,i))(inputs)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.ReLU()(x)
                    
                x = keras.layers.Conv2D(filters, 3, padding='same')(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.ReLU()(x)
                    
                x = keras.layers.Conv2D(filters*4, 1, padding='same')(x)
                x = keras.layers.BatchNormalization()(x)
                    
                if is_resnet:
                    if inputs.shape[1] != x.shape[1]: # 이미지 크기 다른경우 
                        inputs = tf.keras.layers.Conv2D(filters*4, 1, strides=2, padding='same')(inputs)
                        inputs = tf.keras.layers.BatchNormalization()(inputs)
                        
                    if inputs.shape[-1] != x.shape[-1]: # 차원수가 다를때 첫번째 (residual layer 경우에만)
                        inputs = tf.keras.layers.Conv2D(filters*4, 1, strides=1, padding='same')(inputs)
                        inputs = tf.keras.layers.BatchNormalization()(inputs)   
                    x = tf.keras.layers.Add()([x, inputs])
                        
                inputs = keras.layers.ReLU()(x)
 
    return inputs
        
    
                

def build_model(input_shape, num_classes, is_resnet, is_34):
    inputs = keras.layers.Input(shape=input_shape)
    #x = keras.layers.experimental.preprocessing.Rescaling(1.0/255)(inputs)
    x = keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs) #112,112,64
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same',name='first_maxpooling')(x) ##56,56,64
    x = residual_Box(x,is_resnet,is_34)
    output = keras.layers.AveragePooling2D(padding = 'same')(x)
    output = keras.layers.Flatten(name='flatten')(output)
    if num_classes==2:
        output = keras.layers.Dense(1, activation='sigmoid')(output)
        #output = keras.layers.Dense(num_classes, activation='softmax')(output)
    else:
        output = keras.layers.Dense(num_classes, activation='softmax')(output)
    
    model = tf.keras.Model(inputs, output)
    return model

resnet_34_model = build_model((224,224,3),2,True,True)
Plain_34_model = build_model((224,224,3),2,False,True)
resnet_50_model = build_model((224,224,3),2,True,False)
Plain_50_model = build_model((224,224,3),2,False,False)
```
#### Outputs

"Overall, the coding is clean, there are not too many annotations, and it is appropriately used in important parts. It was also great to visualize the structure of the model using the plot_model"
=> Excellent!!

### Ablation study
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(history_resnet34.history['val_accuracy'], label='ResNet-34')
ax1.plot(history_Plain34.history['val_accuracy'], label='Plain-34')
ax1.set_title('Validation Accuracy 34')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history_resnet50.history['val_accuracy'], label='ResNet-50')
ax2.plot(history_Plain50.history['val_accuracy'], label='Plain-50')
ax2.set_title('Validation Accuracy 50')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss') #Accuracy 로 바쭤야 한다
ax2.legend()
  ```
 "I received loss and acc well from history and drew it so that it can be compared properly. As intended in this project, the difference in acc between the presence and absence of a resistive black was well derived and visualized"
 => Very Good!!
  

 ![image](https://github.com/Jaewon-Sa/AIFFEL_WorkPlace/assets/91248817/f9385015-f1f9-452a-8410-0a65d5cce184)



### Conclusion
전체적으로 코드가 깔끔하고 주석도 적절한 부분에 잘 들어가 있다.
Residual block 이 있는 경우와 없는 경우의 loss, acc 차이를 잘 도출해낸 성공적인 Ablation study project였습니다. Loss 도 서로 비교하면 더 좋을것 같습니다!

수고하셨습니다!!

- [O] **1.Did the code work properly and fix the given issue?**
- [O] **2.Did I look at the comments and understand the author's code? And it is suitable?**  
- [NO] **3.Is there a possibility that the code will cause an error?**
- [O] **4.Did the code writer understand and write the code correctly?**  
- [O] **5.Is the code concise and expandable?**  
-----------------------------------------------------------------------
