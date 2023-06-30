# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 사재원
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
def generate_grad_cam(model, activation_layer, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    # Grad cam에서도 cam과 같이 특정 레이어의 output을 필요로 하므로 모델의 input과 output을 새롭게 정의합니다.
    # 이때 원하는 레이어가 다를 수 있으니 해당 레이어의 이름으로 찾은 후 output으로 추가합니다.
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(activation_layer).output, model.output])
    
    # Gradient를 얻기 위해 tape를 사용합니다.
    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(tf.expand_dims(img_tensor, 0))
        
        loss = pred[:, class_idx] # 원하는 class(여기서는 정답으로 활용) 예측값을 얻습니다.
        output = conv_output[0] # 원하는 layer의 output을 얻습니다.
        grad_val = tape.gradient(loss, conv_output)[0] # 예측값에 따른 Layer의 gradient를 얻습니다.
    #print(conv_output.shape)
    
    weights = np.mean(grad_val, axis=(0, 1)) # gradient의 GAP으로 weight를 구합니다.
    #print(weights.shape)
    #print(conv_output.shape[0:2])
    grad_cam_image = np.zeros(dtype=np.float32, shape=output.shape[0:2])
 
    for k, w in enumerate(weights):
        # output의 k번째 채널과 k번째 weight를 곱하고 누적해서 class activation map을 얻습니다.
        grad_cam_image += w * output[:, :, k]
    print(grad_cam_image.shape)
    grad_cam_image = tf.math.maximum(0, grad_cam_image)
    grad_cam_image /= np.max(grad_cam_image)
    grad_cam_image = grad_cam_image.numpy()
    grad_cam_image = cv2.resize(grad_cam_image, (width, height))
    return grad_cam_image
 ```
 > 자세한 주석으로 이해하기 수월했습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
 ```python
origin_image = item['image'].astype(np.uint8)
cam_image_3channel = np.stack([cam_image*255]*3, axis=-1).astype(np.uint8)

cam_blended_image = visualize_cam_on_image(cam_image_3channel, origin_image)
plt.imshow(cam_blended_image)
plt.show()
 ```
 > 적절한 자료형 변환과 함수 사용 등을 통해 에러 유발 가능성을 없애셨습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
def generate_cam(model, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    # 학습한 모델에서 원하는 Layer의 output을 얻기 위해서 모델의 input과 output을 새롭게 정의해줍니다.
    cam_model = tf.keras.models.Model([model.inputs], [model.layers[-3].output, model.output])
    conv_outputs, predictions = cam_model(tf.expand_dims(img_tensor, 0))# 축 하나 추가 인풋 size를 맞추기위해
    conv_outputs = conv_outputs[0, :, :, :]## 7 7 2048
    
    # 모델의 weight activation은 마지막 layer에 있습니다.
    class_weights = model.layers[-1].get_weights()[0] # 2048, 128
    
    cam_image = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])# 7,7
    #7,7,2048  2048, 1 내적 계산

    """
    for i, w in enumerate(class_weights[:, class_idx]):# 2048, 1 특정 class 에 해당하는 가중치값만 가져와서 
        # conv_outputs의 i번째 채널과 i번째 weight를 곱해서 누적하면 활성화된 정도가 나타날 겁니다.
        cam_image += w * conv_outputs[:, :, i] # 채널별*가중치 합연산
    cam_image /= np.max(cam_image) # activation score를 normalize합니다.
    cam_image = cam_image.numpy()
    print(np.max(test))
    print(cam_image.shape)
    print(np.max(cam_image))
    """
    
    cam_image=np.dot(conv_outputs,class_weights[:,class_idx]) # 채널별*가중치 합연산과 동일
    cam_image /= np.max(cam_image) # activation score를 normalize합니다. 최댓값을 1로 만든다
    cam_image[cam_image < 0]=0
    cam_image = cv2.resize(cam_image, (width, height)) # 원래 이미지의 크기로 resize합니다.
    #resize를 해도 되는 이유 
    #conv 레이어를 통해 이미지를 채널별로 나누어 각 특징을 함축적으로 포함되어있기때문이라고 생각한다
    #그렇다면 fully connected layer 경우에는 왜 사용하지 않을까?
    
    #fully connected layer를 통과하게되면 각 채널별을 하나로 합침으로써 
    #이미지의 위치정보를 깨뜨리기 때문에 사용하지 못하는 걸로 생각한다
    
    return cam_image
 ```
 > 이해하고 더 나아가 생각하는 과정까지 적어주셨습니다.

- [x] **5. 코드가 간결한가요?**  
  
 ```python
fig = plt.figure(figsize=(15,10))
img=[item['image'],grad_cam_image, grad_cam_blended_image, src_bbox, grad_cam_bbox]
title=['src','Grad_CAM','blended','src_bbox','cam_bbox']
for i in range(6):
    
    ax = fig.add_subplot(2,3,i+1)
    ax.axis('off')
    if i==5:
        ax.text(0.5,0.5,"IoU = {0}".format(get_iou(src_rect, grad_cam_rect)),fontsize=20, horizontalalignment='center')
        break
    ax.imshow(img[i])
    ax.set_title(title[i])
    
    
plt.show()
 ```
 > 다음과 같이 간결한 코드와 더불어 한눈에 결과물을 확인할 수 있게 정리하셨습니다.

