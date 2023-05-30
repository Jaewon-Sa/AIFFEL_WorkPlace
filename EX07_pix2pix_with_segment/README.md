# AIFFEL Campus Online 4th Code Peer Review
----  
- 코더 : 사재원
- 리뷰어 : 김용석

## **PRT(PeerReviewTemplate)**  
------------------  
- [] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  : 정상적으로 동작하고 주어진 문제를 잘 해결했습니다. 
  
- [] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  : 구두로 해당 내용에 대해서 설명을 들을 수 있어서 코드가 이해가 되었습니다.  
  
- [] **3. 코드가 에러를 유발할 가능성이 있나요?**
  : 기본적인 실습코드를 활용하여 추가적인 에러를 유발할 가능성은 낮다고 생각합니다. 
  
- [] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
- : 네네, 구두로 이해하고 있어 제대로 이해하고 작성된 것으로 판단됩니다. 

- [] **5. 코드가 간결한가요?**  
- : 간결하게 잘 작성되었습니다. 
  
    
## ** 모델 학습 코드 리뷰 **  
------------------  

모델 설명: GAN과 U-net을 결합하여 더 향상된 성능을 얻고 있으며, GAN은 뛰어난 생성능력을 제공하고, U-Net은 생성 과정에 구조적인 문맥을 추가하여 더 정확하게 생성하게 함.

EPOCHS = 30

generator = UNetGenerator()     # U-NetGenerator로 모델을 만들었으며, 입력값('sketch')를 받아 출력값('colored')를 생성
discriminator = Discriminator() # discriminator는 'generator'가 생성한 이미지가 실제 이미지인지 생성된 이미지인지 구분하는 역할을 함.
L1loss,G_loss,D_loss=[],[],[]   # 이 모델에서는 3가지 손실 함수를 사용
for epoch in range(1, EPOCHS+1):
    for i, (sketch, colored) in enumerate(train_images):
        g_loss, l1_loss, d_loss = train_step(sketch, colored)
                # g_loss : generator Loss(G_loss), 판별기가 생성된 이미지를 실제 이미지로 분류한 정보를 측정, 손실을 최소화 시키는 것이 목적
                # l1_loss : 생성된 이미지와 목표 이미지간의 MAE(Mean Absolute Error)를 측정, 손실을 최소화 시키고, 더 정확한 색상을 생성하는 것이 목적
                # d_loss : discriminator가 실제 이미지와 생성된 이미지를 얼마나 잘 구별하는지 측정, 손실을 최소화하는 것이 목적
                                
        # 10회 반복마다 손실을 출력합니다.
        if (i+1) % 10 == 0:
            L1loss.append(l1_loss)
            G_loss.append(g_loss)
            D_loss.append(d_loss)
        if (i+1) % 125 == 0:
            print(f"EPOCH[{epoch}] - STEP[{i+1}] \
                        \nGenerator_loss:{g_loss.numpy():.4f} \
                        \nL1_loss:{l1_loss.numpy():.4f} \
                        \nDiscriminator_loss:{d_loss.numpy():.4f}", end="\n\n") 
         # 'EPOCHS' 만큼 반복하며, 각 반복에는 'trian_images'의 배치에 대해 train_step 함수를 호출하여 각 손실을 계산하고, 이를 바탕으로 모델 가중치를 업데이트 함.
         # 각 에폭에서 10회와 125회 마다 손실값을 저장하고 출력. 
         
         

