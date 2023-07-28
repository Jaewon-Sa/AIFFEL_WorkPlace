
# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 사재원
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**  
------------------  
- [O] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

- [O] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
  ```python
  # Note that (77, 768) is the shape of the text encoding.
  delta = tf.ones_like(encoding) * step_size #잠재 노이즈 추가
  ```
  네 이해가 되었습니다.

- [XX] **3. 코드가 에러를 유발할 가능성이 있나요?**

   > Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.

    > ResourceExhaustedError: {{function_node __wrapped__MatMul_device_/job:localhost/replica:0/task:0/device:GPU:0}} OOM when allocating tensor with shape[150,2457600] and type double on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:MatMul] name: 

    기존의 프롬프트 내용에 대한 제약과 메모리 할당 외 에러는 없었습니다

- [O] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  ```python
  위 방식은 하나의 잠재 벡터를 원형 공간 노이즈 방식으로 새롭게 노이즈를 추가하는 방식인 것으로 보인다.
  ```
  이해하고 코멘트로 내용을 남겨주셨습니다.
- [O] **5. 코드가 간결한가요?**
  ```python
  images = []
  for batch in range(batches):
      images.append(model.generate_image(batched_encodings[batch], batch_size=batch_size))

  images = np.concatenate(images)
  plot_grid(images, "4-way-interpolation-varying-noise.jpg", interpolation_steps)
  ```
  네. 간결해서 가독성이 좋았습니다.
