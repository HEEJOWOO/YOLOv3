# [YOLOv3 : An Incremental Improvement](https://arxiv.org/abs/1804.02767)

![image](https://user-images.githubusercontent.com/61686244/134756128-9abfe939-e392-4c0a-abce-af7aeb70e26d.png)

Bounding Box Prediction
-----------------------
![image](https://user-images.githubusercontent.com/61686244/134756209-d76640c0-6c6b-431b-9db5-541c9f52604f.png)


  * 기존 YOLOv2에서 bounding box를 예측할 때 t_x, t_y, t_w, t_h를 구한 후 위 그림과 같이 b_x, b_y, b_w, b_h로 변형 한 후 L2 Loss를 통해 학습 진행, c_x, c_y는 grid cell의 좌상단 offset
  * t_x, t_y, t_w, t_h -> b_x, b_y, b_w, b_h 변형에 대한 추가 설명
    - YOLO와 Anchor box를 동시에 사용할때 tx, ty는 제한된 범위가 없기 때문에 anchor box는 이미지 내의 임의의 지점에 위치할 수 있다는 문제가 발생
    - 최적화된 값을 찾기 까지 오랜시간이 걸려 모델은 초기에 불안정하게 됨 
    - 이러한 문제를 해결하기 위해 YOLO의 방식을 사용하여 grid cell에 상대적인 위치 좌표를 예측하는 방법을 선택, sigmoid를 사용하여 0~1사이의 값을 갖도록 변경
    - 예측하는 버위가 정해짐으로써 네트워크는 안정적으로 학습을 진행할 수 있음
    - Dimension Clustering를 통해 최적의 prior를 선태갛고, anchor box 중심부 좌표를 예측함으로 recall 값이 YOLOv2에서 5%정도 향상

![image](https://user-images.githubusercontent.com/61686244/134756398-422b4c1b-441e-44f4-b993-8aa04243c377.png)

  * YOLOv3에서는 위 공식처럼 거꾸로 적용시켜 t_*로 변형 시킨 후 L1 loss를 통해 학습시키는 방식을 선택 

  * bounding box마다 objectness score를 logistic function을 이요하여 구함
  * anchor box와 GT box와의 IoU 값이 가장 높은 box만 선택
  * low, medium, high resolution에서 각 3개의 bounding box를 만들어냄 총 9개의 bounding box중 IoU가 가장 높은것을 선택하여 objectness score를 1로 설정해줌


Class Prediction
----------------
  * Multi Label Classification을 수행하기 위해 softmax가 아닌 sigmoid + Cross Entropy Loss인 binary Cross Entropy를 사용
  * 즉, 클래스에 대해 softmax를 사용하는 것이나닌 각 클래스에 대해 독립적으로 적용하깅 위해 sigmoid를 사용

Prediction Across Scales
------------------------
![image](https://user-images.githubusercontent.com/61686244/134756625-61c9250a-ec08-412c-a56f-fbc0f5c9bda6.png)

  * 416x416 크기의 이미지를 네트워크에 입려갛여 특징 맵의 크기가 52x52, 26x26, 13x13이 되는 층의 특징 맵을 추출하여 이용
  * 각 scale의 특징 맵의 출력 수가 셀의 bbox 갯수 x (bbox offset + objetcness score + 클래스 갯수) = 3x(4+1+80) = 255 되도록 1x1 Conv를 통해 채널 수 조정 
  * COCO dataset 기준 k-means clustering를 이용하여 bounding box prior를 설정, 9 cluster, 3 scale
  * (10x13), (16x30), (33x23), (30x61), (62x45), (59x119), (116x90), (156x198), (373x326)
  
Feature Extractor
-----------------
![image](https://user-images.githubusercontent.com/61686244/134756910-24a2ec25-aeb2-409d-88bb-a3ae4f74f60e.png)

![image](https://user-images.githubusercontent.com/61686244/134756911-360b5f9f-b1f9-4217-9ed1-3852affe30c0.png)

  * YOLOv2에서 backbone으로 사용되었던 Darknet19 대신 YOLOv3에서는 residual connection을 사용하는 Darknet53을 backbone network로 사용
  * RestNet 101보다 1.5배 빠르며, ResNet 152와 비슷한 성능을 보이지만 2배 이상 빠름 




