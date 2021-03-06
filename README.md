# [YOLOv3 : An Incremental Improvement](https://arxiv.org/abs/1804.02767)

![image](https://user-images.githubusercontent.com/61686244/134756128-9abfe939-e392-4c0a-abce-af7aeb70e26d.png)

Bounding Box Prediction
-----------------------
![image](https://user-images.githubusercontent.com/61686244/134756209-d76640c0-6c6b-431b-9db5-541c9f52604f.png)


  * 기존 YOLOv2에서 bounding box를 예측할 때 t_x, t_y, t_w, t_h를 구한 후 위 그림과 같이 b_x, b_y, b_w, b_h로 변형 한 후 L2 Loss를 통해 학습 진행, c_x, c_y는 grid cell의 좌상단 offset
  * t_x, t_y, t_w, t_h -> b_x, b_y, b_w, b_h 변형에 대한 추가 설명
    - YOLO와 Anchor box를 동시에 사용할때 tx, ty는 제한된 범위가 없기 때문에 anchor box는 이미지 내의 임의의 지점에 위치할 수 있다는 문제가 발생
    - 최적화된 값을 찾기 까지 오랜시간이 걸려 모델은 초기에 불안정하게 됨 
    - grid cell에 상대적인 위치 좌표를 예측하는 방법을 선택, sigmoid를 사용하여 0~1사이의 값을 갖도록 변경
    - 예측하는 범위가 정해짐으로써 네트워크는 안정적으로 학습을 진행할 수 있음
    - Dimension Clustering를 통해 최적의 prior를 선택하고, anchor box 중심부 좌표를 예측함으로 recall 값이 YOLOv2에서 5%정도 향상

  * bounding box마다 objectness score를 logistic function(데이터가 어떤 범주에 속할 확률을 0에서 1 사이의 값으로 예측하고 그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류해주는 것)을 이용하여 구함
  * anchor box와 GT box와의 IoU 값이 가장 높은 box만 선택
  * low, medium, high resolution에서 각 3개의 bounding box를 만들어냄 총 9개의 bounding box중 IoU가 가장 높은것을 선택하여 objectness score를 1로 설정해줌


Class Prediction
----------------
  * Multi Label Classification을 수행하기 위해 softmax가 아닌 sigmoid + Cross Entropy Loss인 binary Cross Entropy를 사용
  * 즉, 클래스에 대해 softmax를 사용하는 것이 아닌 각 클래스에 대해 독립적으로 적용하기 위해 sigmoid를 사용

Prediction Across Scales
------------------------
![image](https://user-images.githubusercontent.com/61686244/134756625-61c9250a-ec08-412c-a56f-fbc0f5c9bda6.png)

  * 416x416 크기의 이미지를 네트워크에 입력하여 특징 맵의 크기가 52x52, 26x26, 13x13이 되는 층의 특징 맵을 추출하여 이용
  * 각 scale의 특징 맵의 출력 수가 셀의 bbox 갯수 x (bbox offset + objetcness score + 클래스 갯수) = 3x(4+1+80) = 255 되도록 1x1 Conv를 통해 채널 수 조정 
  * COCO dataset 기준 k-means clustering를 이용하여 bounding box prior를 설정, 9 cluster, 3 scale
  * (10x13), (16x30), (33x23), (30x61), (62x45), (59x119), (116x90), (156x198), (373x326)
  
Feature Extractor
-----------------
![image](https://user-images.githubusercontent.com/61686244/134756910-24a2ec25-aeb2-409d-88bb-a3ae4f74f60e.png)

![image](https://user-images.githubusercontent.com/61686244/134756911-360b5f9f-b1f9-4217-9ed1-3852affe30c0.png)

  * YOLOv2에서 backbone으로 사용되었던 Darknet19 대신 YOLOv3에서는 residual connection을 사용하는 Darknet53을 backbone network로 사용
  * RestNet 101보다 1.5배 빠르며, ResNet 152와 비슷한 성능을 보이지만 2배 이상 빠름 

YOLO 시리즈별 bounding box 갯수 비교
------------------------------------

|-|bbox|Element|
|-|----|-------|
|YOLOv1|98|7x7, 2 bbox, 448x448|
|YOLOv2|845|13x13, 5 anchor box, 416x416|
|YOLOv3|10647|(52x52x3)+(26x26x3)+(13x13x3), 416x416|


YOLOv3 학습 과정
----------------

 1) 416x416 이미지를 DarkNet 53에 입력하여 52x52, 26x26, 13x13 특징 맵 추출
 2) Fully convolutional Network를 통해 특징 피라미드 설계, 52x52x255, 26x26x255, 13x13x255
 3) 손실함수를 통한 학습 
  - bbox offset : MSE 
  - 객체를 예측하도록 할당된 bbox의 objectness score의 Binary Cross Entropy
  - 객체를 예측하도록 할당되지 않은 bbox의 no objectness score의 Binary Cross Entropy
  - Bounding Box의 Multi class Binary Cross Entropy
 4) 추론시 non maximum suppresion 이용 
  - 비 최대 억제는 object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법 (1) 동일한 클래스에대해 내림차순으로 confidence를 정렬 (2) 가장 confidence가 높은 바운딩 박스와 IoU가 일정 임계값 이상인 바운딩 박스는 동일한 물체를 검출 했다고 판단하여 지움


YOLOv3 구현
-----------
 * Ubuntu 18.04 LTS, CUDA 11.0, CUDNN 8.0.4
 * 구현중 업데이트 예정
