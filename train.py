import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import model.yolov3
import utils.datasets
import utils.utils

from test import evaluate

# 인자값 받아오기
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=100, help="the number of epoch")
parser.add_argument("--gradient-accumulation", type=int, default=1, help="the number of gradient accums before step")
parser.add_argument("--multiscale_training", type=bool, default=True, help="allow for multi-scale training")
parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
parser.add_argument("--num_workers", type=int, default=8, help="the number of cpu threads to use during batch generation")
parser.add_argument("--data_config", type=str, default="config/voc.data",help="path to data config file")
parser.add_argument("--pretrained_weights", type=str, default='weights/darknet53.conv.74', help="if specified starts from checkpoing model")
parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
args = parser.parse_args()
print(args)

# gpu device 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 텐서보드 객체 생성
now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
log_dir = os.path.join('logs',now) #인수에 전달된 2개의 문자열을 결합하여 1개의 경로로 변경
os.makedirs(log_dir,exist_ok=True)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

# 데이터셋 설정값 가져오기
data_config = utils.utils.parse_data_config(args.data_config)
train_path = data_config['train']
valid_path = data_config['valid']
num_classes = int(data_config['classes'])
class_names = utils.utils.load_claases(data_config['names'])

# 모델 준비하기 & 정규 분포 형태로 변경
model = model.yolov3.YOLOv3(args.image_size, num_classes).to(device)
model.apply(utils.utils.init_weights_normal)
if args.pretrained_weights.endswith('.pth'):
    model.load_state_dict(torch.load(args.pretrained_weights))
else :
    model.load_darknet_weights(args.pretrained_weights)

# 데이터 셋, 데이터 로더 설정
dataset = utils.datasets.ListDataset(train_path, args.image_size, augment=True, multiscale=args.multiscale_training)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         collate_fn=dataset.collate_fn)

# optimizer 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# learning rate scheduler 설정
# 처음에 큰 lr로 빠르게 optimize를 하고 최적값에 가까워질수록 lr를 줄여 미세조정을 하는 것이 학습이 잘된다고 알려져있음
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.8)

# 현재 배치 손실값을 출력하는 tqdm 설정
loss_log = tqdm.tqdm(total=0,position=2, bar_format='{desc}', leave=False)

#Train

for epoch in tqdm.tqdm(range(args.epoch), desc='Epoch'):
    #train 모드 진입
    model.train()

    # 1 epoch의 각 배치에서 처리되는 코드
    for batch_idx, (_, images, targets) in enumerate(tqdm.tqdm(dataloader, desc='Batch', leave=False)): #enumerate를 통해 몇 번째 배치인지 확인
        step = len(dataloader)*epoch+batch_idx
        # 이미지와 정답 값을 GPU로 복사
        images = images.to(device)
        targets = targets.to(device)

        # 순전파(forward), 역전파(backward)
        loss, outputs = model(images, targets)
        loss.backward()
##############################################################################
        # 역전파 단계 전에, optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인) 갱신할
        # 변수들에 대한 모든 변화도(gradient)를 0으로 만듭니다. 이렇게 하는 이유는 기본적으로
        # .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고) 누적되기
        # 때문입니다. 더 자세한 내용은 torch.autograd.backward에 대한 문서를 참조하세요.
        #optimizer.zero_grad()

        # 역전파 단계: 모델의 매개변수들에 대한 손실의 변화도를 계산합니다.
        #loss.backward()

        # optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
        #optimizer.step()
##############################################################################
        # 기울기 누적 (Accumulate gradient)
        if step % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 총 손실값 출력
        loss_log.set_description_str('Loss : {:.6f}'.format(loss.item()))

        # Tensorboard에 훈련 과정 기록
        tensorboard_log = []
        for i, yolo_layer in enumerate(model.yolo_layers):
            writer.add_scalar('loss_bbox_{}'.format(i+1), yolo_layer.metrics['loss_bbox'], step)
            writer.add_scalar('loss_conf_{}'.format(i+1), yolo_layer.metrics['loss_conf'],step)
            writer.add_scalar('loss_cls_{}'.format(i + 1), yolo_layer.metrics['loss_cls'], step)
            writer.add_scalar('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)
        writer.add_scalar('total_loss', loss.item(), step)

    # lr scheduler 진행
    scheduler.step()

    # 검증 데이터셋으로 모델 평가
    precision, recall, AP, f1, _,_,_ = evaluate(model,
                                              path = valid_path,
                                              iou_thres=0.5,
                                              conf_thres=0.5,
                                              nms_thres=0.5,
                                              image_size = args.image_size,
                                              batch_size =args.batch_size,
                                              num_workers =args.num_workers,
                                              device=device)

    #Tensorboard에 평가 결과 기록
    writer.add_scalar('val_precision', precision.mean(),epoch)
    writer.add_scalar('val_recall',recall.mean(), epoch)
    writer.add_scalar('val_mAP',AP.mean(), epoch)
    writer.add_scalar('val_f1',f1.mean(), epoch)

    #checkpoint file 저장
    save_dir = os.path.join('checkpoint', now)
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = os.path.split(args.data_config)[-1].split('.')[0]
    torch.ave(model.state_dict(), os.path.join(save_dir, 'yolov3_{}_{}.pth'.format(dataset_name, epoch)))










