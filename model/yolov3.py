import torch
import torch.nn as nn
import torch.utils.tensorboard
import numpy as np
import utils.utils

class YOLODetection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}

    def forward(self, x, targets):
        device = torch.device('cuda:0' if x.is_cuda else 'cpu')

        # ex) x=[1,3,416,416]일때 x.size(0)=1, x.size(2)=416
        num_batches = x.size(0)
        grid_size = x.size(2)

        # 출력값 형태 변환
        prediction = (
            x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0,1,3,4,2).contiguous()
        )

        # Get outputs
        cx = torch.sigmoid(prediction[...,0]) # Center x, sigmoid를 사용하여 범위 제한, 초기 학습 복잡성 제거
        cy = torch.sigmoid(prediction[...,0]) # Center y, sigmoid를 사용하여 범위 제한, 초기 학습 복잡성 제거
        w = prediction[...,2] # Width
        h = prediction[...,3] # Height
        pred_conf = torch.sigmoid(prediction[...,4]) # objectness
        pred_cls = torch.sigmoid(prediction[...,5]) # Class prediction

        # 각 그리드에 대한 offset 계산
        stride = self.image_size / grid_size
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size,1).view(
            [1,1,grid_size,grid_size])
        grid_y = torch.arange(grid_size,dtype=torch.float,device=device).repeat(grid_size,1).view(
            [1,1,grid_size,grid_size])
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0:1].view(1, self.num_anchors,1,1)
        anchor_h = scaled_anchors[:,1:2].view(1,self.num_anchors,1,1)

        pred_boxes = torch.zeros_like(prediction[...,:4],device=device)
        pred_boxes[...,0] = cx+grid_x
        pred_boxes[...,1] = cy+grid_y
        pred_boxes[...,2] = torch.exp(w)*anchor_w
        pred_boxes[...,3] = torch.exp(h)*anchor_h

        pred = (pred_boxes.view(num_batches, -1, 4) * stride,
                pred_conf.view(num_batches, -1, 1),
                pred_cls.view(num_batches, -1, self.num_classes))

        output = torch.cat(pred,-1)

        if targets is None:
            return output, 0






class YOLOv3(nn.Module):
    def __init__(self,image_size:int, num_classes: int):
        super(YOLOv3, self).__init__()
        anchors={'scale1':[(10,13),(16,30),(33,23)],
                 'scale2':[(30,61),(62,45),(59,119)],
                 'scale3':[(116,90),(156,198),(373,326)]}
        final_out_channel = 3*(4+1+num_classes)

        self.darknet53=self.make_dakrnet53()
        self.conv_block3 = self.make_conv_block(1024,512)
        self.conv_final3 = self.make_conv_final(512,final_out_channel)
        self.yolo_layer3 = YOLODetection(anchors['scale3'],image_size,num_classes)

        self.upsample2 = self.make_upsample(512,256,scale_factor=2)
        self.conv_block2 = self.make_conv_block(768,256)
        self.final2 = self.make_conv_final(256,final_out_channel)
        self.yolo_layer2 = YOLODetection(anchors['scale2'],image_size,num_classes)

        self.upsample1 = self.make_upsample(256,128,scale_factor=2)
        self.conv_block1 = self.make_conv_block(384,128)
        self.conv_final1 = self.make_conv_final(128,final_out_channel)
        self.yolo_layer1 = YOLODetection(anchors['scale1'],image_size,num_classes)

        self.yolo_layers = [self.yolo_layer1, self.yolo_layer2, self.yolo_layer3]

    def forward(self, x, targets=None):
        loss = 0
        residual_output = {}

        # Darknet53 forward

        with torch.no_grad(): #autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함
            for key, module in self.darknet53.items():
                module_type = key.split('_')[0]

                if module_type == 'conv':
                    x = module(x)
                elif module_type == 'residual':
                    out= module(x)
                    x += out
                    if key == 'residual_3_8' or key == 'residual_4_8' or key == 'residual_5_4':
                        residual_output[key] = x

        # YOLOv3 layer forward
        conv_block3 = self.conv_block3(residual_output['residual_5_4'])
        scale3 = self.conv_final3(conv_block3)
        yolo_output3, layer_loss = self.yolo_layer3(scale3, targets)
        loss += layer_loss

        scale2 = self.upsample2(conv_block3)
        scale2 = torch.cat((scale2,residual_output['residual_4_8']), dim=1)
        conv_block2 = self.conv_block2(scale2)
        scale2 = self.conv_final2(conv_block2)
        yolo_output2, layer_loss = self.yolo_layer2(scale2, targets)
        loss += layer_loss

        scale1 = self.upsample1(conv_block2)
        scale1 = torch.cat((scale1, residual_output['residual_3_8']),dim=1)
        conv_block1 = self.conv_block1(scale1)
        scale1 = self.conv_final1(conv_block1)
        yolo_output1, layer_loss = self.yolo_layer1(scale1, targets)
        loss += layer_loss

        yolo_outputs = [yolo_output1, yolo_output2, yolo_output3]
        yolo_outputs = torch.cat(yolo_outputs,1).detach().cpu()
        return yolo_outputs if targets is None else(loss ,yolo_outputs)

    def make_darknet53(self):
        modules = nn.ModuleDict()

        modules['conv_1'] = self.make_conv(3,32, kernel_size=3, requires_grad=False)
        modules['conv_2'] = self.make_conv(32, 64, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_1_1'] = self.make_residual_block(in_channels=64)

        modules['conv_3'] = self.make_conv(64, 128, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_2_1'] = self.make_residual_block(in_channels=128)
        modules['residual_2_2'] = self.make_residual_block(in_channels=128)

        modules['conv_4'] = self.make_conv(128, 256, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_3_1'] = self.make_residual_block(in_channels=256)
        modules['residual_3_2'] = self.make_residual_block(in_channels=256)
        modules['residual_3_3'] = self.make_residual_block(in_channels=256)
        modules['residual_3_4'] = self.make_residual_block(in_channels=256)
        modules['residual_3_5'] = self.make_residual_block(in_channels=256)
        modules['residual_3_6'] = self.make_residual_block(in_channels=256)
        modules['residual_3_7'] = self.make_residual_block(in_channels=256)
        modules['residual_3_8'] = self.make_residual_block(in_channels=256)

        modules['conv_5'] = self.make_conv(256, 512, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_4_1'] = self.make_residual_block(in_channels=512)
        modules['residual_4_2'] = self.make_residual_block(in_channels=512)
        modules['residual_4_3'] = self.make_residual_block(in_channels=512)
        modules['residual_4_4'] = self.make_residual_block(in_channels=512)
        modules['residual_4_5'] = self.make_residual_block(in_channels=512)
        modules['residual_4_6'] = self.make_residual_block(in_channels=512)
        modules['residual_4_7'] = self.make_residual_block(in_channels=512)
        modules['residual_4_8'] = self.make_residual_block(in_channels=512)

        modules['conv_6'] = self.make_conv(512, 1024, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_5_1'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_2'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_3'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_4'] = self.make_residual_block(in_channels=1024)

        return modules

    def make_conv(self, in_channels:int, out_channels: int, kernel_size: int, stride=1, padding=1, requires_grad=True):
        module1 = nn.Conv2d(in_channels,out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels,momentum=0.9, eps=1e-5)

        if not requires_grad:
            for param in module1.parameters():
                param.requires_grad_(False)
            for param in module2.parameters():
                param.requires_grad(False)

        modules = nn.Sequential(module1,module2,nn.LeakyReLU(negative_slope=0.1))
        return modules
    # ?->255
    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, in_channels*2, kernel_size=3),
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
    def make_residual_block(self, in_channels:int):
        half_channels = in_channels // 2
        block = nn.Sequential(
            self.make_conv(in_channels, half_channels, kernel_size=1, padding=0, requires_grad=False),
            self.make_conv(half_channels, in_channels, kernel_size=3, requires_grad=False)
        )
        return block

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules

if __name__ == '__main__':
    model = YOLOv3(image_size=416, num_classes=80)
    model.load_darknet_weights('../weights/yolov3.weights')
    print(model)
    test = torch.rand([1,3,416,416])
    y = model(test)
    writer = torch.utils.tensorboard.SummaryWriter('../logs')
    writer.add_graph(model, test)
    writer.close()

