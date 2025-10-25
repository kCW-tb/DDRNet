import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# 표준 BatchNorm2d를 사용하고, momentum 값을 미리 정의
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 크기의 컨볼루션 레이어를 생성하는 헬퍼 함수 (패딩 포함)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

# ResNet의 기본 블록 구조
class BasicBlock(nn.Module):
    expansion = 1  # 출력 채널과 입력 채널의 비율

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample  # 다운샘플링 레이어 (해상도 및 채널 맞추기 위함)
        self.stride = stride
        self.no_relu = no_relu  # 마지막에 ReLU를 적용할지 여부

    def forward(self, x):
        residual = x  # skip connection을 위한 원본 입력 저장

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 다운샘플링이 필요한 경우 (stride가 1이 아니거나 채널 수가 다를 때)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 잔차 연결 (residual connection)

        if self.no_relu:
            return out
        else:
            return self.relu(out)

# ResNet의 Bottleneck 블록 구조 (1x1, 3x3, 1x1 conv)
class Bottleneck(nn.Module):
    expansion = 2  # DDRNet에서는 expansion을 2로 설정

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=True)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

# Deep Aggregation Pyramid Pooling Module (DAPPM)
# 다양한 크기의 풀링을 통해 multi-scale context 정보를 집약
class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        # 다양한 scale의 average pooling을 정의
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                      BatchNorm2d(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                      )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                      BatchNorm2d(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                      )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                      BatchNorm2d(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                      )
        # Global Average Pooling
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                      BatchNorm2d(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                      )
        # 원본 해상도 처리 브랜치
        self.scale0 = nn.Sequential(
                                      BatchNorm2d(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                      )
        # 각 스케일 브랜치의 결과를 처리하는 레이어
        self.process1 = nn.Sequential(
                                      BatchNorm2d(branch_planes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                      )
        self.process2 = nn.Sequential(
                                      BatchNorm2d(branch_planes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                      )
        self.process3 = nn.Sequential(
                                      BatchNorm2d(branch_planes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                      )
        self.process4 = nn.Sequential(
                                      BatchNorm2d(branch_planes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                      )
        # 모든 브랜치의 결과를 합친 후 채널 수를 줄이는 레이어
        self.compression = nn.Sequential(
                                      BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                      )
        # 최종 출력에 더해줄 shortcut connection
        self.shortcut = nn.Sequential(
                                      BatchNorm2d(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                      )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        # 각 스케일 브랜치를 통과시키고, 이전 스케일의 결과와 합친 후 처리
        x_list.append(self.scale0(x))
        s1_interp = F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=True)
        x_list.append(self.process1(s1_interp + x_list[0]))
        s2_interp = F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=True)
        x_list.append(self.process2(s2_interp + x_list[1]))
        s3_interp = F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=True)
        x_list.append(self.process3(s3_interp + x_list[2]))
        s4_interp = F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=True)
        x_list.append(self.process4(s4_interp + x_list[3]))

        # 모든 브랜치의 결과를 채널 축으로 연결(concatenate)하고 압축
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

# 최종 분할(segmentation) 맵을 생성하는 헤드 모듈
class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=True)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor # 최종 출력을 원본 이미지 크기로 복원하기 위한 배율

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        # scale_factor가 지정된 경우, 업샘플링 수행
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear', align_corners=True)
        return out

# DDRNet 전체 모델 정의
class DDRNet(nn.Module):
    # def __init__(self, num_classes=19, layers=[2, 2, 2, 2], planes=32, spp_planes=128, head_planes=64):
    def __init__(self, num_classes=19, layers=[2, 2, 2, 2], planes=32, spp_planes=128, head_planes=64):
        super(DDRNet, self).__init__()
        highres_planes = planes * 2

        # 초기 컨볼루션 블록: 해상도를 1/4로 줄임
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)

        # --- 저해상도(low-resolution) 브랜치 ---
        self.layer1 = self._make_layer(BasicBlock, planes, planes, layers[0])
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        # --- 고해상도(high-resolution) 브랜치 ---
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, highres_planes, 2)
        self.layer4_ = self._make_layer(BasicBlock, highres_planes, highres_planes, 2)
        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        # --- 두 브랜치 간의 정보 교환(fusion)을 위한 레이어 ---
        # 저해상도 -> 고해상도 (업샘플링 + 1x1 conv)
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=True),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )
        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=True),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )
        # 고해상도 -> 저해상도 (다운샘플링)
        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=True),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )
        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=True),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=True),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        # --- 출력 헤드 ---
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        # 훈련 시에만 사용하는 보조(auxiliary) 헤드 (Deep Supervision 위함)
        if self.training:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """ResNet의 stage를 구성하는 함수"""
        downsample = None
        # stride가 1이 아니거나, 입력과 출력 채널 수가 다를 경우 downsample 레이어(1x1 conv) 생성
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample)) # 첫 블록
        inplanes = planes * block.expansion
        for i in range(1, blocks): # 나머지 블록
            if i == (blocks-1): # 마지막 블록은 ReLU를 적용하지 않을 수 있음
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 최종 출력 특징 맵의 크기 계산 (입력의 1/8)
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = [] # 저해상도 브랜치의 중간 결과 저장용

        # --- Stage 1 & 2 ---
        x = self.conv1(x)
        x = self.layer1(x)
        layers.append(x)
        x = self.layer2(self.relu(x))
        layers.append(x)

        # --- Stage 3 + 첫 번째 Fusion ---
        x_ = self.layer3_(self.relu(layers[1])) # 고해상도 브랜치
        x = self.layer3(self.relu(x))           # 저해상도 브랜치
        layers.append(x)
        
        # 고해상도 -> 저해상도
        x = x + self.down3(self.relu(x_))
        # 저해상도 -> 고해상도 (업샘플링 후 더하기)
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
        
        if self.training:
            temp = x_ # 보조 헤드의 입력으로 사용

        # --- Stage 4 + 두 번째 Fusion ---
        x_ = self.layer4_(self.relu(x_)) # 고해상도 브랜치
        x = self.layer4(self.relu(x))   # 저해상도 브랜치
        layers.append(x)

        # 고해상도 -> 저해상도
        x = x + self.down4(self.relu(x_))
        # 저해상도 -> 고해상도
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)

        # --- Stage 5 & SPP ---
        x_ = self.layer5_(self.relu(x_)) # 고해상도 브랜치
        x = self.layer5(self.relu(x))   # 저해상도 브랜치
        x = self.spp(x) # 저해상도 브랜치에 SPP 적용
        x = F.interpolate(
                        x,
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)

        # --- 최종 결합 및 출력 ---
        # 저해상도(SPP 거친) 특징과 고해상도 특징을 합쳐 최종 분할 맵 생성
        x_ = self.final_layer(x + x_)

        if self.training:
            x_extra = self.seghead_extra(temp) # 보조 출력
            return (x_, x_extra) # 훈련 시에는 주 출력과 보조 출력 모두 반환
        else:
            return x_ # 추론 시에는 주 출력만 반환