# 데이터 증식, 데이터 로드, 리스트 변환 등.

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import ast
from glob import glob
import os
import argparse

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

# argparse로 받은 문자열을 파이썬 리스트로 변환하는 함수
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

# 데이터셋 정보(경로, 이미지 수)를 출력하는 함수
def display_dataset_info(datadir, dataset):      
    print(f'Dataset path: {datadir}')    
    if dataset is not None:
        print(f"Found {len(dataset)} images.")    

# 모델 가중치를 로드할 때 'module.' 접두사 문제를 해결하는 함수
def load_state_dict(model, state_dict):
    """
    model.module vs model 키 불일치 문제를 자동으로 해결
    DDP로 학습한 모델과 일반 모델 간의 state_dict 호환성을 맞춰줌
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    # 현재 모델이 DDP로 감싸져 있는지 확인
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    for k, v in state_dict.items():
        if is_ddp: # 현재 모델이 DDP 모델이면, 키에 'module.'이 있어야 함
            if not k.startswith('module.'):
                k = 'module.' + k
        else: # 현재 모델이 일반 모델이면, 키에 'module.'이 없어야 함
            if k.startswith('module.'):
                k = k[len('module.'):]
        new_state_dict[k] = v

    # 조정된 state_dict를 모델에 로드. strict=False로 일부 키가 없어도 에러 방지.
    model.load_state_dict(new_state_dict, strict=False)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys()) & model_keys

    total = len(model_keys)
    loaded = len(loaded_keys)
    percent = 100.0 * loaded / total if total > 0 else 0.0

    print(f"[Info] Loaded {loaded}/{total} state_dict entries ({percent:.2f}%) from checkpoint.")

# 시맨틱 세그멘테이션을 위한 데이터 증강 클래스
class SegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        # ImageNet 정규화를 위한 평균과 표준편차
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 이미지 보간법: Bilinear (부드럽게)
        self.bilinear = transforms.InterpolationMode.BILINEAR
        # 레이블 보간법: Nearest (인덱스 값 유지)
        self.nearest = transforms.InterpolationMode.NEAREST

    def __call__(self, image, label):
        # --- 1. 랜덤 스케일링 ---
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)

        image = TF.resize(image, (new_height, new_width), interpolation=self.bilinear)
        label = TF.resize(label, (new_height, new_width), interpolation=self.nearest)

        # --- 2. 패딩 (이미지가 crop_size보다 작을 경우) ---
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)

        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0)
            label = TF.pad(label, padding, fill=255)  # 255는 무시할 클래스(void class)

        # --- 3. 랜덤 크롭 ---
        # 이미지와 레이블에 동일한 크롭 영역을 적용하기 위해 파라미터를 먼저 구함
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # --- 4. 랜덤 좌우 반전 ---
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # --- 5. 텐서 변환 및 정규화 ---
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        # 레이블은 NumPy 배열을 거쳐 Long 타입 텐서로 변환
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        return image, label
    
# 시맨틱 세그멘테이션 데이터셋 클래스
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, crop_size, subset, scale_range):
        self.crop_size = crop_size
        # 지정된 경로에서 이미지 파일 목록을 가져옴
        self.image_paths = sorted(glob(os.path.join(root_dir, "image", subset,"*", "*.*"), recursive=True))
        # 각 이미지 경로에 대응하는 레이블 경로를 생성
        self.label_paths = [self._get_label_path(p, root_dir) for p in self.image_paths]
        self.label_map = np.arange(256) # 현재 코드에서는 사용되지 않음
        # 위에서 정의한 데이터 증강 파이프라인을 transform으로 설정
        self.transform = SegmentationTransform(crop_size, scale_range)

    def _get_label_path(self, image_path, root_dir):
        # 이미지 경로를 기반으로 레이블 경로를 추론하는 함수
        # 예: .../image/train/a/b.png -> .../labelmap/train/a/b_CategoryId.png
        image_dir = os.path.join(root_dir, "image")
        label_dir = os.path.join(root_dir, "labelmap")

        rel_path = os.path.relpath(image_path, image_dir)
        rel_path_parts = rel_path.split(os.sep)
        file_name = rel_path_parts[-1]
        base_name, ext = os.path.splitext(file_name)

        # Cityscapes 데이터셋의 파일명 규칙에 따라 레이블 파일명 변경
        if file_name.endswith("_leftImg8bit.png"):
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else: # 다른 데이터셋을 위한 일반적인 규칙
            new_file_name = base_name + "_CategoryId.png"

        rel_path_parts[-1] = new_file_name
        label_path = os.path.join(label_dir, *rel_path_parts)
        return label_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # PIL을 사용해 이미지(RGB)와 레이블(L, 흑백)을 불러옴
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")
        # 데이터 증강 적용
        img, label = self.transform(img, label)
        
        # 만약을 위한 타입 체크 및 변환
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.array(label, dtype=np.uint8))
        
        return img, label.long()

# 보조 출력을 지원하는 CrossEntropy 손실 함수
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight= None, aux_weights = [1, 0.4]):
        super().__init__()
        self.aux_weights = aux_weights # 주 출력과 보조 출력의 손실 가중치
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds, labels):
        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        # 모델 출력이 튜플 형태(여러 개)이면, 가중 합을 계산
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        # 출력이 하나이면, 일반적인 손실 계산
        return self._forward(preds, labels)
    
# OHEM(Online Hard Example Mining)을 적용한 CrossEntropy 손실 함수
class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight = None, thresh = 0.6, aux_weights= [1, 0.4]):
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        # 손실 임계값. 이 값보다 큰 손실을 가진 픽셀을 'hard example'로 간주
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        # reduction='none'으로 설정하여 픽셀별 손실을 모두 얻음
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds, labels):
        # 최소한으로 사용할 픽셀 수 (학습 안정성)
        n_min = labels[labels != self.ignore_label].numel() // 16
        # 각 픽셀에 대한 손실 계산
        loss = self.criterion(preds, labels).view(-1)
        # 임계값보다 큰 손실을 가진 픽셀(hard examples)만 선택
        loss_hard = loss[loss > self.thresh]

        # hard example 수가 너무 적을 경우, 손실이 가장 큰 n_min개의 픽셀을 사용
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        # hard example들의 손실 평균을 최종 손실로 반환
        return torch.mean(loss_hard)

    def forward(self, preds, labels):
        # 보조 출력을 지원
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

# --- 학습률 스케줄러들 ---
from torch.optim.lr_scheduler import _LRScheduler

# Warmup + Cosine Annealing 스케줄러
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=10, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 선형 Warmup: 학습률을 0에서 base_lr까지 점진적으로 증가
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine Annealing: Cosine 곡선을 따라 학습률을 eta_min까지 감소
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]

# Polynomial Decay 스케줄러
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs=500, decay_epoch=1, power=0.9, last_epoch=-1) -> None:
        self.decay_epoch = decay_epoch
        self.total_epochs = total_epochs
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_epoch != 0 or self.last_epoch > self.total_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            # 다항식 감쇠 공식
            factor = (1 - self.last_epoch / float(self.total_epochs)) ** self.power
            return [factor * lr for lr in self.base_lrs]

# Warmup을 위한 기본 스케줄러 클래스
class EpochWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear',total_epochs=500, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio # warmup 시작 시의 학습률 비율
        self.warmup = warmup
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [max(ratio * lr, 1e-7) for lr in self.base_lrs]

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_ratio()
        return self.get_main_ratio() # Warmup 이후의 주된 스케줄링 로직

    def get_warmup_ratio(self):
        # 선형 또는 지수적 warmup 비율 계산
        alpha = self.last_epoch / self.warmup_epochs
        if self.warmup == 'linear':
            return self.warmup_ratio + (1. - self.warmup_ratio) * alpha
        else:
            return self.warmup_ratio ** (1. - alpha)

    def get_main_ratio(self):
        raise NotImplementedError # 상속받는 클래스에서 구현 필요

# EpochWarmupLR를 상속받아 Poly Decay를 결합한 스케줄러
class WarmupPolyEpochLR(EpochWarmupLR):
    def __init__(self, optimizer, power=0.9, total_epochs=500, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear', last_epoch=-1):
        self.power = power
        super().__init__(optimizer, warmup_epochs, warmup_ratio, warmup, total_epochs, last_epoch)

    def get_main_ratio(self):
        # Warmup이 끝난 후, 남은 에폭에 대해 Poly Decay 적용
        real_epoch = self.last_epoch - self.warmup_epochs
        real_total = self.total_epochs - self.warmup_epochs
        alpha = min(real_epoch / real_total, 1.0)
        return (1 - alpha) ** self.power