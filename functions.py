
#클래스별 가중치 폴더별 가중치를 추가한 function.py 파일.
#클래스별 가중치 폴더별 가중치를 추가한 function.py 파일 (날씨 증강 추가)

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import ast
from glob import glob
import os
import argparse
from PIL import ImageFile, Image
from collections import OrderedDict

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def display_dataset_info(datadir, dataset):
    print(f'Dataset path: {datadir}')
    if dataset is not None:
        print(f"Found {len(dataset)} images.")

def load_state_dict(model, state_dict):
    new_state_dict = OrderedDict()
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    for k, v in state_dict.items():
        if is_ddp:
            if not k.startswith('module.'):
                k = 'module.' + k
        else:
            if k.startswith('module.'):
                k = k[len('module.'):]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys()) & model_keys
    total = len(model_keys)
    loaded = len(loaded_keys)
    percent = 100.0 * loaded / total if total > 0 else 0.0
    print(f"[Info] Loaded {loaded}/{total} state_dict entries ({percent:.2f}%) from checkpoint.")

class SegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest = transforms.InterpolationMode.NEAREST
        
        # --- 증강 정의 ---
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        self.augmix = transforms.AugMix(severity=1, mixture_width=3)

    def __call__(self, image, label):
        # --- 기하학적 변환 (이미지 & 라벨) ---
        # 1. 랜덤 스케일링
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        image = TF.resize(image, (new_height, new_width), interpolation=self.bilinear)
        label = TF.resize(label, (new_height, new_width), interpolation=self.nearest)

        # 2. 패딩
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)
        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h)
            image = TF.pad(image, padding, fill=0)
            label = TF.pad(label, padding, fill=255)

        # 3. 랜덤 크롭
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # 4. 랜덤 좌우 반전
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        
        # 5. 랜덤 회전
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=self.bilinear, fill=0)
            label = TF.rotate(label, angle, interpolation=self.nearest, fill=255)
            
        # 6. Elastic Transform (유사 효과)
        if random.random() < 0.3:
            shear_angle = random.uniform(-8, 8)
            image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0, shear=shear_angle, interpolation=self.bilinear, fill=0)
            label = TF.affine(label, angle=0, translate=(0, 0), scale=1.0, shear=shear_angle, interpolation=self.nearest, fill=255)

        # --- 색상/픽셀 변환 (이미지에만 적용) ---
        # 7. 색상 변환
        if random.random() < 0.5:
            image = self.color_jitter(image)
        # 8. 가우시안 블러
        if random.random() < 0.3:
            image = self.gaussian_blur(image)
        # 9. Random Equalize
        if random.random() < 0.2:
            image = TF.equalize(image)
        # 10. AugMix
        if random.random() < 0.3:
            image = self.augmix(image)

        # --- 11. 텐서 변환 및 정규화 ---
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        return image, label

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, crop_size, subset, scale_range, sub_folders=None):
        self.image_paths = []
        # --- [수정] --- 어떤 샘플이 어떤 폴더에서 왔는지 추적하기 위한 리스트
        self.samples = []
        
        if sub_folders is None:
            subset_path = os.path.join(root_dir, "image", subset)
            sub_folders = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        
        for folder_idx, folder_name in enumerate(sub_folders):
            folder_path = os.path.join(root_dir, "image", subset, folder_name)
            paths = sorted(glob(os.path.join(folder_path, "*.*")))
            
            for p in paths:
                self.image_paths.append(p)
                self.samples.append((p, folder_idx))

        self.label_paths = [self._get_label_path(p, root_dir) for p in self.image_paths]
        self.transform = SegmentationTransform(crop_size, scale_range)

    def _get_label_path(self, image_path, root_dir):
        image_dir = os.path.join(root_dir, "image")
        label_dir = os.path.join(root_dir, "labelmap")
        rel_path = os.path.relpath(image_path, image_dir)
        rel_path_parts = rel_path.split(os.sep)
        file_name = rel_path_parts[-1]
        base_name, ext = os.path.splitext(file_name)
        if file_name.endswith("_leftImg8bit.png"):
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            new_file_name = base_name + "_CategoryId.png"
        rel_path_parts[-1] = new_file_name
        label_path = os.path.join(label_dir, *rel_path_parts)
        return label_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        
        img, label = self.transform(img, label)
        
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.array(label, dtype=np.uint8))
            
        return img, label.long()

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight= None, aux_weights = [1, 0.4]):
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds, labels):
        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)
    
class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight = None, thresh = 0.6, aux_weights= [1, 0.4]):
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds, labels):
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)
    
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_label=255, aux_weights=[1, 0.4]):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights

    def _forward(self, preds, labels):
        num_classes = preds.size(1)
        
        # --- ✍️ [추가] 라벨 리사이징 ---
        # preds 텐서의 높이(H)와 너비(W)를 가져옴
        pred_height, pred_width = preds.shape[2], preds.shape[3]
        # 라벨의 현재 높이와 너비
        label_height, label_width = labels.shape[1], labels.shape[2]

        # 만약 예측값과 라벨의 크기가 다르면, 라벨을 예측값 크기로 리사이징
        if pred_height != label_height or pred_width != label_width:
            # 라벨은 클래스 인덱스이므로 'nearest' 보간법 사용
            labels = F.interpolate(labels.float().unsqueeze(1), 
                                   size=(pred_height, pred_width), 
                                   mode='nearest').squeeze(1).long()
        # ---------------------------------

        log_softmax = F.log_softmax(preds, dim=1)

        mask = (labels != self.ignore_label)
        labels_valid = labels[mask]
        
        if labels_valid.numel() == 0:
            return torch.tensor(0.0).to(preds.device)

        log_softmax_flat = log_softmax.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        # ✍️ [수정] 마스크 생성 위치 변경 및 적용
        mask_flat = (labels_flat != self.ignore_label)
        log_softmax_valid = log_softmax_flat[mask_flat] # 마스크 모양이 일치해야 함
        
        # 유효한 라벨의 모양도 (N*H*W)에서 유효한 픽셀만으로 변경
        labels_valid_flat = labels_flat[mask_flat]

        log_pt = log_softmax_valid.gather(1, labels_valid_flat.unsqueeze(1)).squeeze(1)

        pt = torch.exp(log_pt)
        modulating_factor = (1 - pt) ** self.gamma

        focal_loss = modulating_factor * -log_pt

        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, labels_valid_flat)
            focal_loss = alpha_weight * focal_loss
            
        return focal_loss.mean()

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=10, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]

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
            factor = (1 - self.last_epoch / float(self.total_epochs)) ** self.power
            return [factor * lr for lr in self.base_lrs]

class EpochWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear',total_epochs=500, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [max(ratio * lr, 1e-7) for lr in self.base_lrs]

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_ratio()
        return self.get_main_ratio()

    def get_warmup_ratio(self):
        alpha = self.last_epoch / self.warmup_epochs
        if self.warmup == 'linear':
            return self.warmup_ratio + (1. - self.warmup_ratio) * alpha
        else:
            return self.warmup_ratio ** (1. - alpha)

    def get_main_ratio(self):
        raise NotImplementedError

class WarmupPolyEpochLR(EpochWarmupLR):
    def __init__(self, optimizer, power=0.9, total_epochs=500, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear', last_epoch=-1):
        self.power = power
        super().__init__(optimizer, warmup_epochs, warmup_ratio, warmup, total_epochs, last_epoch)

    def get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epochs
        real_total = self.total_epochs - self.warmup_epochs
        alpha = min(real_epoch / real_total, 1.0)
        return (1 - alpha) ** self.power