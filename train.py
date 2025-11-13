# #학습 시작.

# import os
# import argparse
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from tqdm import tqdm
# from DDRNet import DDRNet
# from functions import *
# from pathlib import Path

# def train(args):
#     # --- 분산 학습 환경 설정 ---
#     # torchrun으로부터 환경 변수(RANK, LOCAL_RANK, WORLD_SIZE)를 가져옴
#     # rank = int(os.environ["RANK"])
#     # local_rank = int(os.environ["LOCAL_RANK"])
#     # world_size = int(os.environ["WORLD_SIZE"])
#     rank = 0
#     local_rank = 0
#     world_size = 1

#     # 분산 학습을 위한 프로세스 그룹 초기화 (NCCL 백엔드 사용)
#     # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

#     # 현재 프로세스가 사용할 GPU 장치 설정
#     torch.cuda.set_device(local_rank)
#     device = torch.device("cuda", local_rank)


#     # --- 데이터셋 및 데이터로더 설정 ---
#     train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range)
#     display_dataset_info(args.dataset_dir, train_dataset)
#     # DistributedSampler: 데이터셋을 여러 GPU에 중복되지 않게 분배
#     sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, drop_last=True, shuffle=True)
#     # DataLoader: 데이터셋에서 배치 단위로 데이터를 가져옴
#     dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
#                             num_workers=4, # 개선점: 0보다 큰 값으로 설정하여 데이터 로딩 병목 현상 완화 가능
#                             pin_memory=True) # GPU로의 데이터 전송 속도 향상

#     # --- 모델 설정 ---
#     print(f"[GPU {local_rank}] Before model setup")
#     # DDRNet 모델을 생성하고 현재 GPU로 이동
#     model = DDRNet(num_classes=args.num_classes).to(device)
#     # 모델을 DDP로 감싸 분산 학습 모델로 만듦
#     model = DDP(model, device_ids=[local_rank])
#     print(f"[GPU {local_rank}] DDP initialized")

#     # --- 손실 함수, 옵티마이저, 스케줄러 설정 ---
#     criterion = CrossEntropy(ignore_label=255) # ignore_label=255는 손실 계산에서 255 라벨을 무시
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#     # 학습률 스케줄러: Warmup 후 Poly 정책에 따라 학습률 감소
#     scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

#     # --- 사전 학습된 가중치 로드 ---
#     if args.loadpath is not None:
#         # 다른 GPU에서 저장된 가중치를 현재 GPU에 맞게 불러오기 위한 맵핑
#         map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
#         state_dict = torch.load(args.loadpath, map_location=map_location)
#         load_state_dict(model, state_dict)

#     # --- 로깅 설정 (주 프로세스에서만 수행) ---
#     if local_rank == 0:
#         os.makedirs(args.result_dir, exist_ok=True)
#         log_path = os.path.join(args.result_dir, "log.txt")
#         with open(log_path, 'w') as f:
#             f.write("Epoch\t\tTrain-loss\t\tlearningRate\n")

#     min_loss = 100.0
#     # --- 학습 루프 시작 ---
#     for epoch in range(args.epochs):
        
#         model.train() # 모델을 학습 모드로 설정
#         # DistributedSampler가 매 에폭마다 데이터를 다르게 셔플하도록 epoch 설정
#         sampler.set_epoch(epoch)
#         total_loss = 0.0
#         # 주 프로세스에서만 tqdm으로 진행 상황 표시
#         loop = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch [{epoch+1}/{args.epochs}]", ncols=100) if local_rank == 0 else dataloader

#         for i, (imgs, labels) in enumerate(loop):
#             # 그래디언트 초기화 (메모리 효율을 위해 set_to_none=True 사용)
#             optimizer.zero_grad(set_to_none=True)
            
#             # 데이터를 현재 GPU로 이동
#             imgs, labels = imgs.to(device), labels.to(device)
#             # 순전파 (Forward pass)
#             outputs = model(imgs)
#             # 손실 계산
#             loss = criterion(outputs, labels)
#             # 역전파 (Backward pass)
#             loss.backward()
#             # 모델 파라미터 업데이트
#             optimizer.step()
            
#             # 개선점: 아래 코드는 성능 저하를 유발할 수 있으므로 디버깅 목적이 아니면 제거하는 것이 좋음
#             torch.cuda.synchronize()
            
#             total_loss += loss.item()
#             # 주 프로세스에서만 진행 상황 업데이트
#             if local_rank == 0:
#                 loop.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1), lr=scheduler.get_last_lr()[0])

#         # 개선점: 아래 코드는 성능 저하를 유발하며, 근본적인 메모리 문제 해결책이 아님
#         torch.cuda.empty_cache()
#         # 모든 프로세스가 에폭의 끝까지 도달하도록 동기화
#         dist.barrier()
#         # 학습률 스케줄러 업데이트
#         scheduler.step()

#         # --- 체크포인트 저장 (주 프로세스에서만 수행) ---
#         # 현재까지의 최소 손실보다 작으면 최고의 모델로 저장
#         if local_rank == 0 and ((total_loss / len(dataloader)) < min_loss):
#             ckp_path = os.path.join(args.result_dir, f"model_best.pth")
#             # DDP 모델의 경우 model.module.state_dict()를 저장하는 것이 일반적이나,
#             # DDP로 감싼 상태 그대로 저장해도 로드 시 DDP 모델에 로드하면 문제 없음
#             torch.save(model.state_dict(), ckp_path)
        
#         # 5 에폭마다 현재 모델 저장 및 로그 기록
#         if local_rank == 0 and (epoch + 1) % 5 == 0:
#             ckp_path = os.path.join(args.result_dir, f"model_epoch{epoch+1}.pth")
#             torch.save(model.state_dict(), ckp_path)
            
#             lr = scheduler.get_last_lr()
#             lr = sum(lr) / len(lr) # 평균 학습률 계산
#             with open(log_path, "a") as f:
#                 f.write("\n%d\t\t%.4f\t\t%.8f" % (epoch + 1, total_loss / len(dataloader), scheduler.get_last_lr()[0]))

#     # 분산 학습 프로세스 그룹 정리
#     dist.destroy_process_group()


# # ---------- 스크립트 실행 시작점 ----------
# if __name__ == "__main__":
#     # NCCL 관련 환경 변수 설정 (네트워크 문제 해결용)
#     os.environ["NCCL_DEBUG"] = "INFO"
#     os.environ["NCCL_P2P_DISABLE"] = "1" 
#     os.environ["NCCL_IB_DISABLE"] = "1"
    
#     # 커맨드 라인 인자 파싱
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_dir", type=str, help="Path to dataset root", 
#                         default="./data" )
#     parser.add_argument("--loadpath", type=str, help="Path to dataset root", 
#                         default="./pths/DDRNet23s_imagenet.pth" ) # "ex: ./pths/DDRNet23s_imagenet.pth"
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--result_dir", type=str, default="output")
#     parser.add_argument("--lr", type=float, default=1e-2)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--num_classes", type=int, default=19)
#     parser.add_argument("--crop_size", default=[512, 1024], type=arg_as_list, help="crop size (H W)")
#     parser.add_argument("--scale_range", default=[0.75, 1.5], type=arg_as_list, help="resize Input")
    
#     args = parser.parse_args()
    
#     print(f'Initial learning rate: {args.lr}')
#     print(f'Total epochs: {args.epochs}')
#     print(f'dataset path: {args.dataset_dir}')
            
#     # 결과 저장 디렉토리 생성
#     result_dir = Path(args.result_dir)
#     result_dir.mkdir(parents=True, exist_ok=True)
#     # 멀티프로세싱 시작 방식 설정
#     torch.multiprocessing.set_start_method('spawn', force=True)
#     # 학습 함수 호출
#     train(args)


import os
import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import OrderedDict
import json
from pathlib import Path

# 이전에 정의한 DDRNet 모델과 functions.py의 모든 클래스/함수를 임포트
from DDRNet import DDRNet
from functions import *

# 시각화를 위한 matplotlib 임포트
import matplotlib.pyplot as plt
import numpy as np

# argparse용 헬퍼 함수
def arg_as_dict(s):
    try:
        return json.loads(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Argument must be a JSON-formatted dictionary string. Error: {e}")

# 증강 결과 시각화 함수
# def visualize_augmentations(dataset, num_samples=3, num_classes=19):
#     print(f"Displaying augmentations for {num_samples} sample(s)...")
#     if len(dataset) < num_samples:
#         print(f"Warning: Requested {num_samples} samples, but dataset only has {len(dataset)}. Displaying all.")
#         num_samples = len(dataset)

#     for i in range(num_samples):
#         original_img = Image.open(dataset.image_paths[i]).convert("RGB")
#         original_label = Image.open(dataset.label_paths[i]).convert("L")
#         augmented_img_tensor, augmented_label_tensor = dataset.transform(original_img.copy(), original_label.copy())

#         mean, std = np.array(dataset.transform.mean), np.array(dataset.transform.std)
#         augmented_img_np = augmented_img_tensor.numpy().transpose((1, 2, 0)) * std + mean
#         augmented_img_np = np.clip(augmented_img_np, 0, 1)
#         augmented_label_np = augmented_label_tensor.numpy()
        
#         colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
#         colors[0] = [0, 0, 0]

#         original_label_color = np.zeros((*np.array(original_label).shape, 3), dtype=np.uint8)
#         for c in range(num_classes):
#             original_label_color[np.array(original_label) == c] = colors[c]

#         augmented_label_color = np.zeros((*augmented_label_np.shape, 3), dtype=np.uint8)
#         for c in range(num_classes):
#             augmented_label_color[augmented_label_np == c] = colors[c]

#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#         fig.suptitle(f"Augmentation Example {i+1}", fontsize=16)
#         axes[0, 0].imshow(original_img); axes[0, 0].set_title("Original Image"); axes[0, 0].axis('off')
#         axes[0, 1].imshow(original_label_color); axes[0, 1].set_title("Original Label"); axes[0, 1].axis('off')
#         axes[1, 0].imshow(augmented_img_np); axes[1, 0].set_title("Augmented Image"); axes[1, 0].axis('off')
#         axes[1, 1].imshow(augmented_label_color); axes[1, 1].set_title("Augmented Label"); axes[1, 1].axis('off')
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def train(args):
    # --- 환경 설정 ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Initialized single GPU training on device: {device}")

    # --- 데이터셋 설정 ---
    train_sub_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set1', 'set2', 'set3']
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range, sub_folders=train_sub_folders)
    
    # # 증강 시각화 플래그 확인
    # if args.visualize_augmentations:
    #     visualize_augmentations(train_dataset, num_samples=args.vis_count, num_classes=args.num_classes)
    #     return # 시각화 후 학습을 시작하지 않고 종료

    # --- 데이터로더 설정 ---
    sampler = None
    shuffle = True
    if args.folder_weights:
        print("Applying folder-wise weights for sampling...")
        folder_indices = [sample[1] for sample in train_dataset.samples]
        folder_names_per_sample = [train_sub_folders[i] for i in folder_indices]
        sample_weights = [args.folder_weights.get(name, 1.0) for name in folder_names_per_sample]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False # Sampler가 셔플을 담당하므로 False로 설정

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # --- 모델 설정 ---
    model = DDRNet(num_classes=args.num_classes).to(device)
    
    # --- 손실 함수 설정 ---
    class_weights = None
    if args.class_weights:
        if len(args.class_weights) != args.num_classes: raise ValueError("Class weights count mismatch")
        print(f"Applying class weights: {args.class_weights}")
        class_weights = torch.tensor(args.class_weights, dtype=torch.float).to(device)

    if args.use_ohem:
        print("Using OhemCrossEntropy Loss.")
        criterion = OhemCrossEntropy(ignore_label=255, weight=class_weights)
    elif args.use_focal_loss:
        print(f"Using FocalLoss with gamma={args.focal_gamma}.")
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, ignore_label=255)
    else:
        print("Using standard CrossEntropy Loss.")
        criterion = CrossEntropy(ignore_label=255, weight=class_weights)
    
    # --- 옵티마이저 및 스케줄러 설정 ---
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    # --- 체크포인트 로드 ---
    start_epoch = 0
    if args.loadpath:
        print(f"Loading checkpoint from: {args.loadpath}")
        checkpoint = torch.load(args.loadpath, map_location=device)
        try:
            # 학습 재개를 위한 전체 상태 로드
            load_state_dict(model, checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except KeyError:
            # 모델 가중치만 있는 구형 체크포인트 로드
            print("Old checkpoint format. Loading model state_dict only.")
            load_state_dict(model, checkpoint)

    # --- 로깅 설정 ---
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    with open(log_path, 'a' if start_epoch > 0 else 'w') as f:
        if start_epoch == 0:
            f.write("Epoch\t\tTrain-loss\t\tlearningRate\n")

    # --- 학습 루프 ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_dataloader, desc=f"Train [{epoch+1}/{args.epochs}]", ncols=100)
        
        for i, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1), lr=scheduler.get_last_lr()[0])
        
        scheduler.step()

        # --- 체크포인트 저장 및 로그 기록 ---
        # 5 에폭마다, 그리고 마지막 에폭에 저장
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            periodic_path = os.path.join(args.result_dir, f"model_epoch_{epoch+1}.pth")
            state_to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state_to_save, periodic_path)
            print(f"Periodic checkpoint saved for epoch {epoch+1} to {periodic_path}")
            
        # 로그 기록
        avg_train_loss = total_loss / len(train_dataloader)
        lr = scheduler.get_last_lr()[0]
        with open(log_path, "a") as f:
            f.write(f"\n{epoch + 1}\t\t{avg_train_loss:.4f}\t\t{lr:.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDRNet Single GPU Training Script")
    
    # 기본 인자
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--result_dir", type=str, default="output")
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    
    # 데이터셋 및 Augmentation
    parser.add_argument("--crop_size", default=[512, 1024], type=arg_as_list)
    parser.add_argument("--scale_range", default=[0.75, 1.5], type=arg_as_list)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 가중치 조절
    parser.add_argument("--folder_weights", type=arg_as_dict, default={"cam0":1.8, "cam1":0.8, "cam2":0.8, "cam3":0.8, "cam4":0.8, "cam5":0.8, "set1":1.5, "set2":2.0, "set3":1.5})
    parser.add_argument("--class_weights", type=arg_as_list, default=[2.0166, 3.481, 4.0911, 3.9912, 3.9619, 2.0864, 1.8396, 4.3168, 3.79, 8.4674, 5.7661, 5.642, 10.0, 5.9525, 2.2137, 5.2137, 8.1661, 4.195, 1.0])
    
    # 손실 함수 선택
    parser.add_argument("--use_ohem", action='store_true')
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    # 증강 시각화
    # parser.add_argument("--visualize_augmentations", action='store_true')
    # parser.add_argument("--vis_count", type=int, default=3)
    
    args = parser.parse_args()
    
    train(args)