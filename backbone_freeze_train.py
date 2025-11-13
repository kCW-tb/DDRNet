import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from DDRNet import DDRNet
from functions import *
from pathlib import Path

def train_and_validate(args):
    # --- 단일 GPU 환경 설정 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Initialized single GPU training on device: {device}")

    # --- 데이터셋 및 데이터로더 설정 ---
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range)
    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range)

    # Parser로 전달받은 인자를 DataLoader에 적용
    print(f"DataLoader settings: num_workers={args.num_workers}, pin_memory={args.pin_memory}, shuffle={args.shuffle}, drop_last={args.drop_last}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    # --- 모델 설정 ---
    model = DDRNet(num_classes=args.num_classes).to(device)
    
    criterion = CrossEntropy(ignore_label=255)
    
    # --- ❄️ 백본 동결(Freeze) 로직 ---
    if args.freeze_backbone:
        print("❄️ Freezing backbone layers...")
        backbone_layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'spp'] 
        for name, param in model.named_parameters():
            if any(name.startswith(layer_name) for layer_name in backbone_layer_names):
                param.requires_grad = False

    # --- 옵티마이저 및 스케줄러 설정 ---
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    print(f"Total parameters: {len(list(model.parameters()))}, Trainable parameters: {len(params_to_update)}")
    
    # Parser로 전달받은 인자를 Optimizer와 Scheduler에 적용
    optimizer = torch.optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs, warmup_ratio=5e-4)

    # --- 체크포인트 로드 (학습 재개) ---
    start_epoch = 0
    min_val_loss = float('inf')
    if args.loadpath is not None:
        print(f"Loading checkpoint from: {args.loadpath}")
        checkpoint = torch.load(args.loadpath, map_location=device)
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            min_val_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming training from epoch {start_epoch}, with min_val_loss: {min_val_loss:.4f}")
        except KeyError:
            print("Old checkpoint format. Loading model state_dict only.")
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k.startswith('module.'): name = k[7:]
                elif k.startswith('model.'): name = k[6:]
                else: name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)

    # --- 로깅 설정 ---
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    mode = 'a' if start_epoch > 0 else 'w'
    with open(log_path, mode) as f:
        if start_epoch == 0: f.write("Epoch\t\tTrain-loss\t\tVal-loss\t\tlearningRate\n")

    # --- 학습 루프 시작 ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        loop = tqdm(train_dataloader, desc=f"Train [{epoch+1}/{args.epochs}]", ncols=100)
        
        for i, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), avg_loss=total_train_loss/(i+1), lr=scheduler.get_last_lr()[0])
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        scheduler.step()

        avg_val_loss_str = "N/A"
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                loop_val = tqdm(val_dataloader, desc=f"Val [{epoch+1}/{args.epochs}]", ncols=100)
                for i, (imgs, labels) in enumerate(loop_val):
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_loss_str = f"{avg_val_loss:.4f}"
            
            print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                ckp_path = os.path.join(args.result_dir, "model_best.pth")
                state_to_save = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'loss': min_val_loss,
                }
                torch.save(state_to_save, ckp_path)
                print(f"Best model saved at epoch {epoch+1} with val loss {min_val_loss:.4f}")
            
            ckp_path = os.path.join(args.result_dir, f"model_epoch{epoch+1}.pth")
            state_to_save = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
            }
            torch.save(state_to_save, ckp_path)

        lr = scheduler.get_last_lr()[0]
        with open(log_path, "a") as f:
            log_entry = f"\n{epoch + 1}\t\t{avg_train_loss:.4f}\t\t{avg_val_loss_str}\t\t{lr:.8f}"
            f.write(log_entry)

# --- 스크립트 실행 시작점 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDRNet Training Script")
    
    # --- 기본 인자 ---
    parser.add_argument("--dataset_dir", type=str, default="./data", help="Path to dataset root")
    parser.add_argument("--loadpath", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--result_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=400, help="Total number of training epochs")
    parser.add_argument("--num_classes", type=int, default=19, help="Number of segmentation classes")

    # --- 학습 하이퍼파라미터 인자 ---
    parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
    # 한 번에 학습할 이미지의 수(배치 크기)를 지정합니다. GPU 메모리에 맞춰 조절해야 합니다.
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    # SGD 옵티마이저의 모멘텀 값을 설정합니다.
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    # 가중치 감쇠(Weight Decay) 값을 설정합니다. L2 정규화와 유사하며, 모델의 과적합을 방지합니다.
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer")
    # 학습 초기에 학습률을 서서히 증가시키는 웜업(warmup) 에폭 수를 지정합니다.
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for scheduler")
    
    # --- 데이터셋 및 Augmentation 인자 ---
    parser.add_argument("--crop_size", default=[512, 1024], type=arg_as_list, help="Crop size (H W)")
    # 학습 시 이미지를 무작위로 확대/축소할 비율의 범위를 지정합니다.
    parser.add_argument("--scale_range", default=[0.75, 1.5], type=arg_as_list, help="Resize input scale range")
    
    # --- DataLoader 성능 관련 인자 ---
    # 데이터를 불러올 때 사용할 CPU 프로세스의 개수를 지정합니다. CPU 코어 수에 맞게 설정하면 좋습니다.
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for DataLoader")
    # 이 플래그를 사용하면 pin_memory 기능을 끕니다. (기본값: 켜짐) pin_memory는 CPU->GPU 데이터 전송 속도를 높여줍니다.
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory", help="Disable pin_memory for DataLoader")
    # 이 플래그를 사용하면 학습 데이터셋을 섞는(shuffle) 기능을 끕니다. (기본값: 켜짐)
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle", help="Disable shuffling for training data")
    # 이 플래그를 사용하면 마지막 배치의 크기가 batch_size보다 작을 때 해당 배치를 버리는 기능을 끕니다. (기본값: 켜짐)
    parser.add_argument("--no_drop_last", action="store_false", dest="drop_last", help="Disable drop_last for training data")
    parser.set_defaults(pin_memory=True, shuffle=True, drop_last=True)
    
    # --- 미세 조정(Fine-tuning) 관련 인자 ---
    # 이 플래그를 사용하면 모델의 백본 부분을 동결(학습되지 않도록 고정)하고, Head 부분만 학습합니다.
    parser.add_argument("--freeze_backbone", action='store_true', help="Freeze backbone layers for fine-tuning")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    train_and_validate(args)