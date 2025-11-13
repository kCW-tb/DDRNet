import os
import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import OrderedDict
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# 모델 및 함수 임포트
from DDRNet import DDRNet
# from DDRNet_39 import get_seg_model
from functions import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------
# ⚙️ Argument Helper Functions
# -----------------------------
def arg_as_dict(s):
    try:
        return json.loads(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Argument must be a JSON-formatted dictionary string. Error: {e}")

def arg_as_list(s):
    try:
        return json.loads(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Argument must be a JSON list, e.g. '[1,2,3]'. Error: {e}")

# -----------------------------
# 🔍 데이터 증강 시각화 함수
# -----------------------------
def visualize_augmentations(dataset, num_samples=3, num_classes=19):
    print(f"Displaying augmentations for {num_samples} sample(s)...")
    if len(dataset) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(dataset)}. Displaying all.")
        num_samples = len(dataset)

    for i in range(num_samples):
        original_img = Image.open(dataset.image_paths[i]).convert("RGB")
        original_label = Image.open(dataset.label_paths[i]).convert("L")

        augmented_img_tensor, augmented_label_tensor = dataset.transform(original_img.copy(), original_label.copy())

        mean = np.array(dataset.transform.mean)
        std = np.array(dataset.transform.std)
        augmented_img_np = augmented_img_tensor.numpy().transpose((1, 2, 0))
        augmented_img_np = std * augmented_img_np + mean
        augmented_img_np = np.clip(augmented_img_np, 0, 1)
        augmented_label_np = augmented_label_tensor.numpy()

        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]

        original_label_color = np.zeros((*np.array(original_label).shape, 3), dtype=np.uint8)
        for c in range(num_classes):
            original_label_color[np.array(original_label) == c] = colors[c]

        augmented_label_color = np.zeros((*augmented_label_np.shape, 3), dtype=np.uint8)
        for c in range(num_classes):
            augmented_label_color[augmented_label_np == c] = colors[c]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Augmentation Example {i+1}", fontsize=16)
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(original_label_color)
        axes[0, 1].set_title("Original Label")
        axes[0, 1].axis('off')
        axes[1, 0].imshow(augmented_img_np)
        axes[1, 0].set_title("Augmented Image")
        axes[1, 0].axis('off')
        axes[1, 1].imshow(augmented_label_color)
        axes[1, 1].set_title("Augmented Label")
        axes[1, 1].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# -----------------------------
# 🚀 Training Function
# -----------------------------
def train_and_validate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Initialized training on device: {device}")

    train_sub_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set1', 'set2', 'set3']
    val_sub_folders = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'set1', 'set2', 'set3']
    
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range, sub_folders=train_sub_folders)
    if args.visualize_augmentations:
        visualize_augmentations(train_dataset, num_samples=args.vis_count, num_classes=args.num_classes)
        return  # 시각화 후 종료

    if args.folder_weights:
        print("Applying folder-wise weights for sampling...")
        folder_indices = [sample[1] for sample in train_dataset.samples]
        folder_names_per_sample = [train_sub_folders[i] for i in folder_indices]
        sample_weights = [args.folder_weights.get(name, 1.0) for name in folder_names_per_sample]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range, sub_folders=val_sub_folders)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DDRNet(num_classes=args.num_classes).to(device)
    # model = get_seg_model(num_classes=args.num_classes).to(device)

    # -----------------------------
    # 🧊 Backbone Freeze
    # -----------------------------
    if args.freeze_backbone:
        print("Freezing backbone parameters...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in ["backbone", "stem", "encoder", "resnet"]):
                param.requires_grad = False
                frozen_count += 1
        print(f"✅ Frozen {frozen_count} parameters from backbone.")

    # -----------------------------
    # ⚖️ Loss Function Setup
    # -----------------------------
    class_weights = None
    if args.class_weights:
        if len(args.class_weights) != args.num_classes:
            raise ValueError(f"Number of class_weights ({len(args.class_weights)}) must match num_classes ({args.num_classes})")
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

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    start_epoch = 25
    min_val_loss = float('inf')
    writer = SummaryWriter(log_dir=os.path.join(args.result_dir, 'tensorboard'))

    if args.loadpath:
        print(f"Loading checkpoint from: {args.loadpath}")
        checkpoint = torch.load(args.loadpath, map_location=device)
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            print(f"[Info] Loaded {len(new_state_dict)}/{len(model.state_dict())} state_dict entries from checkpoint.")
        except KeyError:
            print("Old checkpoint format. Loading model state_dict only.")
            load_state_dict(model, checkpoint)

    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch\tTrain-loss\tVal-loss\tLR\n")

    # -----------------------------
    # 🔁 Training Loop
    # -----------------------------
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
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch + 1)

        avg_val_loss_str = "N/A"
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                loop_val = tqdm(val_dataloader, desc=f"Val [{epoch+1}/{args.epochs}]", ncols=100)
                for imgs, labels in loop_val:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_loss_str = f"{avg_val_loss:.4f}"
            print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

            writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_path = os.path.join(args.result_dir, "model_best.pth")
                torch.save({'model_state_dict': model.state_dict()}, best_path)
                print(f"✅ Best model saved at epoch {epoch+1} with val loss {min_val_loss:.4f}")

            periodic_path = os.path.join(args.result_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_val_loss}, periodic_path)
            print(f"💾 Checkpoint saved: {periodic_path}")

        lr = scheduler.get_last_lr()[0]
        with open(log_path, "a") as f:
            f.write(f"\n{epoch + 1}\t{avg_train_loss:.4f}\t{avg_val_loss_str}\t{lr:.8f}")

    writer.close()
    print("🎉 Training completed.")

# -----------------------------
# 🧩 Argument Setup
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDRNet Weighted Training Script")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--result_dir", type=str, default="output")
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    parser.add_argument("--crop_size", type=arg_as_list, default=[512, 1024])
    parser.add_argument("--scale_range", type=arg_as_list, default=[0.75, 1.5])
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())

    parser.add_argument("--folder_weights", type=arg_as_dict, help='{"cam0": 1.0, "set1": 2.0}')
    parser.add_argument("--class_weights", type=arg_as_list, default=[2.0166, 3.481, 4.0911, 3.9912, 3.9619, 2.0864, 1.8396, 4.3168, 3.79, 6.4674, 1.0, 5.642, 1.0, 5.9525, 2.2137, 5.2137, 6.1661, 4.195, 1.0])

    parser.add_argument("--use_ohem", action='store_true', help="Use OHEM Cross Entropy loss")
    parser.add_argument("--use_focal_loss", action='store_true', help="Use Focal Loss instead of Cross Entropy.")
    parser.add_argument("--focal_gamma", type=float, default=2.5, help="Gamma value for Focal Loss.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze DDRNet backbone during training.")
    parser.add_argument("--visualize_augmentations", action='store_true', help="Visualize augmentations and exit.")
    parser.add_argument("--vis_count", type=int, default=3)

    args = parser.parse_args()
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    train_and_validate(args)
