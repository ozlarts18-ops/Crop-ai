import os
import glob
import time
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from src.config import CNN_DIR, MODELS_DIR, LOGS_DIR

def get_dataloaders(batch_size=64):
    train_dir = os.path.join(CNN_DIR, "train")
    val_dir = os.path.join(CNN_DIR, "val")
    
    # Train augmentation (ONLY applied to train set as instructed)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, train_dataset.classes

def build_model(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def compute_class_weights(dataset, device):
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    # Inverse frequency weights
    weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float, device=device)

def log_checkpoint(checkpoint_path, epoch, val_loss, val_acc, val_f1):
    chk_log = os.path.join(LOGS_DIR, "checkpoint_log.csv")
    file_exists = os.path.exists(chk_log)
    with open(chk_log, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model_type", "checkpoint_path", "epoch", "val_loss", "val_acc", "val_f1"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "EfficientNet-B0",
            checkpoint_path,
            epoch,
            f"{val_loss:.4f}",
            f"{val_acc:.4f}",
            f"{val_f1:.4f}"
        ])

def train_cnn(epochs=100, batch_size=64, lr=1e-3):
    print("="*60)
    print("STARTING CNN TRAINING (EfficientNet-B0 for 100 EPOCHS)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    train_loader, val_loader, class_names = get_dataloaders(batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    
    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    class_weights = compute_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    checkpoint_dir = os.path.join(MODELS_DIR, "cnn", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_file = os.path.join(LOGS_DIR, "cnn_training.log")
    
    # Check for existing checkpoints to support automatic resumption
    existing_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    start_epoch = 1
    best_val_f1 = 0.0
    
    if existing_checkpoints:
        latest_checkpoint = existing_checkpoints[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        ckpt = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"Resumed at Epoch {start_epoch}, previous best Val F1: {best_val_f1:.4f}")
    else:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Epoch,Train_Loss,Val_Loss,LR,Time_s,Val_Acc,Val_Precision,Val_Recall,Val_F1\n")

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to file and console
        log_line = f"{epoch},{train_loss:.4f},{val_loss:.4f},{current_lr:.6f},{epoch_time:.2f},{val_acc:.4f},{val_prec:.4f},{val_rec:.4f},{val_f1:.4f}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
            
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")
        
        checkpoint_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "class_names": class_names,
            "configuration": {
                "arch": "EfficientNet-B0",
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs
            }
        }
        
        # Save models/cnn/last.pt
        torch.save(checkpoint_dict, os.path.join(MODELS_DIR, "cnn", "last.pt"))
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_dict["best_val_f1"] = best_val_f1
            torch.save(checkpoint_dict, os.path.join(MODELS_DIR, "cnn", "best.pt"))
            print(f"  --> Saved new best model (Val F1: {best_val_f1:.4f})")
            
        # MANDATORY 5-EPOCH CHECKPOINT
        if epoch % 5 == 0:
            ckpt_name = f"epoch_{epoch:03d}.pt"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            torch.save(checkpoint_dict, ckpt_path)
            log_checkpoint(ckpt_path, epoch, val_loss, val_acc, val_f1)
            print(f"  [CHECKPOINT] Successfully saved mandatory checkpoint: {ckpt_name}")

    print("CNN Training Complete!")
    return os.path.join(MODELS_DIR, "cnn", "best.pt")

if __name__ == "__main__":
    train_cnn()
