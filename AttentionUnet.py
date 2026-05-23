import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


BASE_DIR = r"D:\Capstone project\BUSBRA"
IMG_DIR = os.path.join(BASE_DIR, "Images")
MASK_DIR = os.path.join(BASE_DIR, "Masks")
CSV_PATH = os.path.join(BASE_DIR, "bus_data.csv")

CHECKPOINT_DIR = r"D:\Capstone project\Attention Unet\checkpoints_attention_unet"
UNCERTAINTY_DIR = os.path.join(CHECKPOINT_DIR, "uncertainty_results")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.0
BATCH_NORM = True

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

# Monte Carlo Dropout Parameters
MC_PASSES = 20
MC_DROPOUT_RATE = 0.10
MC_NUM_LAYERS = 5


def conv_block(in_channels, out_channels, dropout_rate=0.0, batch_norm=True):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    if dropout_rate > 0:
        layers.append(nn.Dropout2d(dropout_rate))
    return nn.Sequential(*layers)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.0, batch_norm=True):
        super().__init__()
        FILTER_NUM = 64
        self.conv_128 = conv_block(in_channels, FILTER_NUM, dropout_rate, batch_norm)
        self.pool_64 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_64 = conv_block(FILTER_NUM, 2*FILTER_NUM, dropout_rate, batch_norm)
        self.pool_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_32 = conv_block(2*FILTER_NUM, 4*FILTER_NUM, dropout_rate, batch_norm)
        self.pool_16 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_16 = conv_block(4*FILTER_NUM, 8*FILTER_NUM, dropout_rate, batch_norm)
        self.pool_8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_8 = conv_block(8*FILTER_NUM, 16*FILTER_NUM, dropout_rate, batch_norm)
        
        self.att_16 = AttentionBlock(F_g=16*FILTER_NUM, F_l=8*FILTER_NUM, F_int=8*FILTER_NUM)
        self.att_32 = AttentionBlock(F_g=8*FILTER_NUM, F_l=4*FILTER_NUM, F_int=4*FILTER_NUM)
        self.att_64 = AttentionBlock(F_g=4*FILTER_NUM, F_l=2*FILTER_NUM, F_int=2*FILTER_NUM)
        self.att_128 = AttentionBlock(F_g=2*FILTER_NUM, F_l=FILTER_NUM, F_int=FILTER_NUM)
        
        self.up_16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_16 = conv_block(16*FILTER_NUM + 8*FILTER_NUM, 8*FILTER_NUM, dropout_rate, batch_norm)
        self.up_32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_32 = conv_block(8*FILTER_NUM + 4*FILTER_NUM, 4*FILTER_NUM, dropout_rate, batch_norm)
        self.up_64 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_64 = conv_block(4*FILTER_NUM + 2*FILTER_NUM, 2*FILTER_NUM, dropout_rate, batch_norm)
        self.up_128 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_128 = conv_block(2*FILTER_NUM + FILTER_NUM, FILTER_NUM, dropout_rate, batch_norm)
        
        self.conv_final = nn.Conv2d(FILTER_NUM, num_classes, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(num_classes) if batch_norm else nn.Identity()

    def forward(self, x):
        conv_128 = self.conv_128(x)
        pool_64 = self.pool_64(conv_128)
        conv_64 = self.conv_64(pool_64)
        pool_32 = self.pool_32(conv_64)
        conv_32 = self.conv_32(pool_32)
        pool_16 = self.pool_16(conv_32)
        conv_16 = self.conv_16(pool_16)
        pool_8 = self.pool_8(conv_16)
        conv_8 = self.conv_8(pool_8)
        
        att_conv_16 = self.att_16(g=conv_8, x=conv_16)
        up_16 = self.up_16(conv_8)
        up_16 = torch.cat([up_16, att_conv_16], dim=1)
        up_conv_16 = self.up_conv_16(up_16)
        
        att_conv_32 = self.att_32(g=up_conv_16, x=conv_32)
        up_32 = self.up_32(up_conv_16)
        up_32 = torch.cat([up_32, att_conv_32], dim=1)
        up_conv_32 = self.up_conv_32(up_32)
        
        att_conv_64 = self.att_64(g=up_conv_32, x=conv_64)
        up_64 = self.up_64(up_conv_32)
        up_64 = torch.cat([up_64, att_conv_64], dim=1)
        up_conv_64 = self.up_conv_64(up_64)
        
        att_conv_128 = self.att_128(g=up_conv_64, x=conv_128)
        up_128 = self.up_128(up_conv_64)
        up_128 = torch.cat([up_128, att_conv_128], dim=1)
        up_conv_128 = self.up_conv_128(up_128)
        
        conv_final = self.conv_final(up_conv_128)
        conv_final = self.final_bn(conv_final)
        return conv_final


class BUSBRADataset(Dataset):
    def __init__(self, images_path, masks_path, size=(256, 256), transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask)


def load_data(csv_path, img_dir, mask_dir):
    df = pd.read_csv(csv_path)
    unique_cases = df["Case"].unique()
    train_cases, temp_cases = train_test_split(
        unique_cases, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    val_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=val_test_ratio, random_state=RANDOM_SEED
    )
    def get_paths(subset_df):
        img_paths, mask_paths = [], []
        for _, row in subset_df.iterrows():
            img_id = row["ID"]
            img_path = os.path.join(img_dir, img_id + ".png")
            mask_path = os.path.join(mask_dir, img_id.replace("bus_", "mask_") + ".png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
        return img_paths, mask_paths
    train_x, train_y = get_paths(df[df["Case"].isin(train_cases)])
    val_x, val_y = get_paths(df[df["Case"].isin(val_cases)])
    test_x, test_y = get_paths(df[df["Case"].isin(test_cases)])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def dice_coef(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + 1.0) / (pred.sum() + target.sum() + 1.0)
    return dice.item()


def iou_coef(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return iou.item()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2.0 * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
        bce_loss = self.bce(pred, target)
        return 0.5 * dice_loss + 0.5 * bce_loss


def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
        dice = dice_coef(outputs, masks)
        epoch_loss += loss.item()
        epoch_dice += dice
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate_epoch(model, loader, loss_fn, device, scaler=None):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            dice = dice_coef(outputs, masks)
            iou = iou_coef(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_iou / len(loader)


def inject_mc_dropout(model, dropout_rate=0.1, num_layers=5):
    """Inject MC Dropout into the model."""
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1]
    target_layers = conv_layers[-num_layers:]
    handles = []
    for layer in target_layers:
        dropout = nn.Dropout2d(p=dropout_rate)
        def make_hook(drop):
            def hook(module, inp, out):
                if out.dim() == 4:
                    return drop(out)
                return out
            return hook
        handle = layer.register_forward_hook(make_hook(dropout))
        handles.append(handle)
    return handles, len(handles)


def compute_boundary_uncertainty(model, test_loader, device):
    """Compute BOUNDARY-ONLY uncertainty."""
    print("\n" + "="*70)
    print("MONTE CARLO DROPOUT - BOUNDARY UNCERTAINTY")
    print("="*70)
    
    handles, n_hooks = inject_mc_dropout(model, MC_DROPOUT_RATE, MC_NUM_LAYERS)
    print(f"✓ MC Dropout injected on {n_hooks} layers")
    print(f"  MC Passes: {MC_PASSES}")
    print(f"  Dropout Rate: {MC_DROPOUT_RATE}")
    
    model.eval()
    all_predictions = []
    all_uncertainties = []
    all_masks = []
    all_images = []
    
    TP = TN = FP = FN = 0
    total_boundary_uncertainty = 0.0
    boundary_pixels_count = 0
    
    for images, masks in tqdm(test_loader, desc='Computing Uncertainty'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # MC Dropout passes
        mc_preds = []
        with torch.no_grad():
            for _ in range(MC_PASSES):
                mc_preds.append(torch.sigmoid(model(images)))
        mc_preds = torch.stack(mc_preds, dim=0)
        mean_pred = mc_preds.mean(dim=0)
        variance = mc_preds.var(dim=0)
        pred_binary = (mean_pred > 0.5).float()
        
        # Metrics
        p = pred_binary.view(-1)
        t = masks.view(-1)
        TP += (p * t).sum().item()
        TN += ((1-p) * (1-t)).sum().item()
        FP += (p * (1-t)).sum().item()
        FN += ((1-p) * t).sum().item()
        
        # Boundary uncertainty
        for i in range(images.size(0)):
            mask_np = masks[i, 0].cpu().numpy()
            var_np = variance[i, 0].cpu().numpy()
            kernel = np.ones((5, 5), np.uint8)
            boundary = cv2.dilate(mask_np, kernel, iterations=1) - cv2.erode(mask_np, kernel, iterations=1)
            if boundary.sum() > 0:
                total_boundary_uncertainty += var_np[boundary > 0].mean()
                boundary_pixels_count += 1
        
        all_predictions.append(mean_pred.cpu())
        all_uncertainties.append(variance.cpu())
        all_masks.append(masks.cpu())
        all_images.append(images.cpu())
    
    for h in handles:
        h.remove()
    
    eps = 1e-8
    dice = 2*TP / (2*TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    avg_boundary_unc = total_boundary_uncertainty / max(boundary_pixels_count, 1)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Dice:     {dice:.4f}")
    print(f"IoU:      {iou:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Boundary Uncertainty: {avg_boundary_unc:.6f}")
    print("="*70)
    
    return {
        'dice': dice, 'iou': iou, 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1,
        'boundary_uncertainty': avg_boundary_unc,
        'predictions': all_predictions,
        'uncertainties': all_uncertainties,
        'masks': all_masks,
        'images': all_images
    }


def visualize_boundary_only(results, save_dir, num_samples=10):
    """Visualize BOUNDARY-ONLY uncertainty."""
    os.makedirs(save_dir, exist_ok=True)
    
    sample_count = 0
    for batch_idx in range(len(results['predictions'])):
        if sample_count >= num_samples:
            break
        batch_preds = results['predictions'][batch_idx]
        batch_uncerts = results['uncertainties'][batch_idx]
        batch_masks = results['masks'][batch_idx]
        batch_images = results['images'][batch_idx]
        
        for i in range(batch_preds.size(0)):
            if sample_count >= num_samples:
                break
            
            img = batch_images[i, 0].numpy()
            mask_gt = batch_masks[i, 0].numpy()
            pred_mean = batch_preds[i, 0].numpy()
            pred_var = batch_uncerts[i, 0].numpy()
            pred_binary = (pred_mean > 0.5).astype(np.float32)
            
            # Extract boundary
            kernel = np.ones((5, 5), np.uint8)
            boundary = cv2.dilate(mask_gt, kernel, iterations=1) - cv2.erode(mask_gt, kernel, iterations=1)
            
            # Normalize uncertainty
            unc_norm = (pred_var - pred_var.min()) / (pred_var.max() - pred_var.min() + 1e-8)
            
            # BOUNDARY ONLY
            boundary_unc = np.full_like(unc_norm, np.nan)
            boundary_unc[boundary > 0] = unc_norm[boundary > 0]
            
            # Visualize
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img, cmap='gray')
            axes[0, 1].contour(mask_gt, colors='green', linewidths=3)
            axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(img, cmap='gray')
            axes[1, 0].contour(pred_binary, colors='blue', linewidths=3)
            axes[1, 0].set_title('Prediction', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            # BOUNDARY UNCERTAINTY ONLY
            axes[1, 1].imshow(img, cmap='gray')
            cmap = mcolors.LinearSegmentedColormap.from_list('conf', ['green', 'yellow', 'red'], N=100)
            im = axes[1, 1].imshow(boundary_unc, cmap=cmap, alpha=0.85, vmin=0, vmax=1)
            axes[1, 1].set_title('⭐ BOUNDARY UNCERTAINTY ⭐', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
            
            dice = 2*np.sum(pred_binary*mask_gt)/(np.sum(pred_binary)+np.sum(mask_gt)+1e-8)
            fig.suptitle(f'Sample {sample_count+1} | Dice: {dice:.4f}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{sample_count+1}.png'), dpi=200, bbox_inches='tight')
            plt.close()
            sample_count += 1
    
    print(f"✓ Saved {sample_count} visualizations")


def plot_performance_metrics(results, save_dir):
    """Visual performance chart."""
    metrics = ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1']
    values = [results['dice'], results['iou'], results['accuracy'], 
              results['precision'], results['recall'], results['f1']]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.ylabel('Score', fontsize=13, fontweight='bold')
    plt.title('Segmentation Performance', fontsize=15, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance chart saved")


def main():
    print("=" * 70)
    print("ATTENTION U-NET + BOUNDARY UNCERTAINTY")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(UNCERTAINTY_DIR, exist_ok=True)

    # GPU/CPU STATUS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("DEVICE INFORMATION")
    print(f"{'='*70}")

    if torch.cuda.is_available():
        print(f"✓ Running on: GPU")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        use_amp = True
        scaler = torch.amp.GradScaler('cuda')
        print(f"  Mixed Precision (AMP): Enabled")
    else:
        print(f"⚠ Running on: CPU")
        print(f"  WARNING: Training will be VERY slow on CPU!")
        print(f"  Recommendation: Use GPU for faster training")
        use_amp = False
        scaler = None

    print(f"{'='*70}\n")

    print("Loading data...")
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(CSV_PATH, IMG_DIR, MASK_DIR)
    print(f"Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
    ])

    train_dataset = BUSBRADataset(train_x, train_y, IMAGE_SIZE, train_transform)
    val_dataset = BUSBRADataset(val_x, val_y, IMAGE_SIZE, None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = AttentionUNet(in_channels=1, num_classes=1, dropout_rate=DROPOUT_RATE, batch_norm=BATCH_NORM).to(device)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    best_val_dice = 0.0
    print("\nTraining...")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, loss_fn, device, scaler)

        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")
        print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_attention_unet.pth'))
            print(f"  ✓ Best model saved!")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    # UNCERTAINTY EVALUATION
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_attention_unet.pth')))
    test_dataset = BUSBRADataset(test_x, test_y, IMAGE_SIZE, None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    results = compute_boundary_uncertainty(model, test_loader, device)
    visualize_boundary_only(results, UNCERTAINTY_DIR, num_samples=10)
    plot_performance_metrics(results, UNCERTAINTY_DIR)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()