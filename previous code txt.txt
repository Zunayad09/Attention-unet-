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


BASE_DIR = r"D:\Capstone project\BUSBRA"
IMG_DIR = os.path.join(BASE_DIR, "Images")
MASK_DIR = os.path.join(BASE_DIR, "Masks")
CSV_PATH = os.path.join(BASE_DIR, "bus_data.csv")

CHECKPOINT_DIR = r"D:\Capstone project\Attention Unet\checkpoints_attention_unet"

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


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        dice = dice_coef(outputs, masks)
        epoch_loss += loss.item()
        epoch_dice += dice

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0

    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            dice = dice_coef(outputs, masks)
            iou = iou_coef(outputs, masks)

            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_iou / len(loader)


def main():
    print("=" * 70)
    print("ATTENTION U-NET TRAINING")
    print("Following: https://github.com/MoleImg/Attention_UNet")
    print("With Attention Gates in Decoder")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
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

    print("\nBuilding Attention U-Net...")
    model = AttentionUNet(in_channels=1, num_classes=1, dropout_rate=DROPOUT_RATE, batch_norm=BATCH_NORM).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print("Note: Includes attention gates for better feature selection")

    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("\nStarting training...")
    print("=" * 70)

    best_val_dice = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_dice = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, loss_fn, device)

        elapsed = time.time() - start_time
        mins, secs = int(elapsed / 60), int(elapsed % 60)

        print(f"\nEpoch [{epoch:03d}/{NUM_EPOCHS}] {mins}m {secs}s")
        print(f"  Train Dice: {train_dice:.4f}")
        print(f"  Val Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_attention_unet.pth'))
            print(f"  ✓ New best! Saved.")

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'attention_unet_epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'attention_unet_final.pth'))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Trained for: {NUM_EPOCHS} epochs")
    print("=" * 70)


if __name__ == "__main__":
    main()