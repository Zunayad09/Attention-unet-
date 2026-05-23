import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.model_selection import train_test_split

# ========== COPY THESE FROM YOUR TRAINING SCRIPT ==========
# (Just the class definitions - no training code!)

# 1. Copy your conv_block function
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

# 2. Copy your AttentionBlock class
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# 3. Copy your AttentionUNet class (abbreviated here)
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.0, batch_norm=True):
        super().__init__()
        F = 64
        self.conv_128 = conv_block(in_channels, F, dropout_rate, batch_norm)
        self.pool_64 = nn.MaxPool2d(2, 2)
        self.conv_64 = conv_block(F, 2*F, dropout_rate, batch_norm)
        self.pool_32 = nn.MaxPool2d(2, 2)
        self.conv_32 = conv_block(2*F, 4*F, dropout_rate, batch_norm)
        self.pool_16 = nn.MaxPool2d(2, 2)
        self.conv_16 = conv_block(4*F, 8*F, dropout_rate, batch_norm)
        self.pool_8 = nn.MaxPool2d(2, 2)
        self.conv_8 = conv_block(8*F, 16*F, dropout_rate, batch_norm)
        self.att_16 = AttentionBlock(16*F, 8*F, 8*F)
        self.att_32 = AttentionBlock(8*F, 4*F, 4*F)
        self.att_64 = AttentionBlock(4*F, 2*F, 2*F)
        self.att_128 = AttentionBlock(2*F, F, F)
        self.up_16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_16 = conv_block(24*F, 8*F, dropout_rate, batch_norm)
        self.up_32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_32 = conv_block(12*F, 4*F, dropout_rate, batch_norm)
        self.up_64 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_64 = conv_block(6*F, 2*F, dropout_rate, batch_norm)
        self.up_128 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_128 = conv_block(3*F, F, dropout_rate, batch_norm)
        self.conv_final = nn.Conv2d(F, num_classes, 1)
        self.final_bn = nn.BatchNorm2d(num_classes) if batch_norm else nn.Identity()
    
    def forward(self, x):
        c128 = self.conv_128(x)
        c64 = self.conv_64(self.pool_64(c128))
        c32 = self.conv_32(self.pool_32(c64))
        c16 = self.conv_16(self.pool_16(c32))
        c8 = self.conv_8(self.pool_8(c16))
        u16 = self.up_conv_16(torch.cat([self.up_16(c8), self.att_16(c8, c16)], 1))
        u32 = self.up_conv_32(torch.cat([self.up_32(u16), self.att_32(u16, c32)], 1))
        u64 = self.up_conv_64(torch.cat([self.up_64(u32), self.att_64(u32, c64)], 1))
        u128 = self.up_conv_128(torch.cat([self.up_128(u64), self.att_128(u64, c128)], 1))
        return self.final_bn(self.conv_final(u128))

# 4. Copy your Dataset class
class BUSBRADataset(Dataset):
    def __init__(self, images_path, masks_path, size=(256, 256)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx], 0)
        msk = cv2.imread(self.masks_path[idx], 0)
        img = cv2.resize(img, self.size).astype(np.float32) / 255.0
        msk = (cv2.resize(msk, self.size, interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        return torch.from_numpy(img[None]), torch.from_numpy(msk[None])

# ========== PATHS ==========
BASE_DIR = r"D:\Capstone project\BUSBRA"
CHECKPOINT = r"D:\Capstone project\Attention Unet\checkpoints_attention_unet\best_attention_unet.pth"
OUTPUT_DIR = r"D:\Capstone project\Attention Unet\checkpoints_attention_unet\uncertainty_PREDICTED_boundary"

# ========== MC DROPOUT ==========
def inject_mc_dropout(model, rate=0.1, n=5):
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1]
    handles = []
    for layer in convs[-n:]:
        drop = nn.Dropout2d(rate)
        handles.append(layer.register_forward_hook(lambda m, i, o, d=drop: d(o) if o.dim()==4 else o))
    return handles

# ========== QUICK RUN ==========
def quick_uncertainty():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"✓ Loaded model from {CHECKPOINT}")
    
    # Load test data
    df = pd.read_csv(os.path.join(BASE_DIR, "bus_data.csv"))
    cases = df["Case"].unique()
    _, temp = train_test_split(cases, test_size=0.3, random_state=42)
    _, test = train_test_split(temp, test_size=0.33, random_state=42)
    test_df = df[df["Case"].isin(test)]
    
    test_x, test_y = [], []
    for _, row in test_df.iterrows():
        img_p = os.path.join(BASE_DIR, "Images", row["ID"] + ".png")
        msk_p = os.path.join(BASE_DIR, "Masks", row["ID"].replace("bus_", "mask_") + ".png")
        if os.path.exists(img_p) and os.path.exists(msk_p):
            test_x.append(img_p)
            test_y.append(msk_p)
    
    loader = DataLoader(BUSBRADataset(test_x, test_y), batch_size=8, shuffle=False)
    print(f"✓ Loaded {len(test_x)} test samples")
    
    # Inject dropout
    handles = inject_mc_dropout(model, 0.1, 5)
    model.eval()
    
    # Run MC
    all_imgs, all_msks, all_preds, all_vars = [], [], [], []
    for imgs, msks in tqdm(loader, desc="Computing"):
        imgs, msks = imgs.to(device), msks.to(device)
        preds = torch.stack([torch.sigmoid(model(imgs)) for _ in range(20)])
        all_imgs.append(imgs.cpu())
        all_msks.append(msks.cpu())
        all_preds.append(preds.mean(0).cpu())
        all_vars.append(preds.var(0).cpu())
    
    for h in handles: h.remove()
    
    # ✅ VISUALIZE WITH PREDICTED BOUNDARY
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(min(10, len(all_imgs))):
        for j in range(all_imgs[i].size(0)):
            if i*all_imgs[i].size(0)+j >= 10: break
            
            img = all_imgs[i][j,0].numpy()
            msk = all_msks[i][j,0].numpy()
            pred = all_preds[i][j,0].numpy()
            var = all_vars[i][j,0].numpy()
            pred_bin = (pred > 0.5).astype(float)
            
            # ✅ PREDICTED boundary
            k = np.ones((5,5), np.uint8)
            bnd = cv2.dilate(pred_bin, k, 1) - cv2.erode(pred_bin, k, 1)
            
            unc = (var - var.min()) / (var.max() - var.min() + 1e-8)
            bnd_unc = np.full_like(unc, np.nan)
            bnd_unc[bnd > 0] = unc[bnd > 0]
            
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))
            ax[0,0].imshow(img, 'gray'); ax[0,0].set_title('Input'); ax[0,0].axis('off')
            ax[0,1].imshow(img, 'gray'); ax[0,1].contour(msk, colors='g', linewidths=3)
            ax[0,1].set_title('Ground Truth'); ax[0,1].axis('off')
            ax[1,0].imshow(img, 'gray'); ax[1,0].contour(pred_bin, colors='b', linewidths=3)
            ax[1,0].set_title('Prediction'); ax[1,0].axis('off')
            ax[1,1].imshow(img, 'gray')
            cmap = mcolors.LinearSegmentedColormap.from_list('c', ['green','yellow','red'], 100)
            im = ax[1,1].imshow(bnd_unc, cmap=cmap, alpha=0.85, vmin=0, vmax=1)
            ax[1,1].set_title('⭐ PREDICTED BOUNDARY ⭐'); ax[1,1].axis('off')
            plt.colorbar(im, ax=ax[1,1], fraction=0.046)
            
            dice = 2*np.sum(pred_bin*msk)/(np.sum(pred_bin)+np.sum(msk)+1e-8)
            fig.suptitle(f'Sample {i*all_imgs[i].size(0)+j+1} | Dice: {dice:.4f}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/sample_{i*all_imgs[i].size(0)+j+1}.png', dpi=200, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    quick_uncertainty()