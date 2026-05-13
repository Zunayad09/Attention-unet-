"""
Attention U-Net - Evaluation & Visualization
- Loss / Dice / IoU on test set
- Training loss vs Validation loss graph over epochs
- Ground truth vs Prediction plots
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from AttentionUnet import (
    AttentionUNet, load_data, DiceBCELoss, dice_coef, iou_coef,
    CSV_PATH, IMG_DIR, MASK_DIR, CHECKPOINT_DIR,
    IMAGE_SIZE, DROPOUT_RATE, BATCH_NORM
)


# =============================================================================
# PLOT TRAINING HISTORY (Loss & Dice vs Epochs)
# =============================================================================

def plot_training_history():
    history_path = os.path.join(CHECKPOINT_DIR, 'training_history.csv')

    if not os.path.exists(history_path):
        print(f"[WARNING] training_history.csv not found at: {history_path}")
        print("  Run training first to generate it.")
        return

    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Attention U-Net - Training History', fontsize=14, fontweight='bold')

    # --- Loss ---
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(df['epoch'], df['val_loss'],   label='Val Loss',   color='orange')
    axes[0].set_title('Loss vs Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # --- Dice ---
    axes[1].plot(df['epoch'], df['train_dice'], label='Train Dice', color='green')
    axes[1].plot(df['epoch'], df['val_dice'],   label='Val Dice',   color='red')
    axes[1].set_title('Dice vs Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True)

    # --- IoU ---
    axes[2].plot(df['epoch'], df['val_iou'], label='Val IoU', color='purple')
    axes[2].set_title('IoU vs Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = os.path.join(CHECKPOINT_DIR, 'training_history_plot.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# =============================================================================
# PREDICT SINGLE IMAGE
# =============================================================================

def predict_single(model, img_path, mask_path, device):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, IMAGE_SIZE)
    mask  = cv2.resize(mask,  IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    img_norm = image.astype(np.float32) / 255.0
    inp = torch.from_numpy(img_norm[np.newaxis, np.newaxis, ...]).to(device)

    with torch.no_grad():
        output = model(inp)
        pred = (torch.sigmoid(output)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    ground_truth = (mask > 127).astype(np.uint8)
    return image, ground_truth, pred


# =============================================================================
# EVALUATE TEST SET - Loss / Dice / IoU
# =============================================================================

def evaluate_test_set(model, test_x, test_y, device):
    model.eval()
    loss_fn = DiceBCELoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    print("\nEvaluating test set...")
    for img_path, mask_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        image = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
        mask  = (cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)

        img_t  = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).to(device)
        mask_t = torch.from_numpy(mask [np.newaxis, np.newaxis, ...]).to(device)

        with torch.no_grad():
            output = model(img_t)
            loss   = loss_fn(output, mask_t)

        total_loss += loss.item()
        total_dice += dice_coef(output, mask_t)
        total_iou  += iou_coef(output,  mask_t)

    n = len(test_x)
    avg_loss = total_loss / n
    avg_dice = total_dice / n
    avg_iou  = total_iou  / n

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"  Loss : {avg_loss:.4f}")
    print(f"  Dice : {avg_dice:.4f}")
    print(f"  IoU  : {avg_iou:.4f}")
    print("=" * 50)

    return avg_loss, avg_dice, avg_iou


# =============================================================================
# VISUALIZE - Single prediction (matches screenshot layout)
# =============================================================================

def visualize_single(model, test_x, test_y, device):
    idx = random.randint(0, len(test_x) - 1)
    image, ground_truth, pred = predict_single(model, test_x[idx], test_y[idx], device)

    plt.figure(figsize=(16, 8))

    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.axis('off')

    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(pred, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(CHECKPOINT_DIR, 'single_prediction.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# =============================================================================
# VISUALIZE - Multiple samples
# =============================================================================

def visualize_multiple(model, test_x, test_y, device, num_samples=5):
    indices = random.sample(range(len(test_x)), min(num_samples, len(test_x)))

    _, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        image, ground_truth, pred = predict_single(model, test_x[idx], test_y[idx], device)

        axes[row][0].imshow(image, cmap='gray')
        axes[row][0].set_title('Testing Image')
        axes[row][0].axis('off')

        axes[row][1].imshow(ground_truth, cmap='gray')
        axes[row][1].set_title('Testing Label')
        axes[row][1].axis('off')

        axes[row][2].imshow(pred, cmap='gray')
        axes[row][2].set_title('Prediction on test image')
        axes[row][2].axis('off')

    plt.suptitle('Attention U-Net - Test Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(CHECKPOINT_DIR, 'multiple_predictions.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("ATTENTION U-NET - EVALUATION")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1 - Training history graphs (loss/dice/iou vs epochs)
    print("\nPlotting training history...")
    plot_training_history()

    # Load data splits
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(CSV_PATH, IMG_DIR, MASK_DIR)
    print(f"Test samples: {len(test_x)}")

    # Load trained model
    model = AttentionUNet(
        in_channels=1, num_classes=1,
        dropout_rate=DROPOUT_RATE, batch_norm=BATCH_NORM
    ).to(device)

    weights_path = os.path.join(CHECKPOINT_DIR, 'best_attention_unet.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded weights: {weights_path}")

    # 2 - Test set metrics
    evaluate_test_set(model, test_x, test_y, device)

    # 3 - Single prediction plot
    print("\nSingle prediction plot...")
    visualize_single(model, test_x, test_y, device)

    # 4 - Multiple predictions plot
    print("\n5-sample predictions plot...")
    visualize_multiple(model, test_x, test_y, device, num_samples=5)

    print("\nDone! All plots saved in:", CHECKPOINT_DIR)
