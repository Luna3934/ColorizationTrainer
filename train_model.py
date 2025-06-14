# 1. Imports
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage.color import rgb2lab
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
from dataset import ColorizationDataset
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import models
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import cv2



# 3. Model
class ColorizationNet(nn.Module):
    def __init__(self, freeze_encoder=True):
        super(ColorizationNet, self).__init__()

        weights = VGG16_BN_Weights.DEFAULT
        vgg = vgg16_bn(weights=weights)

        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(*features[:6])
        self.enc2 = nn.Sequential(*features[6:13])
        self.enc3 = nn.Sequential(*features[13:23])
        self.enc4 = nn.Sequential(*features[23:33])

        if freeze_encoder:
            for module in [self.enc1, self.enc2, self.enc3, self.enc4]:
                for param in module.parameters():
                    param.requires_grad = False

        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.final = nn.Conv2d(128, 313, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        out = self.final(d3)
        return out

def confidence_weighted_blur(ab_img, confidence, threshold=0.6, kernel_size=9):
    # Expand confidence to shape (2, H, W)
    confidence_exp = np.broadcast_to(confidence, ab_img.shape)

    # Create binary mask of where confidence is above the threshold
    high_conf_mask = (confidence_exp >= threshold).astype(float)

    # Multiply ab by high-confidence mask (zeroing out low-confidence pixels)
    masked_ab = ab_img * high_conf_mask

    # Smooth both numerator and denominator
    smoothed_ab = uniform_filter(masked_ab, size=(1, kernel_size, kernel_size))
    smoothed_mask = uniform_filter(high_conf_mask, size=(1, kernel_size, kernel_size))

    # Avoid division by zero
    smoothed_mask = np.clip(smoothed_mask, 1e-5, None)

    # Weighted average
    weighted_blur = smoothed_ab / smoothed_mask
    return weighted_blur



def suppress_red_splotches(rgb_img, confidence_map, sat_thresh=0.6, red_range=((0, 0.05), (0.95, 1.0)), conf_thresh=0.4):
    """
    Suppress overly saturated red pixels that are also low confidence by blending with a blurred version based on confidence.
    """
    hsv = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Build red mask
    red_mask = (
        (sat > sat_thresh) & (confidence_map < conf_thresh) & (
            ((hue >= red_range[0][0]) & (hue <= red_range[0][1])) |
            ((hue >= red_range[1][0]) & (hue <= red_range[1][1]))
        )
    )

    # Create blurred RGB version
    blurred_rgb = cv2.GaussianBlur((rgb_img * 255).astype(np.uint8), (5, 5), 0) / 255.0
    
    # Create per-pixel blend weight based on confidence (lower confidence = more blur)
    blend_weight = 1.0 - confidence_map  # shape (H, W)
    blend_weight = np.clip(blend_weight+0.1, 0.0, 1.0)

    # Expand for RGB channels
    blend_weight = np.repeat(blend_weight[:, :, np.newaxis], 3, axis=2)

    # Only apply weights to red-mask regions
    red_mask_3ch = np.repeat(red_mask[:, :, np.newaxis], 3, axis=2)
    blend_weight *= red_mask_3ch

    # Blend
    result = rgb_img * (1 - blend_weight) + blurred_rgb * blend_weight
    return result

def suppress_color_splotches(rgb_img, confidence_map, sat_thresh=0.7, conf_thresh=0.4):
    """
    Suppress overly saturated colorful pixels (e.g., red, green, yellow) with low confidence
    by blending with a blurred version based on confidence.
    """
    hsv = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Mask of any high-saturation, low-confidence color
    color_mask = (sat > sat_thresh) & (confidence_map < conf_thresh)

    # Create blurred RGB version
    blurred_rgb = cv2.GaussianBlur((rgb_img * 255).astype(np.uint8), (5, 5), 0) / 255.0

    # Create per-pixel blend weight based on confidence (lower confidence = more blur)
    blend_weight = (1.0 - confidence_map) + 0.1  # shape (H, W)
    blend_weight = np.clip(blend_weight, 0.0, 1.0)

    # Only apply weights to masked areas
    color_mask_3ch = np.repeat(color_mask[:, :, np.newaxis], 3, axis=2)
    blend_weight = np.repeat(blend_weight[:, :, np.newaxis], 3, axis=2)
    blend_weight *= color_mask_3ch

    # Blend blurred and original
    result = rgb_img * (1 - blend_weight) + blurred_rgb * blend_weight
    return result


def suppress_color_splotches2(rgb_img, confidence_map, sat_thresh=0.7, conf_thresh=0.4, kernel_size=20, min_support=0.1):
    """
    Suppress splotchy regions in high-saturation, low-confidence pixels using neighborhood-aware repair.
    
    Parameters:
        rgb_img (np.ndarray): Input RGB image in [0, 1], shape (H, W, 3)
        confidence_map (np.ndarray): Grayscale confidence map in [0, 1], shape (H, W)
        sat_thresh (float): Minimum saturation to consider a pixel "splotchy"
        conf_thresh (float): Maximum confidence for splotch detection
        kernel_size (int): Size of the uniform filter kernel
        min_support (float): Minimum fraction of confident pixels in the neighborhood to apply averaging

    Returns:
        np.ndarray: Smoothed RGB image
    """
    # Convert to HSV for saturation/hue filtering
    hsv = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Detect splotchy regions (high sat + low conf)
    splotch_mask = (sat > sat_thresh) & (confidence_map < conf_thresh)
    splotch_mask_3ch = np.repeat(splotch_mask[:, :, np.newaxis], 3, axis=2)

    # Expand confidence for all channels
    confidence_3ch = np.repeat(confidence_map[:, :, np.newaxis], 3, axis=2)

    # Weighted image and mask for confident pixels
    weighted_rgb = rgb_img * confidence_3ch
    valid_mask = (confidence_map >= conf_thresh).astype(np.float32)
    valid_mask_3ch = np.repeat(valid_mask[:, :, np.newaxis], 3, axis=2)

    # Smooth the weighted RGB and mask
    smoothed_rgb = uniform_filter(weighted_rgb, size=(kernel_size, kernel_size, 1))
    smoothed_mask = uniform_filter(valid_mask_3ch, size=(kernel_size, kernel_size, 1))
    smoothed_mask = np.clip(smoothed_mask, 1e-5, 1.0)

    # Neighborhood average from confident pixels
    neighborhood_avg = smoothed_rgb / smoothed_mask
    neighborhood_avg = np.nan_to_num(neighborhood_avg, nan=0.0)

    # Check if enough confident pixels are present in the neighborhood
    support_mask = (smoothed_mask[..., 0] >= min_support)
    support_mask_3ch = np.repeat(support_mask[:, :, np.newaxis], 3, axis=2)

    # Final mask for valid replacements
    replace_mask = splotch_mask_3ch & support_mask_3ch

    # Fallback blur for unsupported splotches
    fallback_blur = cv2.GaussianBlur((rgb_img * 255).astype(np.uint8), (5, 5), 0) / 255.0
    fallback_mask = splotch_mask_3ch & ~support_mask_3ch

    # Apply replacements
    result = rgb_img.copy()
    result[replace_mask] = neighborhood_avg[replace_mask]
    result[fallback_mask] = fallback_blur[fallback_mask]

    return result


def suppress_color_splotches3(rgb_img, confidence_map, sat_thresh=0.7, conf_thresh=0.4, neighborhood=7, support_thresh=0.15):
    """
    Replace high-saturation low-confidence pixels with the average of nearby confident pixels.
    """
    H, W, _ = rgb_img.shape

    # Convert to HSV to get saturation
    hsv = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    sat = hsv[..., 1]

    # Find artifact-prone pixels
    bad_mask = (sat > sat_thresh) & (confidence_map < conf_thresh)

    # Pad the image and confidence map for easy neighborhood processing
    pad = neighborhood // 2
    padded_rgb = np.pad(rgb_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_conf = np.pad(confidence_map, ((pad, pad), (pad, pad)), mode='reflect')

    result = rgb_img.copy()

    # Iterate over bad pixels only
    bad_coords = np.argwhere(bad_mask)
    for y, x in bad_coords:
        y0, y1 = y, y + neighborhood
        x0, x1 = x, x + neighborhood

        patch = padded_rgb[y0:y1, x0:x1, :]
        conf_patch = padded_conf[y0:y1, x0:x1]

        valid_mask = conf_patch >= conf_thresh
        if np.mean(valid_mask) < support_thresh:
            continue  # not enough support nearby

        # Weighted average using confidence
        weights = conf_patch[valid_mask][..., np.newaxis]
        values = patch[valid_mask]
        replacement = np.sum(values * weights, axis=0) / np.sum(weights)
        result[y, x, :] = replacement

    return result









# 4. Training
def train():
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    train_dataset = ColorizationDataset('train')
    val_dataset = ColorizationDataset('val')

    # compute_class_weights(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationNet().to(device)

    # === Load or define your weights (you'll need to fill in how you compute them) ===
    weights_path = "resources/class_weights.npy"  # You need to make this
    weights = np.load(weights_path)               # shape (313,)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    checkpoint_path = "colorization_checkpoint.pth"
    epoch = 0
    global_step = 0
    writer = None
    start_epoch = 0

    if os.path.exists("log_progress.pth"):
        global_step = torch.load("log_progress.pth")['global_step']

    if os.path.exists("colorization_model_interrupt.pth"):
        print("Loading interrupted model...")
        model.load_state_dict(torch.load("colorization_model_interrupt.pth", map_location=device))
        os.remove("colorization_model_interrupt.pth")
    elif os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
    elif os.path.exists("colorization_model.pth"):
        print("Loading saved model...")
        model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
    else:
        print("Training from scratch.")

    writer = SummaryWriter(log_dir="runs/colorization_exp")

    num_epochs = 10
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            progress_bar = tqdm(train_loader, desc="Training", leave=False)

            for batch_idx, (L, ab_bin, filenames) in enumerate(progress_bar):

                L, ab_bin = L.to(device), ab_bin.to(device)

                optimizer.zero_grad()
                output = model(L)
                loss = criterion(output, ab_bin)

                # print("output shape:", output.shape)   # should be (B, 313, H, W)
                # print("ab_bin shape:", ab_bin.shape)   # should be (B, H, W)
                # print("ab_bin dtype:", ab_bin.dtype)   # should be torch.long

                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()

                writer.add_scalar("Loss/train", loss.detach().item(), global_step)
                global_step += 1

                progress_bar.set_postfix(loss=loss.detach().item())

            print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss / len(train_loader):.4f}")

            model.eval()
            output_validation(val_dataset, device, model, epoch, writer, global_step)

            model.train()

    except KeyboardInterrupt:
        output_validation(val_dataset, device, model, epoch, writer, global_step)
        print("\nTraining interrupted. Saving model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, "colorization_checkpoint_interrupt.pth")
        torch.save({'global_step': global_step}, "log_progress.pth")
        print("Model saved as 'colorization_checkpoint_interrupt.pth'. Exiting.")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, "colorization_checkpoint.pth")
    torch.save({'global_step': global_step}, "log_progress.pth")



def output_validation(val_dataset, device, model, epoch, writer, global_step):
    with torch.no_grad():
        rand_index = random.randint(0, len(val_dataset) - 1)
        val_L, _, val_filename = val_dataset[rand_index]
        val_L_tensor = val_L.unsqueeze(0).float().to(device)
        val_output = model(val_L_tensor)
        original_img_path = val_filename
        original_rgb = np.array(Image.open(original_img_path).convert("RGB")) / 255.0
        safe_name = os.path.splitext(os.path.basename(val_filename))[0]
        psnr_val, mse_val = evaluate_and_visualize(val_L_tensor, val_output, original_rgb, safe_name, epoch)
        writer.add_scalar("PSNR/val", psnr_val, global_step)
        writer.add_scalar("MSE/val", mse_val, global_step)

def evaluate_and_visualize(L, ab_logits, original_rgb, filename, epoch, output_dir="val_outputs"):
    pts = np.load("resources/pts_in_hull.npy")
    print("pts range:", pts.min(), pts.max())

    os.makedirs(output_dir, exist_ok=True)
    if not hasattr(evaluate_and_visualize, "pts"):
        evaluate_and_visualize.pts = np.load("resources/pts_in_hull.npy")
    pts = evaluate_and_visualize.pts
    temperature = 0.38  # Lower = sharper, Higher = smoother
    ab_probs = torch.nn.functional.softmax(ab_logits / temperature, dim=1)[0].cpu().numpy()

    confidence = np.max(ab_probs, axis=0)
    # plt.imshow(confidence, cmap='viridis')
    # plt.title("Confidence per pixel")
    # plt.colorbar()
    # plt.show()

    # plt.imshow(np.argmax(ab_probs, axis=0), cmap='nipy_spectral')
    # plt.colorbar()
    # plt.title("Argmax of ab probabilities per pixel")
    # plt.show()

    # === Convert ab probabilities to ab channels ===
    ab_img_raw = np.tensordot(ab_probs.transpose(1, 2, 0), pts, axes=([2], [0]))  # (H, W, 2)
    ab_img_raw = ab_img_raw.transpose(2, 0, 1)  # (2, H, W)

    # === Apply confidence-based blending ===
    confidence = np.max(ab_probs, axis=0)
    blurred_ab = gaussian_filter(ab_img_raw, sigma=(0, 3, 3))  # Smooth spatially
    confidence_map = np.clip(confidence, 0.0, 1.0).reshape(1, *confidence.shape)
    ab_img_smooth = ab_img_raw * confidence_map + blurred_ab * (1 - confidence_map)


    # neutral_ab[0, :, :] = 5   # 'a' channel
    # neutral_ab[1, :, :] = 10  # 'b' channel
    

    confidence = np.clip(confidence, 0.0, 1.0).reshape(1, *confidence.shape)
    ab_img = ab_img_raw * confidence + blurred_ab * (1 - confidence)

    # plt.imshow(confidence[0], cmap='viridis')
    # plt.title("Confidence per pixel")
    # plt.colorbar()
    # plt.show()


    # plt.imshow(ab_img[0], cmap='RdBu')
    # plt.title("a channel")
    # plt.colorbar()
    # plt.show()

    # plt.imshow(ab_img[1], cmap='RdBu')
    # plt.title("b channel")
    # plt.colorbar()
    # plt.show()



    print("ab_img range:", ab_img.min(), ab_img.max())
    print("max logit:", ab_logits.max().detach().item(), "min logit:", ab_logits.min().detach().item())


    L_img = L[0].cpu().numpy()[0] * 100
    # Create both LAB stacks
    lab_original = np.concatenate([L_img[np.newaxis, :, :], ab_img_raw], axis=0).transpose(1, 2, 0)
    lab_smoothed = np.concatenate([L_img[np.newaxis, :, :], ab_img_smooth], axis=0).transpose(1, 2, 0)

    # Convert both to RGB
    pred_rgb_original = lab2rgb(lab_original)
    pred_rgb_smooth = lab2rgb(lab_smoothed)

    confidence_map_hwc = confidence_map[0]  # (H, W), from shape (1, H, W)
    pred_rgb_smooth = suppress_color_splotches3(pred_rgb_smooth, confidence_map_hwc)

    

    original_rgb_resized = np.array(Image.fromarray((original_rgb * 255).astype("uint8")).resize((256, 256))) / 255.0
    mse_raw = np.mean((pred_rgb_original - original_rgb_resized) ** 2)
    psnr_raw = psnr(original_rgb_resized, pred_rgb_original, data_range=1.0)

    mse_smooth = np.mean((pred_rgb_smooth - original_rgb_resized) ** 2)
    psnr_smooth = psnr(original_rgb_resized, pred_rgb_smooth, data_range=1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(L_img, cmap="gray")
    axes[0].set_title("Grayscale (L)")
    axes[1].imshow(pred_rgb_original)
    axes[1].set_title("Model Output (Raw)")
    axes[2].imshow(pred_rgb_smooth)
    axes[2].set_title("Model Output (Smoothed)")
    axes[3].imshow(original_rgb_resized)
    axes[3].set_title("Original Color")
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"MSE raw: {mse_raw:.4f} | PSNR raw: {psnr_raw:.2f} dB\n"+
                 f"MSE smooth: {mse_smooth:.4f} | PSNR: {psnr_smooth:.2f} dB", fontsize=12)
    out_path = os.path.join(output_dir, f"{filename}_epoch{epoch}_comparison.png")
    plt.savefig(out_path)
    plt.close()
    return psnr_smooth, mse_smooth

# One-time precomputation script (you could add this above train())
def compute_class_weights(dataset):
    counts = np.zeros(313)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    print("Computing class weights...")
    for _, q_ab, _ in tqdm(loader, total=len(dataset), desc="Scanning dataset"):
        values, counts_i = np.unique(q_ab.numpy().flatten(), return_counts=True)
        counts[values] += counts_i

    weights = 1 / (counts + 1e-5)
    weights = weights / weights.sum()
    weights = weights ** 0.5

    np.save("resources/class_weights.npy", weights)
    print("Saved weights to resources/class_weights.npy")


# 5. Run it!
if __name__ == '__main__':
    train()