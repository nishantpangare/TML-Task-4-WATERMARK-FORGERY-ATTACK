import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import torch
import torchvision
import omegaconf
import subprocess


os.chdir("/home/atml_team013/tml4")


ZIP_FILE = "Dataset.zip"  # Path to the downloaded dataset zip
DATASET_DIR = Path(".")  # Unzipped folder
DELTA_DIR = Path("deltas_extracted3")
DELTA_DIR.mkdir(exist_ok=True)


wmforger_dir = Path("videoseal/wmforger")
model_checkpoint = wmforger_dir / "convnext_pref_model.pth"
sys.path.insert(0,str(wmforger_dir))
from wmforger.models import build_extractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


n_extraction = 10
steps = 300
lr = 0.05

# 1. UNZIP DATASET
if not (DATASET_DIR / "watermarked_sources").exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")

    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")


CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]


def get_artifact_discriminator(model_checkpoint, DEVICE):
    model_type = "convnext_tiny"
    state_dict = torch.load(model_checkpoint,weights_only=True,map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load(wmforger_dir/ "configs/extractor.yaml")[model_type]

    model = build_extractor(model_type,extractor_params,img_size=256,nbits=0)
    model.load_state_dict(state_dict)
    model = model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad=False
    return model

print("Pretrained Preference Model:")
model_preference = get_artifact_discriminator(model_checkpoint,DEVICE)

transform_image = torchvision.transforms.Compose([
    lambda x: x.convert("RGB"),
    torchvision.transforms.Resize((768, 768)),
    torchvision.transforms.ToTensor(),
    lambda x: x.view(1, 3, 768, 768),
])


def optimize(img: Image.Image, model, device, num_steps=steps, lr=lr):
    img = transform_image(img).to(device)
    param = torch.nn.Parameter(torch.zeros_like(img)).to(device)

    optim = torch.optim.SGD([param], lr=lr)
    for _ in range(num_steps):
        optim.zero_grad()
        loss = -model((img + param).clip(0, 1)).mean()
        loss.backward()
        optim.step()
    
    return (img + param).clip(0, 1).detach().cpu()


def get_watermark(img: Image.Image, optimized_img: torch.Tensor):
    optimized_img = optimized_img.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).squeeze(0).numpy()
    optimized_img = Image.fromarray(optimized_img).resize(img.size, Image.BILINEAR)

    watermark = np.array(img).astype(np.float32) - np.array(optimized_img).astype(np.float32)
    return optimized_img, watermark

def extract_watermark(source,device):
    img = Image.open(source).convert("RGB")
    optimized = optimize(img,model_preference,device=device)
    _, watermark = get_watermark(img,optimized)
    return watermark

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Extracting watermark for {source_wm}")

    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = list(source_dir.glob("*.png"))[:n_extraction]

    if not source_images:
        print(f"  [Warning] No source images found in {source_dir}")
        continue

    print(f"  Extracting watermark from {len(source_images)} images.")
    watermarks = []
    for source_path in source_images:
        w = extract_watermark(source_path,DEVICE)
        watermarks.append(w)
    delta_hat = np.median(watermarks,axis=0)

    np.save(DELTA_DIR / f"{source_wm}_delta.npy", delta_hat)
    print(f"Saved {source_wm}_delta.npy")

print("\nExtraction Complete. Saved All Deltas to the Extracted_delta folder")