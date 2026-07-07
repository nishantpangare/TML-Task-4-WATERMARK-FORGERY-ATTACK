import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import torch


os.chdir("/home/atml_team013/tml4")


ZIP_FILE = "Dataset.zip"  # Path to the downloaded dataset zip
DATASET_DIR = Path(".")  # Unzipped folder
TEMP_OUT_DIR = Path("submission_temp_wmf19")  # Temporary folder for forged images
FILE_PATH = "submission_wmf19.zip"  # Final file to upload
DELTA_DIR = Path("deltas_extracted4")

# Leaderboard submission
BASE_URL  = "http://35.192.205.84:80"
API_KEY  = "4ffd0e8bba3b9e8977c5c163c465fd4f"  # REPLACE WITH YOUR API KEY
TASK_ID   = "22-forging-task"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
TEMP_OUT_DIR.mkdir(exist_ok=True)

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



alpha_values = {"WM_1":1.5,"WM_2":1.5,"WM_3":1.5,"WM_4":1.5,"WM_5":1.5,"WM_6":1.5,"WM_7":1.5,"WM_8":1.5}

reduce_residual = True
allow_rate = 70

def reduce_residual(delta_hat,allow=70):
    mag = np.abs(delta_hat)
    threshold = np.percentile(mag,allow)
    mask = mag >= threshold
    return delta_hat * mask


total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    alpha = alpha_values[source_wm]
    print(f"Processing {source_wm} dataset -> Forging onto images {target_start}.png to {target_stop}.png ...")

    delta_hat = np.load(DELTA_DIR / f"{source_wm}_delta.npy")

    if reduce_residual:
        delta_hat = reduce_residual(delta_hat,allow=allow_rate)


    target_dir = DATASET_DIR / "clean_targets"
    target_paths = [target_dir / f"{n}.png" for n in range(target_start,target_stop + 1)]

    for target_path in target_paths:
        target_img = Image.open(target_path).convert("RGB")
        target_arr = np.array(target_img).astype(np.float32)

        if delta_hat.shape[:2] != target_arr.shape[:2]:
            delta_pil = Image.fromarray(np.clip(delta_hat + 128,0,255).astype(np.uint8)).resize((target_arr.shape[1], target_arr.shape[0]), Image.BILINEAR)
            delta_resized = np.array(delta_pil).astype(np.float32) - 128.0
        else:
            delta_resized = delta_hat


         # Clip values to valid pixel range [0, 255] and convert to uint8
        forged_img = np.clip(target_arr + alpha * delta_resized, 0, 255).astype(np.uint8)

        # Save to our temporary flat directory using the exact original filename (e.g., "104.png")
        out_path = TEMP_OUT_DIR / target_path.name
        Image.fromarray(forged_img).save(out_path)
        total_processed += 1


print(f"\nSuccessfully forged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200 images, but processed {total_processed}. Your submission may be rejected!")


# 3. PACKAGE INTO FLAT ZIP FILE
print(f"Packaging images into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in TEMP_OUT_DIR.glob("*.png"):
        zipf.write(img_path, arcname=img_path.name)

print(f"Saved submission file to {FILE_PATH}")