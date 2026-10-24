"""
src/stage8/download_and_eval.py - Resilient downloader for MedCPT Cross-Encoder weights followed by automatic zero-shot evaluation.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

TARGET_DIR = Path(r"C:\Users\mmahd\.cache\huggingface\hub\models--ncbi--MedCPT-Cross-Encoder\snapshots\71caf65d4927987813984f54c284405a13fcca49")
TARGET_FILE = TARGET_DIR / "pytorch_model.bin"
URL = "https://huggingface.co/ncbi/MedCPT-Cross-Encoder/resolve/main/pytorch_model.bin"

def download_file():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    existing_bytes = TARGET_FILE.stat().st_size if TARGET_FILE.exists() else 0
    print(f"Target file: {TARGET_FILE}")
    print(f"Already downloaded: {existing_bytes / (1024*1024):.2f} MB")

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
            if existing_bytes > 0 and response.getcode() == 206:
                total_expected = existing_bytes + total_size
            else:
                total_expected = total_size
                existing_bytes = 0

            print(f"Downloading {total_expected / (1024*1024):.2f} MB total...")
            mode = "ab" if existing_bytes > 0 else "wb"
            chunk_size = 1024 * 1024  # 1 MB
            downloaded = existing_bytes
            t0 = time.time()
            last_log = t0

            with open(TARGET_FILE, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log > 10.0:
                        speed = (downloaded - existing_bytes) / (now - t0 + 1e-6) / (1024 * 1024)
                        pct = (downloaded / total_expected * 100) if total_expected > 0 else 0
                        print(f"Progress: {downloaded / (1024*1024):.1f}/{total_expected / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s", flush=True)
                        last_log = now

            print("Download completed successfully!")
            return True
    except Exception as e:
        print(f"Download interrupted or error: {e}")
        return False

if __name__ == "__main__":
    success = False
    for attempt in range(5):
        print(f"\n--- Download attempt {attempt + 1}/5 ---")
        if download_file():
            success = True
            break
        time.sleep(3)

    if success or (TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 400 * 1024 * 1024):
        print("\nWeights ready! Launching Zero-Shot evaluation script...")
        import subprocess
        base_dir = Path(__file__).resolve().parent.parent.parent
        eval_script = base_dir / "src" / "stage8" / "eval_stage8_zero_shot.py"
        subprocess.run([sys.executable, "-u", str(eval_script)], check=True)
    else:
        print("Failed to complete weight download after attempts.")
