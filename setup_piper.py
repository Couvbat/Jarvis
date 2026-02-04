"""Setup script for installing Piper TTS."""

import os
import sys
import platform
import urllib.request
import tarfile
import zipfile
from pathlib import Path


def download_file(url: str, dest: Path):
    """Download a file with progress."""
    print(f"Downloading {url}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        print(f"\rProgress: {percent:.1f}%", end="")
    
    urllib.request.urlretrieve(url, dest, report_progress)
    print()  # New line after progress


def setup_piper():
    """Download and set up Piper TTS."""
    piper_dir = Path("piper")
    piper_dir.mkdir(exist_ok=True)
    
    # Detect system
    system = platform.system()
    machine = platform.machine()
    
    print(f"Detected system: {system} {machine}")
    
    # Determine download URL
    if system == "Linux":
        if machine == "x86_64":
            url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz"
            archive_name = "piper_amd64.tar.gz"
        elif machine == "aarch64":
            url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz"
            archive_name = "piper_arm64.tar.gz"
        else:
            print(f"Unsupported architecture: {machine}")
            return False
    else:
        print(f"Unsupported system: {system}")
        print("Please install Piper manually from: https://github.com/rhasspy/piper")
        return False
    
    # Download Piper
    archive_path = piper_dir / archive_name
    if not archive_path.exists():
        download_file(url, archive_path)
    
    # Extract
    print("Extracting Piper...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(piper_dir)
    
    # Make executable
    piper_binary = piper_dir / "piper" / "piper"
    if piper_binary.exists():
        piper_binary.chmod(0o755)
        print(f"Piper installed to: {piper_binary}")
    
    # Download a default model
    print("\nDownloading default voice model (en_US-lessac-medium)...")
    models_dir = piper_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    
    model_path = models_dir / "en_US-lessac-medium.onnx"
    config_path = models_dir / "en_US-lessac-medium.onnx.json"
    
    if not model_path.exists():
        download_file(model_url, model_path)
    
    if not config_path.exists():
        download_file(config_url, config_path)
    
    print("\nPiper setup complete!")
    print(f"Binary: {piper_binary}")
    print(f"Model: {model_path}")
    
    return True


if __name__ == "__main__":
    print("="*50)
    print("Piper TTS Setup")
    print("="*50)
    print()
    
    if setup_piper():
        print("\n✓ Setup successful!")
        print("\nNext steps:")
        print("1. Install Python dependencies: pip install -r requirements.txt")
        print("2. Set up Ollama: https://ollama.ai/download")
        print("3. Copy .env.example to .env and configure")
        print("4. Run: python main.py")
    else:
        print("\n✗ Setup failed")
        sys.exit(1)
