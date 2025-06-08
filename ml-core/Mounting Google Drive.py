# ✅ 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ✅ Step 2: Define the folder path you want to open in File Explorer
# Replace 'your_folder_name' with your actual folder name
folder_path = '/content/drive/MyDrive/TrustFusion'  # <-- Change this

# ✅ Step 3: Open the folder in the File Explorer (by navigating to it)
import os
from google.colab import files

if os.path.exists(folder_path):
    print(f"📁 Directory found: {folder_path}")
    # Change current working directory to the folder
    os.chdir(folder_path)
    print("📂 Current working directory changed to:", os.getcwd())
else:
    print("❌ Folder not found. Please check the path.")