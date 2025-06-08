import pandas as pd
import os

# 📂 Define the path to the preprocessed dataset
PREPROCESSED_PATH = "/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv"

# 📄 Load the preprocessed dataset
df_preprocessed = pd.read_csv(PREPROCESSED_PATH)

# ✅ Confirm the load
print(f"✅ Preprocessed dataset loaded successfully with shape: {df_preprocessed.shape}")
df_preprocessed.head()
