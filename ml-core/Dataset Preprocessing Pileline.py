# 📦 Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder
from google.colab import drive

# 📂 Mount Google Drive
drive.mount('/content/drive')

# 📁 Paths
RAW_PATH = "/content/drive/MyDrive/TrustFusion/raw/raw-dataset.csv"
FINAL_PATH = "/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv"

# 📄 Load Raw Dataset
df = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])
df.sort_values('timestamp', inplace=True)

# 🔧 Scalers and Encoders
scalers = {}
encoders = {}

# 🔄 Normalize Numeric Columns
def normalize_column(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    if df[column_name].isna().any():
        df[column_name].fillna(df[column_name].median(), inplace=True)
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    scalers[column_name] = scaler
    return df

# 🔡 Encode Categorical Columns
def encode_column(df, column_name):
    encoder = LabelEncoder()
    df[column_name] = df[column_name].astype(str)
    df[column_name] = encoder.fit_transform(df[column_name])
    encoders[column_name] = encoder
    return df

# 🧪 Phase 1: Normalize and Encode Raw Data
def preprocess_trustfusion(df):
    numeric_cols = [
        'usage_frequency', 'violation_count', 'peer_rating', 'complaint_count',
        'account_age_days', 'total_interactions', 'rolling_success',
        'violation_trend', 'time_since_last_access', 'denial_ratio'
    ]
    categorical_cols = [
        'user_id', 'device_id', 'role', 'location',
        'day_of_week', 'term', 'user_trust_tier'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df = normalize_column(df, col)

    for col in categorical_cols:
        if col in df.columns:
            df = encode_column(df, col)

    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].map({-1: 0, 0: 1, 1: 2})

    df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9

    if 'feedback_text' in df.columns:
        df.drop(columns=['feedback_text'], inplace=True)

    return df

# 🚀 Apply Preprocessing
df = preprocess_trustfusion(df)

# 🧪 Phase 2: Feature Engineering and Scaling
df.drop(columns=["user_id", "device_id", "timestamp", "timestamp_unix"], inplace=True)

# ➿ Cyclical Time Features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df.drop(columns=["hour", "day_of_week"], inplace=True)

# 🔢 Encode Additional Categorical Columns
cat_cols = ["role", "location", "admin_flag", "term"]
df[cat_cols] = df[cat_cols].astype("category")
encoder = OrdinalEncoder()
df[cat_cols] = encoder.fit_transform(df[cat_cols])

# 🚫 Drop Leakage Columns
leak_cols = ["rolling_success", "violation_trend", "denial_ratio"]
df.drop(columns=[col for col in leak_cols if col in df.columns], inplace=True)

# 🎯 Target Columns
target_cols = ["success", "user_trust_tier", "anomaly_flag", "sentiment"]
X = df.drop(columns=target_cols)
y = df[target_cols]

# 🔄 Standardize Features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 🔗 Combine Features and Targets
df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

# 🧼 Convert to Float32
df_final = df_final.astype(np.float32)

# ✅ Sanity Check
non_numeric = df_final.select_dtypes(exclude=['number'])
if not non_numeric.empty:
    raise ValueError(f"❌ Non-numeric columns found: {non_numeric.columns.tolist()}")
else:
    print("✅ All columns are numeric float32. Ready for ML pipeline.")

# 💾 Save Final Preprocessed Dataset
df_final.to_csv(FINAL_PATH, index=False)
print(f"\n✅ Preprocessing complete. Final dataset saved to:\n{FINAL_PATH}")
