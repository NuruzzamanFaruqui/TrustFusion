# merge_and_weight.py
import pandas as pd
import numpy as np

PREV_DATA = "../preprocessed/preprocessed-dataset.csv"
RL_DATA = "../LightGBM-RL/rl-generated-data.csv"
NEW_DATA = "processed/new_samples.csv"

def combine_datasets():
    df_original = pd.read_csv(PREV_DATA)
    df_rl = pd.read_csv(RL_DATA)
    df_new = pd.read_csv(NEW_DATA)

    y_orig = df_original["user_trust_tier"].astype(int)
    y_new = df_new["user_trust_tier"].astype(int)

    weights = np.concatenate([
        np.ones(len(df_original)),
        np.full(len(df_rl), 3.0),
        np.full(len(df_new), 5.0)
    ])

    combined = pd.concat([df_original, df_rl, df_new], ignore_index=True)
    y_combined = pd.concat([y_orig, y_orig.sample(len(df_rl)), y_new], ignore_index=True)

    return combined.drop(columns=["user_trust_tier"]), y_combined, weights
