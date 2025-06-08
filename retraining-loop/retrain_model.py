# retrain_model.py
import lightgbm as lgb
import os
import time
from merge_and_weight import combine_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def retrain_and_save():
    X, y, weights = combine_datasets()
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        'objective': 'multiclass',
        'num_class': len(set(y)),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    test_data = lgb.Dataset(X_test, label=y_test, weight=w_test, reference=train_data)

    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)
    
    version = time.strftime("%Y%m%d-%H%M%S")
    out_dir = f"models/model_{version}"
    os.makedirs(out_dir, exist_ok=True)

    model.save_model(f"{out_dir}/lightgbm_retrained.txt")
    print(f"✅ Model saved to {out_dir}/lightgbm_retrained.txt")

    preds = model.predict(X_test).argmax(axis=1)
    print("\n📊 Retrained Model Report:\n", classification_report(y_test, preds))
