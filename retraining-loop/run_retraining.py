# run_retraining.py
from prepare_data import extract_new_labeled_samples
from retrain_model import retrain_and_save

if __name__ == "__main__":
    new_data = extract_new_labeled_samples()
    if len(new_data) > 10:
        print("Starting retraining...")
        retrain_and_save()
    else:
        print("Not enough new samples to retrain.")
