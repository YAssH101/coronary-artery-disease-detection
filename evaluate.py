import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocess import get_patients, patient_to_slices, MRIDataset, val_transform
from model_arch import build_model
from train import full_evaluation

DATASET_PATH = r"C:\Users\yycha\Downloads\dataset"
MODEL_PATH   = "model/cad_model.pth"
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader():
    patient_dirs, patient_labels = get_patients(DATASET_PATH)

    normal_patients = [(d, l) for d, l in zip(patient_dirs, patient_labels) if l == 0]
    sick_patients   = [(d, l) for d, l in zip(patient_dirs, patient_labels) if l == 1]

    rng = random.Random(42)
    rng.shuffle(normal_patients)
    rng.shuffle(sick_patients)

    N_VAL = N_TEST = 2
    test_patients = (normal_patients[N_VAL:N_VAL + N_TEST] +
                     sick_patients[N_VAL:N_VAL + N_TEST])

    print(f"Test patients ({len(test_patients)}):")
    for d, l in test_patients:
        print(f"  {'Normal' if l == 0 else 'Sick  '}  {os.path.basename(d)}")

    test_dirs, test_lbls = zip(*test_patients)
    X_test, y_test = patient_to_slices(test_dirs, test_lbls)
    print(f"Test slices: {len(X_test)}  (Normal={y_test.count(0)}  Sick={y_test.count(1)})")

    return DataLoader(
        MRIDataset(X_test, y_test, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Loading model from {MODEL_PATH} ...")
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    test_loader = get_test_loader()
    metrics = full_evaluation(model, test_loader, DEVICE, save_dir="static")

    print("\nPlots saved to static/")
    print("  confusion_matrix.png")
    print("  roc_pr_curves.png")
