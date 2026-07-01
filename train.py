import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from model_arch import build_model, unfreeze_top_layers
from preprocess import (
    MRIDataset, get_patients, patient_to_slices,
    train_transform, val_transform,
)

DATASET_PATH      = r"C:\Users\yycha\Downloads\dataset"
TEST_DATASET_PATH = r"C:\Users\yycha\Downloads\test_data"
MODEL_SAVE_PATH   = "model/cad_model.pth"
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion, device, class_weights):
    model.train()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []

    w = torch.tensor(class_weights, dtype=torch.float32).to(device)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        probs = model(imgs).squeeze(1)

        weights = torch.where(labels == 1, w[1], w[0])
        loss = (criterion(probs, labels) * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            probs = model(imgs).squeeze(1)
            loss  = criterion(probs, labels).mean()

            total_loss += loss.item() * imgs.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


def run_phase(model, train_loader, val_loader, optimizer, scheduler,
              criterion, epochs, device, class_weights, model_path):
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_auc": [], "val_auc": []}
    patience_counter = 0
    patience = 5

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, class_weights
        )
        vl_loss, vl_acc, vl_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        print(
            f"Epoch {epoch:02d}  "
            f"Loss {tr_loss:.4f}/{vl_loss:.4f}  "
            f"Acc {tr_acc:.4f}/{vl_acc:.4f}  "
            f"AUC {tr_auc:.4f}/{vl_auc:.4f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> saved best model (val_loss={vl_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return history


def full_evaluation(model, loader, device, save_dir="static"):
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score,
    )
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = model(imgs).squeeze(1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    print(f"\n=== Full Evaluation ===")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"ROC AUC:              {roc_auc:.4f}")
    print(f"Avg Precision (PR):   {ap:.4f}")
    print(f"Confusion Matrix  →  TN={tn}  FP={fp}  |  FN={fn}  TP={tp}")

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Sick"])
    ax.set_yticklabels(["Normal", "Sick"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=100)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CAD Detection — Model Evaluation")
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(rec_c, prec_c, lw=2, label=f"AP = {ap:.4f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_pr_curves.png"), dpi=100)
    plt.show()

    return {
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "roc_auc": roc_auc, "ap": ap,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def plot_history(h1, h2, save_path="static/training_history.png"):
    acc      = h1["train_acc"]  + h2["train_acc"]
    val_acc  = h1["val_acc"]    + h2["val_acc"]
    loss     = h1["train_loss"] + h2["train_loss"]
    val_loss = h1["val_loss"]   + h2["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CAD Detection Model — Training History")
    axes[0].plot(acc, label="Train"); axes[0].plot(val_acc, label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(loss, label="Train"); axes[1].plot(val_loss, label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    print(f"Plot saved -> {save_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    print("Scanning dataset (patient-level split)...")
    patient_dirs, patient_labels = get_patients(DATASET_PATH)

    # Separate by class so we can stratify the split manually
    normal_patients = [(d, l) for d, l in zip(patient_dirs, patient_labels) if l == 0]
    sick_patients   = [(d, l) for d, l in zip(patient_dirs, patient_labels) if l == 1]

    # Load unseen test patients from the physically separated test_data folder
    test_dirs, test_lbls = get_patients(TEST_DATASET_PATH)
    test_patients = list(zip(test_dirs, test_lbls))

    # Shuffle within each class with a fixed seed for train/val split
    rng = random.Random(42)
    rng.shuffle(normal_patients)
    rng.shuffle(sick_patients)

    # Normal (14): 12 train | 2 val
    # Sick   (12): 10 train | 2 val
    N_VAL = 2

    val_patients   = normal_patients[:N_VAL] + sick_patients[:N_VAL]
    train_patients = normal_patients[N_VAL:] + sick_patients[N_VAL:]

    print(f"Patients — Train: {len(train_patients)}  Val: {len(val_patients)}  Test: {len(test_patients)}")

    X_train, y_train = patient_to_slices(*zip(*train_patients))
    X_val,   y_val   = patient_to_slices(*zip(*val_patients))
    X_test,  y_test  = patient_to_slices(*zip(*test_patients))  # from test_data/

    print(f"Slices  — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Train label balance: Normal={y_train.count(0)}  Sick={y_train.count(1)}")

    train_ds = MRIDataset(X_train, y_train, transform=train_transform)
    val_ds   = MRIDataset(X_val,   y_val,   transform=val_transform)
    test_ds  = MRIDataset(X_test,  y_test,  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    n_normal = y_train.count(0)
    n_sick   = y_train.count(1)
    total    = len(y_train)
    class_weights = [total / (2 * n_normal), total / (2 * n_sick)]
    print(f"Class weights: Normal={class_weights[0]:.3f}  Sick={class_weights[1]:.3f}")

    os.makedirs("model", exist_ok=True)
    model     = build_model().to(DEVICE)
    criterion = nn.BCELoss(reduction="none")

    # Phase 1 — head only
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    print("\n=== Phase 1: Training classification head ===")
    h1 = run_phase(model, train_loader, val_loader, optimizer, scheduler,
                   criterion, EPOCHS_PHASE1, DEVICE, class_weights, MODEL_SAVE_PATH)

    # Phase 2 — fine-tune top 3 EfficientNet blocks
    unfreeze_top_layers(model, n_blocks=3)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    print("\n=== Phase 2: Fine-tuning top EfficientNet blocks ===")
    h2 = run_phase(model, train_loader, val_loader, optimizer, scheduler,
                   criterion, EPOCHS_PHASE2, DEVICE, class_weights, MODEL_SAVE_PATH)

    # Test evaluation
    test_loss, test_acc, test_auc = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"\n=== Test Results ===")
    print(f"Loss: {test_loss:.4f}  Accuracy: {test_acc:.4f}  AUC: {test_auc:.4f}")

    full_evaluation(model, test_loader, DEVICE, save_dir="static")
    plot_history(h1, h2)
