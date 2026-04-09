from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LATENT_DIM,
    LR,
    NUM_LAYERS,
    PLOTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SEQ_LEN,
)
from src.models.autoencoder import LSTMAutoencoder


class PianoRollDataset(Dataset):
    """
    Memory-efficient dataset for saved .npy piano-roll sequences.

    Expected array shape:
        [N, seq_len, input_dim]
    """

    def __init__(self, npy_path: str | Path):
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.npy_path}")

        self.data = np.load(self.npy_path, mmap_mode="r")

        if self.data.ndim != 3:
            raise ValueError(
                f"Expected 3D array [N, seq_len, input_dim], got {self.data.shape}"
            )

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.array(self.data[idx], dtype=np.float32)
        return torch.from_numpy(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PianoRollDataset(train_path)
    val_dataset = PianoRollDataset(val_path)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def get_input_dim_from_npy(npy_path: str | Path) -> int:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [N, T, F], got {arr.shape}")
    return int(arr.shape[2])


def estimate_positive_weight(
    train_path: str | Path,
    max_samples: int = 2048,
    seed: int = 42,
) -> float:
    """
    Estimate class imbalance from training data.

    pos_weight ~= negatives / positives

    Returns a scalar weight for positive cells.
    """
    arr = np.load(train_path, mmap_mode="r")
    total_samples = int(arr.shape[0])

    if total_samples == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, total_samples)
    indices = rng.choice(total_samples, size=sample_count, replace=False)

    positives = 0.0
    total = 0

    chunk_size = 128
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        batch_idx = indices[start:end]
        batch = np.array(arr[batch_idx], dtype=np.float32)

        positives += float(batch.sum())
        total += int(batch.size)

    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)

    pos_weight = negatives / positives
    pos_weight = float(np.clip(pos_weight, 1.0, 5.0))
    return pos_weight


def weighted_bce_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    pos_weight: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Weighted BCE on probabilities.

    Since current model outputs sigmoid probabilities already,
    we use weighted BCE directly instead of BCEWithLogitsLoss.
    """
    x_hat = torch.clamp(x_hat, eps, 1.0 - eps)

    loss = -(
        pos_weight * x * torch.log(x_hat)
        + (1.0 - x) * torch.log(1.0 - x_hat)
    )

    return loss.mean()


def train_one_epoch(
    model: LSTMAutoencoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    pos_weight: float,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        x_hat, _ = model(batch)
        loss = weighted_bce_loss(x_hat, batch, pos_weight=pos_weight)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def validate_one_epoch(
    model: LSTMAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: float,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        x_hat, _ = model(batch)
        loss = weighted_bce_loss(x_hat, batch, pos_weight=pos_weight)

        batch_size = batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def save_history_csv(history: Dict[str, list], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

        for i, (tr, vl, lr_val) in enumerate(
            zip(
                history["train_loss"],
                history["val_loss"],
                history["learning_rate"],
            ),
            start=1,
        ):
            writer.writerow([i, tr, vl, lr_val])

    return output_path


def save_loss_plot(history: Dict[str, list], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss plot.")
        return output_path

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.title("LSTM Autoencoder Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def build_model(input_dim: int) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        dropout=0.1,
    )
    return model


def train_autoencoder(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    num_workers: int = 0,
    patience: int = 5,
    min_delta: float = 1e-4,
    grad_clip: float = 1.0,
) -> Dict[str, object]:
    device = get_device()
    print(f"Using device: {device}")

    input_dim = get_input_dim_from_npy(train_path)
    print(f"Input dim: {input_dim}")

    pos_weight = estimate_positive_weight(train_path)
    print(f"Estimated positive class weight: {pos_weight:.4f}")

    train_loader, val_loader = get_data_loaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    checkpoint_dir = Path(PROJECT_ROOT) / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_checkpoint_path = checkpoint_dir / "ae_best.pt"
    last_checkpoint_path = checkpoint_dir / "ae_last.pt"

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            grad_clip=grad_clip,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            pos_weight=pos_weight,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "input_dim": input_dim,
                "seq_len": SEQ_LEN,
                "pos_weight": pos_weight,
            },
            last_checkpoint_path,
        )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "input_dim": input_dim,
                    "seq_len": SEQ_LEN,
                    "pos_weight": pos_weight,
                },
                best_checkpoint_path,
            )
            print(f"  -> Saved new best model to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No significant improvement ({epochs_without_improvement}/{patience})")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    history_csv_path = Path(PLOTS_DIR) / "ae_training_history.csv"
    history_plot_path = Path(PLOTS_DIR) / "ae_loss_curve.png"

    save_history_csv(history, history_csv_path)
    save_loss_plot(history, history_plot_path)

    return {
        "history": history,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "history_csv_path": history_csv_path,
        "history_plot_path": history_plot_path,
        "best_val_loss": best_val_loss,
        "device": str(device),
        "pos_weight": pos_weight,
    }


def main() -> None:
    train_path = Path(PROCESSED_DIR) / "train_sequences.npy"
    val_path = Path(PROCESSED_DIR) / "validation_sequences.npy"

    results = train_autoencoder(
        train_path=train_path,
        val_path=val_path,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        num_workers=0,
        patience=5,
        min_delta=1e-4,
        grad_clip=1.0,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {results['best_checkpoint_path']}")
    print(f"Last checkpoint: {results['last_checkpoint_path']}")
    print(f"History CSV:     {results['history_csv_path']}")
    print(f"Loss plot:       {results['history_plot_path']}")
    print(f"Best val loss:   {results['best_val_loss']:.6f}")
    print(f"Pos weight:      {results['pos_weight']:.4f}")
    print(f"Device used:     {results['device']}")


if __name__ == "__main__":
    main()