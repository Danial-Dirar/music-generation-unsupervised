from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    PLOTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from src.models.transformer import MusicTransformer, transformer_loss


class PianoRollDataset(Dataset):
    def __init__(self, npy_path: str | Path):
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.npy_path}")

        self.data = np.load(self.npy_path, mmap_mode="r")

        if self.data.ndim != 3:
            raise ValueError(f"Expected [N, T, F], got {self.data.shape}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.array(self.data[idx], dtype=np.float32)
        return torch.from_numpy(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input_dim_from_npy(npy_path: str | Path) -> int:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, T, F], got {arr.shape}")
    return int(arr.shape[2])


def maybe_subset_dataset(
    dataset: Dataset,
    max_samples: Optional[int] = None,
) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = list(range(max_samples))
    return Subset(dataset, indices)


def get_data_loaders(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PianoRollDataset(train_path)
    val_dataset = PianoRollDataset(val_path)

    train_dataset = maybe_subset_dataset(train_dataset, max_train_samples)
    val_dataset = maybe_subset_dataset(val_dataset, max_val_samples)

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


def build_model(input_dim: int) -> MusicTransformer:
    return MusicTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=512,
    )


def shift_for_next_step_prediction(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[:, :-1, :], batch[:, 1:, :]


def train_one_epoch(
    model: MusicTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        x_in, x_target = shift_for_next_step_prediction(batch)

        optimizer.zero_grad()
        logits = model(x_in)
        loss = transformer_loss(logits, x_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def validate_one_epoch(
    model: MusicTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        x_in, x_target = shift_for_next_step_prediction(batch)
        logits = model(x_in)
        loss = transformer_loss(logits, x_target)

        batch_size = batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def compute_perplexity(loss_value: float) -> float:
    return float(np.exp(loss_value))


def save_history_csv(history: Dict[str, list], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_perplexity", "learning_rate"])
        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["val_loss"][i],
                    history["val_perplexity"][i],
                    history["learning_rate"][i],
                ]
            )

    return output_path


def save_loss_plot(history: Dict[str, list], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss plot.")
        return output_path

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogits Loss")
    plt.title("Transformer Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def train_transformer(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    num_workers: int = 0,
    patience: int = 3,
    min_delta: float = 1e-4,
    grad_clip: float = 1.0,
    max_train_samples: Optional[int] = 20000,
    max_val_samples: Optional[int] = 3000,
) -> Dict[str, object]:
    device = get_device()
    print(f"Using device: {device}")

    input_dim = get_input_dim_from_npy(train_path)
    print(f"Input dim: {input_dim}")

    train_loader, val_loader = get_data_loaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    model = build_model(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
    )

    checkpoint_dir = Path(PROJECT_ROOT) / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_checkpoint_path = checkpoint_dir / "transformer_best.pt"
    last_checkpoint_path = checkpoint_dir / "transformer_last.pt"

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        scheduler.step(val_loss)
        val_perplexity = compute_perplexity(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_perplexity)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_perplexity={val_perplexity:.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": input_dim,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
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
                    "input_dim": input_dim,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
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

    history_csv_path = Path(PLOTS_DIR) / "transformer_training_history.csv"
    history_plot_path = Path(PLOTS_DIR) / "transformer_loss_curve.png"

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
    }


def main() -> None:
    train_path = Path(PROCESSED_DIR) / "train_sequences.npy"
    val_path = Path(PROCESSED_DIR) / "validation_sequences.npy"

    results = train_transformer(
        train_path=train_path,
        val_path=val_path,
        batch_size=8,
        epochs=20,
        lr=LR,
        num_workers=2,
        patience=5,
        min_delta=1e-4,
        grad_clip=1.0,
        max_train_samples=None,
        max_val_samples=None,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {results['best_checkpoint_path']}")
    print(f"Last checkpoint: {results['last_checkpoint_path']}")
    print(f"History CSV:     {results['history_csv_path']}")
    print(f"Loss plot:       {results['history_plot_path']}")
    print(f"Best val loss:   {results['best_val_loss']:.6f}")
    print(f"Device used:     {results['device']}")


if __name__ == "__main__":
    main()