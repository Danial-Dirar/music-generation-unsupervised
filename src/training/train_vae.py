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
from src.models.vae import MusicVAE, vae_loss


class PianoRollDataset(Dataset):
    """
    Memory-efficient dataset for saved .npy piano-roll sequences.
    Expected shape:
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


def get_input_dim_from_npy(npy_path: str | Path) -> int:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [N, T, F], got {arr.shape}")
    return int(arr.shape[2])


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


def build_model(input_dim: int) -> MusicVAE:
    return MusicVAE(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        dropout=0.1,
    )


def kl_beta_schedule(
    epoch: int,
    total_epochs: int,
    beta_start: float = 0.0,
    beta_end: float = 1.0,
    warmup_fraction: float = 0.3,
) -> float:
    """
    Linearly warm up beta for the first part of training.
    This often helps stabilize VAE training.
    """
    warmup_epochs = max(1, int(total_epochs * warmup_fraction))

    if epoch >= warmup_epochs:
        return float(beta_end)

    alpha = epoch / warmup_epochs
    beta = beta_start + alpha * (beta_end - beta_start)
    return float(beta)


def train_one_epoch(
    model: MusicVAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
    grad_clip: float = 1.0,
    recon_loss_type: str = "bce",
) -> Dict[str, float]:
    model.train()

    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        x_hat, mu, logvar, _ = model(batch)
        total_loss, recon_loss, kl_loss = vae_loss(
            x_hat=x_hat,
            x=batch,
            mu=mu,
            logvar=logvar,
            beta=beta,
            recon_loss_type=recon_loss_type,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = batch.size(0)
        running_total += total_loss.item() * batch_size
        running_recon += recon_loss.item() * batch_size
        running_kl += kl_loss.item() * batch_size
        total_samples += batch_size

    return {
        "total_loss": running_total / max(total_samples, 1),
        "recon_loss": running_recon / max(total_samples, 1),
        "kl_loss": running_kl / max(total_samples, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    model: MusicVAE,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    recon_loss_type: str = "bce",
) -> Dict[str, float]:
    model.eval()

    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        x_hat, mu, logvar, _ = model(batch)
        total_loss, recon_loss, kl_loss = vae_loss(
            x_hat=x_hat,
            x=batch,
            mu=mu,
            logvar=logvar,
            beta=beta,
            recon_loss_type=recon_loss_type,
        )

        batch_size = batch.size(0)
        running_total += total_loss.item() * batch_size
        running_recon += recon_loss.item() * batch_size
        running_kl += kl_loss.item() * batch_size
        total_samples += batch_size

    return {
        "total_loss": running_total / max(total_samples, 1),
        "recon_loss": running_recon / max(total_samples, 1),
        "kl_loss": running_kl / max(total_samples, 1),
    }


def save_history_csv(history: Dict[str, list], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "beta",
                "learning_rate",
                "train_total_loss",
                "train_recon_loss",
                "train_kl_loss",
                "val_total_loss",
                "val_recon_loss",
                "val_kl_loss",
            ]
        )

        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["beta"][i],
                    history["learning_rate"][i],
                    history["train_total_loss"][i],
                    history["train_recon_loss"][i],
                    history["train_kl_loss"][i],
                    history["val_total_loss"][i],
                    history["val_recon_loss"][i],
                    history["val_kl_loss"][i],
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
    plt.plot(epochs, history["train_total_loss"], label="Train Total")
    plt.plot(epochs, history["val_total_loss"], label="Val Total")
    plt.plot(epochs, history["train_recon_loss"], label="Train Recon")
    plt.plot(epochs, history["val_recon_loss"], label="Val Recon")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def train_vae(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    num_workers: int = 0,
    patience: int = 5,
    min_delta: float = 1e-4,
    grad_clip: float = 1.0,
    recon_loss_type: str = "bce",
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
        "epoch": [],
        "beta": [],
        "learning_rate": [],
        "train_total_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_checkpoint_path = checkpoint_dir / "vae_best.pt"
    last_checkpoint_path = checkpoint_dir / "vae_last.pt"

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        beta = kl_beta_schedule(
            epoch=epoch,
            total_epochs=epochs,
            beta_start=0.0,
            beta_end=1.0,
            warmup_fraction=0.3,
        )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            grad_clip=grad_clip,
            recon_loss_type=recon_loss_type,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            beta=beta,
            recon_loss_type=recon_loss_type,
        )

        scheduler.step(val_metrics["total_loss"])

        history["epoch"].append(epoch)
        history["beta"].append(beta)
        history["learning_rate"].append(current_lr)
        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])
        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_recon_loss"].append(val_metrics["recon_loss"])
        history["val_kl_loss"].append(val_metrics["kl_loss"])

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"lr={current_lr:.6f} beta={beta:.4f} "
            f"train_total={train_metrics['total_loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f} "
            f"train_kl={train_metrics['kl_loss']:.6f} "
            f"val_total={val_metrics['total_loss']:.6f} "
            f"val_recon={val_metrics['recon_loss']:.6f} "
            f"val_kl={val_metrics['kl_loss']:.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": input_dim,
                "seq_len": SEQ_LEN,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
                "num_layers": NUM_LAYERS,
                "beta": beta,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            last_checkpoint_path,
        )

        if val_metrics["total_loss"] < (best_val_loss - min_delta):
            best_val_loss = val_metrics["total_loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "seq_len": SEQ_LEN,
                    "hidden_dim": HIDDEN_DIM,
                    "latent_dim": LATENT_DIM,
                    "num_layers": NUM_LAYERS,
                    "beta": beta,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
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

    history_csv_path = Path(PLOTS_DIR) / "vae_training_history.csv"
    history_plot_path = Path(PLOTS_DIR) / "vae_loss_curve.png"

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

    results = train_vae(
        train_path=train_path,
        val_path=val_path,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        num_workers=0,
        patience=5,
        min_delta=1e-4,
        grad_clip=1.0,
        recon_loss_type="bce",
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