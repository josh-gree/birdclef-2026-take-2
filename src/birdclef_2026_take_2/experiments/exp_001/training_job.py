from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from wm import Experiment

from birdclef_2026_take_2.dataset import MiddleWindow, RandomWindow, TrainClipDataset
from birdclef_2026_take_2.experiments.exp_001.model import EfficientNetMLP
from birdclef_2026_take_2.transforms import build_spectrogram_pipeline


class Exp001(Experiment):
    name = "exp_001"

    class Config(BaseModel):
        lr: float = 1e-3
        batch_size: int = 64
        epochs: int = 20
        hidden_dim: int = 512
        dropout: float = 0.3
        val_fraction: float = 0.2
        seed: int = 42

    @staticmethod
    def run(config: "Exp001.Config", wandb_run, run_dir: Path) -> None:
        import shutil

        data_dir = Path("/data")
        taxonomy_path = data_dir / "taxonomy.csv"
        index_path = data_dir / "train_index.parquet"

        memmap_path = Path("/tmp/train.npy")
        shutil.copy2(data_dir / "train.npy", memmap_path)

        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        taxonomy = pd.read_csv(taxonomy_path).sort_values("primary_label").reset_index(drop=True)
        num_classes = len(taxonomy)

        index = pd.read_parquet(index_path)
        rng = np.random.default_rng(config.seed)

        train_rows, val_rows = [], []
        for _, group in index.groupby("primary_label", sort=False):
            perm = rng.permutation(len(group))
            n_val = max(1, int(len(group) * config.val_fraction))
            val_rows.append(group.iloc[perm[:n_val]])
            train_rows.append(group.iloc[perm[n_val:]])

        train_index = pd.concat(train_rows).reset_index(drop=True)
        val_index = pd.concat(val_rows).reset_index(drop=True)

        train_index_path = run_dir / "train_index.parquet"
        val_index_path = run_dir / "val_index.parquet"
        train_index.to_parquet(train_index_path, index=False)
        val_index.to_parquet(val_index_path, index=False)

        train_dataset = TrainClipDataset(
            memmap_path=memmap_path,
            index_path=train_index_path,
            taxonomy_path=taxonomy_path,
            window_strategy=RandomWindow(),
            seed=config.seed,
        )
        val_dataset = TrainClipDataset(
            memmap_path=memmap_path,
            index_path=val_index_path,
            taxonomy_path=taxonomy_path,
            window_strategy=MiddleWindow(),
            seed=config.seed,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = EfficientNetMLP(num_classes, config.hidden_dim, config.dropout).to(device)
        spectrogram = build_spectrogram_pipeline().to(device)

        optimizer = AdamW(model.parameters(), lr=config.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("batch_loss", step_metric="batch_step")
        wandb_run.define_metric("batch_acc", step_metric="batch_step")
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("epoch_train_loss", step_metric="epoch")
        wandb_run.define_metric("epoch_train_acc", step_metric="epoch")
        wandb_run.define_metric("epoch_val_loss", step_metric="epoch")
        wandb_run.define_metric("epoch_val_acc", step_metric="epoch")

        criterion = nn.CrossEntropyLoss()
        batch_step = 0

        for epoch in range(config.epochs):
            train_dataset.set_epoch(epoch)
            model.train()

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch in train_loader:
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                with torch.no_grad():
                    specs = spectrogram(audio)

                logits = model(specs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                acc = correct / total

                wandb_run.log({"batch_loss": loss.item(), "batch_acc": acc, "batch_step": batch_step})
                batch_step += 1

                epoch_loss += loss.item() * total
                epoch_correct += correct
                epoch_total += total

            train_loss = epoch_loss / epoch_total
            train_acc = epoch_correct / epoch_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    audio = batch["audio"].to(device)
                    labels = batch["label"].to(device)
                    specs = spectrogram(audio)
                    logits = model(specs)
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)
                    val_loss += loss.item() * labels.size(0)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            wandb_run.log({
                "epoch_train_loss": train_loss,
                "epoch_train_acc": train_acc,
                "epoch_val_loss": val_loss,
                "epoch_val_acc": val_acc,
                "epoch": epoch,
            })

            torch.save(
                model.state_dict(),
                checkpoints_dir / f"epoch_{epoch:03d}.pt",
            )

            scheduler.step()
