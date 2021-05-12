from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from utils.io import json_dump
from utils.logger import logger
from utils.timer import timer
from utils.tqdmm import tqdmm


class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_accumulate=1,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_freq: int = 5,
        device=None,
    ):
        self.model = model

        self.cur_epoch = 1
        self.total_epochs = epochs

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None
        self.gradient_accumulate = gradient_accumulate

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        logger.info(self.model)

        self.metrics = self.create_metrics()
        logger.info(f"Created {len(self.metrics)} metrics: {', '.join(self.metrics.keys())}")

    @abstractmethod
    def create_metrics(self):
        return {}

    def update_metrics(self, *args):
        return {name: metric.update(*args) for name, metric in self.metrics.items()}

    @abstractmethod
    def run_batch(self, batch):
        return 0, {}

    @abstractmethod
    def run_predict_batch(self, batch):
        return []

    def load_checkpoint(self, checkpoint_path: Path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.cur_epoch = checkpoint["cur_epoch"] + 1

    def save_checkpoint(self, checkpoint_path: Path):
        logger.info(f"Saving current checkpoint to {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cur_epoch": self.cur_epoch,
            },
            checkpoint_path,
        )

    @timer
    def run_epoch(self, dataloader, split="", train=False, epoch=0):
        self.model.train() if train else self.model.eval()
        self.model.to(self.device)

        all_losses = []
        for metric in self.metrics.values():
            metric.initialize_epoch()

        with torch.set_grad_enabled(train):
            original_desc = f"{split} | loss: [LOSS] | [METRICS]"

            tqdm_iterator = tqdmm(dataloader, desc=original_desc)
            for i, batch in enumerate(tqdm_iterator):
                if train:
                    self.optimizer.zero_grad()

                loss, metrics = self.run_batch(batch)
                loss /= self.gradient_accumulate

                if train:
                    loss.backward()
                    if (i + 1) % self.gradient_accumulate == 0:
                        self.optimizer.step()

                all_losses.append(loss.item())
                tqdm_iterator.set_description(
                    original_desc.replace("[LOSS]", f"{loss:.4f}").replace(
                        "[METRICS]", self.format_metrics(metrics)
                    )
                )
            else:
                if train:
                    self.optimizer.step()

        return np.mean(all_losses).item(), {n: m.finalize_epoch() for n, m in self.metrics.items()}

    @staticmethod
    def format_metrics(metrics):
        return " | ".join(map(lambda p: f"{p[0]}: {p[1]:.3f}", metrics.items()))

    def train(self, train_dataloader, val_dataloader):
        logger.info(f"Training model for {self.total_epochs} epochs...")
        training_log = []
        for epoch in range(self.cur_epoch, self.total_epochs + 1):
            self.cur_epoch = epoch

            logger.info(f"Epoch {epoch:2d} / {self.total_epochs:2d}")
            train_time, (train_loss, train_metrics) = self.run_epoch(
                train_dataloader, split="train", train=True, epoch=epoch
            )
            logger.info(
                f"Train | {train_time:8.3f}s | loss: {train_loss:.3f} | "
                f"{self.format_metrics(train_metrics)}"
            )

            val_time, (val_loss, val_metrics) = self.run_epoch(
                val_dataloader, split="val", epoch=epoch
            )
            logger.info(
                f"Val   | {val_time:8.3f}s | loss: {val_loss:.3f} | "
                f"{self.format_metrics(val_metrics)}"
            )

            if epoch % self.checkpoint_freq == 0:
                try:  # In order to bypass meow1 / meow2 disk full issue
                    self.save_checkpoint(self.checkpoint_dir / f"checkpoint_{epoch:03d}.pt")
                except OSError as e:
                    logger.warning(f"Trying to save a checkpoint, but '{e}' occured")

            training_log.append(
                {
                    "epoch": epoch,
                    "train_time": train_time,
                    "train_loss": train_loss,
                    "train_metrics": train_metrics,
                    "val_time": val_time,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                }
            )

        self.model.save_weights(self.checkpoint_dir / "model_weights.pt")
        json_dump(training_log, self.checkpoint_dir / "train_log.json")

    def evaluate(self, dataloader, split=""):
        logger.info(f"Evaluating model using {split} data...")
        duration, (loss, metrics) = self.run_epoch(dataloader, split=split, train=False, epoch=-1)
        logger.info(
            f"{split[:5]:5s} | {duration:7.3f}s | loss: {loss:.3f} | {self.format_metrics(metrics)}"
        )

    def predict(self, dataloader):
        logger.info("Predicting")
        predictions = []
        self.model.to(self.device)
        with torch.no_grad():
            for batch in tqdmm(dataloader, desc="Predicting"):
                predictions += self.run_predict_batch(batch)

        return predictions
