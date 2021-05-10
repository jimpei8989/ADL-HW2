from typing import Dict
from pathlib import Path

import torch

from utils.logger import logger


class BaseModel:
    @classmethod
    def load_weights(cls, config: Dict, weights_path: Path, device=None):
        logger.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, device=device)
        return cls.from_state_dict(config, state_dict)

    @classmethod
    def from_checkpoint(cls, config: Dict, checkpoint_path: Path, device=None):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, device=device)
        return cls.from_state_dict(config, checkpoint["model_state_dict"])

    @classmethod
    def from_state_dict(cls, config: Dict, state_dict: Dict):
        model = cls(**config)
        model.load_state_dict(state_dict)
        return model

    def save_weights(self, weights_path: Path):
        logger.info(f"Saving model weights to {weights_path}")
        torch.save(self.state_dict(), weights_path)
