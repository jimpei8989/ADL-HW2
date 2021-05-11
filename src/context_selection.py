from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datasets.context_dataset import ContextDataset
from datasets.utils import create_mini_batch
from models.context_selector import ContextSelector
from trainers.context_selection_trainer import ContextSelectionTrainer
from utils import set_seed
from utils.config import Config
from utils.logger import logger


def main(args):
    set_seed(args.seed)
    config = Config.load(args.config_json)
    logger.info(f"Config: {config}")

    tokenizer = BertTokenizer.from_pretrained(config.model.bert_name)
    cache_dir = args.cache_dir / config.model.bert_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=args.override_batch_size or config.misc.batch_size,
            num_workers=config.misc.num_workers,
            collate_fn=partial(
                create_mini_batch,
                pad_keys=set(ContextDataset.TO_BE_PADDED),
                padding_value=tokenizer.pad_token_id,
            ),
            **kwargs,
        )

    if args.do_train:
        model = ContextSelector(**config.model)
        trainer = ContextSelectionTrainer(
            model, checkpoint_dir=config.checkpoint_dir, device=args.device, **config.trainer
        )

        trainer.train(
            to_dataloader(
                ContextDataset.from_json(
                    args.dataset_dir / "context.json",
                    args.dataset_dir / "train_splitted.json",
                    tokenizer=tokenizer,
                    split_name="train",
                    cache_dir=cache_dir,
                )
            ),
            to_dataloader(
                ContextDataset.from_json(
                    args.dataset_dir / "context.json",
                    args.dataset_dir / "val_splitted.json",
                    tokenizer=tokenizer,
                    split_name="val",
                    cache_dir=cache_dir,
                )
            ),
        )

    if args.do_evaluate or args.do_predict:
        model = (
            ContextSelector.from_checkpoint(
                config.model, args.specify_checkpoint, device=args.device
            )
            if args.specify_checkpoint
            else ContextSelector.load_weights(
                config.model, config.checkpoint_dir / "model_weights.pt", device=args.device
            )
        )
        trainer = ContextSelectionTrainer(
            model, checkpoint_dir=config.checkpoint_dir, device=args.device, **config.trainer
        )

        if args.do_evaluate:
            trainer.evaluate(
                to_dataloader(
                    ContextDataset.from_json(
                        args.dataset_dir / "context.json",
                        args.dataset_dir / "val_splitted.json",
                        tokenizer=tokenizer,
                        split_name="val",
                        cache_dir=cache_dir,
                    )
                ),
                split="val",
            )
            trainer.evaluate(
                to_dataloader(
                    ContextDataset.from_json(
                        args.dataset_dir / "context.json",
                        args.dataset_dir / "public.json",
                        tokenizer=tokenizer,
                        split_name="public",
                        cache_dir=cache_dir,
                    )
                ),
                split="public",
            )

        if args.do_predict:
            pass


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("config_json", type=Path, help="Path to config json file")

    # Filesystem
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/chineseQA"))
    parser.add_argument("--cache_dir", type=Path, default=Path("dataset/cache"))
    parser.add_argument("--test_json", type=Path)
    parser.add_argument("--predict_csv", type=Path)

    # Actions
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Misc
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", default=0x06902029)
    parser.add_argument("--specify_checkpoint", type=Path)
    parser.add_argument("--override_batch_size", type=int)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu else "cpu")

    assert not args.do_predict or all(a is not None for a in [args.test_json, args.predict_csv])

    return args


if __name__ == "__main__":
    main(parse_arguments())
