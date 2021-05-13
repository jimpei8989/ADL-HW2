from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datasets.context_dataset import ContextDataset
from datasets.qa_dataset import QADataset
from datasets.utils import create_mini_batch
from models.context_selector import ContextSelector
from models.qa_model import QAModel
from trainers.context_selection_trainer import ContextSelectionTrainer
from trainers.qa_trainer import QATrainer
from utils import set_seed
from utils.config import Config
from utils.io import json_dump, pickle_dump
from utils.timer import timer
from utils.logger import logger


@timer
def context_selection(args, config):
    logger.info("=== Context Selection ===")
    config = Config.load(config)
    logger.info(f"Config: {config}")

    tokenizer = BertTokenizer.from_pretrained(config.model.bert_name)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=args.override_batch_size or config.misc.batch_size,
            num_workers=args.override_num_workers,
            collate_fn=partial(
                create_mini_batch,
                pad_keys=set(ContextDataset.TO_BE_PADDED),
                padding_value=tokenizer.pad_token_id,
            ),
            **kwargs,
        )

    dataset = ContextDataset.from_json(
        args.context_json,
        args.test_json,
        tokenizer=tokenizer,
        test=True,
    )

    model = (
        ContextSelector.from_checkpoint(
            config.model, args.context_specify_checkpoint, device=args.device
        )
        if args.context_specify_checkpoint
        else ContextSelector.load_weights(
            config.model, config.checkpoint_dir / "model_weights.pt", device=args.device
        )
    )
    trainer = ContextSelectionTrainer(
        model, checkpoint_dir=config.checkpoint_dir, device=args.device, **config.trainer
    )

    predictions = trainer.predict(to_dataloader(dataset))
    by_question = defaultdict(list)
    for pred in predictions:
        by_question[pred["id"]].append(pred)

    return [max(v, key=lambda d: d["context_score"]) for v in by_question.values()]


@timer
def question_answering(args, config, data):
    logger.info("=== Question Answering ===")
    config = Config.load(config)
    logger.info(f"Config: {config}")

    tokenizer = BertTokenizer.from_pretrained(config.model.bert_name)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=args.override_batch_size or config.misc.batch_size,
            num_workers=args.override_num_workers,
            collate_fn=partial(
                create_mini_batch,
                pad_keys=set(ContextDataset.TO_BE_PADDED),
                padding_value=tokenizer.pad_token_id,
            ),
            **kwargs,
        )

    dataset = QADataset.from_data(
        data,
        tokenizer=tokenizer,
        test=True,
    )

    model = (
        QAModel.from_checkpoint(config.model, args.qa_specify_checkpoint, device=args.device)
        if args.qa_specify_checkpoint
        else QAModel.load_weights(
            config.model, config.checkpoint_dir / "model_weights.pt", device=args.device
        )
    )
    trainer = QATrainer(
        model, checkpoint_dir=config.checkpoint_dir, device=args.device, **config.trainer
    )

    predictions = trainer.predict(to_dataloader(dataset))
    return predictions


def postprocess(d):
    offset = len(d["question_tokens"]) + 2
    answer = "".join(
        map(
            lambda s: s[2:] if s.startswith("##") else s,
            d["paragraph_tokens"][d["start_index"] - offset : d["end_index"] - offset],
        )
    )
    return {"id": d["id"], "answer": answer}


def main(args):
    set_seed(args.seed)

    elapsed, context_predictions = context_selection(args, args.context_config_json)
    logger.info(
        f"Finished context selection, get {len(context_predictions)} paragraphs"
        f"({elapsed:.2f}s elapsed)"
    )

    elapsed, qa_predictions = question_answering(args, args.qa_config_json, context_predictions)
    logger.info(
        f"Finished question_answering, get {len(qa_predictions)} answer spans"
        f"({elapsed:.2f}s elapsed)"
    )

    predictions = {d["id"]: d["answer"] for d in map(postprocess, qa_predictions)}
    json_dump(predictions, args.predict_json, ensure_ascii=False)

    if args.tmp_dir:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        pickle_dump(context_predictions, args.tmp_dir / "context_predictions.pkl")
        pickle_dump(qa_predictions, args.tmp_dir / "qa_predictions.pkl")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("context_config_json", type=Path, help="Path to config json file")
    parser.add_argument("qa_config_json", type=Path, help="Path to config json file")

    # Filesystem
    parser.add_argument("--context_json", type=Path)
    parser.add_argument("--test_json", type=Path)
    parser.add_argument("--predict_json", type=Path)
    parser.add_argument("--tmp_dir", type=Path)

    # Misc
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", default=0x06902029)
    parser.add_argument("--context_specify_checkpoint", type=Path)
    parser.add_argument("--qa_specify_checkpoint", type=Path)
    parser.add_argument("--override_batch_size", type=int)
    parser.add_argument("--override_num_workers", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu else "cpu")

    assert all(a is not None for a in [args.context_json, args.test_json, args.predict_json])

    return args


if __name__ == "__main__":
    main(parse_arguments())
