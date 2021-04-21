from argparse import ArgumentParser
from pathlib import Path

from tokenizers.tokenizer import Tokenizer
from tokenizers.utils import concurrent_tokenize

from utils.logger import logger
from utils.io import json_load
from utils.tqdmm import tqdmm


def main(args):
    logger.info(args)

    logger.info("Analyzing context.json")
    context = json_load(args.dataset_dir / "context.json")
    logger.info(f"#contexts: {len(context)}")
    logger.info(
        "\n".join(
            [
                "About the lengthes (character level)",
                f"mean:\t{sum(map(len, context)) / len(context):.2f}",
                f"min:\t{min(map(len, context))}",
                f"max:\t{max(map(len, context))}",
                f">510:\t{sum(map(lambda t: len(t) > 510, context))} / {len(context)}",
            ]
        )
    )

    tokenizer = Tokenizer()
    tokenized = concurrent_tokenize(tokenizer, context)
    logger.info(
        "\n".join(
            [
                "About the lengthes (token level)",
                f"mean:\t{sum(map(len, tokenized)) / len(tokenized):.2f}",
                f"min:\t{min(map(len, tokenized))}",
                f"max:\t{max(map(len, tokenized))}",
                f">510:\t{sum(map(lambda t: len(t) > 510, tokenized))}/{len(context)}"
            ]
        )
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default="dataset/chineseQA/")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
