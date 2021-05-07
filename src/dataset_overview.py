from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from utils.logger import logger
from utils.io import json_load


def main(args):
    logger.info(args)

    logger.info("Analyzing context.json...")
    context = json_load(args.dataset_dir / "context.json")
    logger.info(f"#contexts: {len(context)}")
    context_lengthes = sorted(map(len, context))
    logger.info(
        "\n".join(
            [
                "About the lengthes (character level)",
                f"mean:\t{sum(context_lengthes) / len(context):.2f}",
                f"min:\t{min(context_lengthes)}",
                f"max:\t{max(context_lengthes)}",
                f"90%:\t{context_lengthes[int(len(context) * 0.9)]}",
                f">510:\t{sum(map(lambda t: len(t) > 510, context))} / {len(context)}",
            ]
        )
    )

    def print_counter(counter):
        return "{\n" + "\n".join(f"  {k} -> {v}" for k, v in sorted(counter.items())) + "\n}"

    def analyze(json_name, is_private=False):
        logger.info(f"Analyzing {json_name}...")
        data = json_load(args.dataset_dir / json_name)
        logger.info(f"#training examples: {len(data)}")

        num_paragraph_counter = Counter(map(lambda d: len(d["paragraphs"]), data))
        logger.info(f"About the related paragraphs: {print_counter(num_paragraph_counter)}")

        question_length_counter = Counter(map(lambda d: len(d["question"]) // 10, data))
        logger.info(f"About the question lengths: {print_counter(question_length_counter)}")

        if not is_private:
            assert all(
                all(
                    a["text"] == context[d["relevant"]][a["start"] : a["start"] + len(a["text"])]
                    for a in d["answers"]
                )
                for d in data
            )

    analyze("train.json")
    analyze("public.json")
    analyze("private.json", is_private=True)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default="dataset/chineseQA/")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
