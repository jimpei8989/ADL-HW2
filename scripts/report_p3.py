import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt


def main(args):
    with open(args.log_json) as f:
        train_log = json.load(f)

    def extract(*keys):
        def recursive_get(d, keys):
            return d[keys[0]] if len(keys) == 1 else recursive_get(d[keys[0]], keys[1:])

        return [recursive_get(d, keys) for d in train_log]

    epochs = [d["epoch"] for d in train_log]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(
        epochs,
        extract("train_loss"),
        label="train loss",
        color="#f9647e",
        alpha=0.6,
        linestyle="--",
    )
    ax.plot(epochs, extract("val_loss"), label="train loss", color="#f9647e", alpha=0.6)
    ax.plot(epochs, extract("val_metrics", "acc"), label="EM", color="#988bde")
    ax.plot(epochs, extract("val_metrics", "f1"), label="F1", color="#ffa500")

    ax.legend()
    ax.set_title("Question Answering")
    ax.set_xlabel("Epochs")

    fig.tight_layout()
    fig.savefig(args.output_png)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--log_json")
    parser.add_argument("--output_png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
