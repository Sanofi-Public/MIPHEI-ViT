import argparse
import pyvips  # avoid errors
from benchmark.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)

    parser.add_argument("--save_logreg", action="store_true")
    parser.add_argument("--config_dir", required=False, default="configs")
    parser.add_argument("--pred_dir", required=False, default=None)
    parser.add_argument("--min_area", type=float, default=10)

    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args
    )
