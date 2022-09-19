import argparse

from config.config import get_cfg_defaults
from runner.main_runner import MainRunner

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--start-from', type=int, default=-1)
    parser.add_argument('--eval-epoch', type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file("config/" + args.config + ".yaml")
    cfg.freeze()
    runner = MainRunner(cfg)
    if args.eval_epoch != -1:
        runner.load_model("checkpoints/new_mel/models-{}.pt".format(args.eval_epoch))
        # runner.eval(args.eval_epoch, "eval")
        runner.eval(args.eval_epoch, "test")
    else:
        if args.start_from >= 0:
            runner.load_model("checkpoints/new_mel/models-{}.pt".format(args.start_from))
        runner.train(args.start_from + 1)
    # for epoch in range(10, 20):
    #     runner.load_model("checkpoints/new_mel/models-{}.pt".format(epoch))
    #     runner.eval(epoch, "eval")
    #     runner.eval(epoch, "test")