"""Đọc tham số dòng lệnh. Mọi khóa trong configure/<Model>.txt đều ghi đè được bằng --<khóa> <giá trị>.

Kế thừa cách tổ chức của ID-GRec (https://github.com/BlueGhostYi/ID-GRec).
"""

import argparse
import sys

CLI_ALIASES = {"training_epochs": ("--epochs",)}

# Seed huấn luyện mặc định (lõi 1.4): 42, 0, 1 – mọi đề tài chạy 3 seed (bộ đề ghi 2026, 2027, 2028; mức Dễ 2 seed).
DEFAULT_TRAINING_SEEDS = (42, 0, 1)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _base_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Project 3 – huấn luyện mô hình nền", add_help=add_help)
    parser.add_argument("--model", type=str, default="unknown", help="tên mô hình: BPRMF hoặc LightGCN")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_TRAINING_SEEDS),
                        help="các seed chạy lần lượt (mặc định 42 0 1)")
    parser.add_argument("--seed", type=int, default=None, help="chạy đúng một seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None,
                        help="tên thư mục cấu hình trong log/ (mặc định: mã băm cấu hình)")
    return parser


def parse_model_args(argv=None):
    parser = _base_parser(add_help=False)
    args, _ = parser.parse_known_args(argv)
    actual_argv = sys.argv[1:] if argv is None else argv
    if args.model == "unknown" and any(value in ("-h", "--help") for value in actual_argv):
        _base_parser().parse_args(actual_argv)
    return args


def parse_args(config=None, argv=None):
    parser = _base_parser()
    for key in sorted(config or {}):
        option_strings = ["--" + key] + list(CLI_ALIASES.get(key, ()))
        parser.add_argument(*option_strings, dest="config_" + key, default=None, metavar="VALUE",
                            help="ghi đè cấu hình: {}".format(key))
    return parser.parse_args(argv)


def apply_config_overrides(config, args):
    overridden = dict(config)
    for key in config:
        value = getattr(args, "config_" + key, None)
        if value is not None:
            overridden[key] = value
    return overridden
