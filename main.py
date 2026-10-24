"""Huấn luyện mô hình nền (BPR-MF, LightGCN) trên split G1, chọn checkpoint theo validation, mở test một lần.

Ví dụ:
    python main.py --model LightGCN --seeds 42 0 1
    python main.py --model BPRMF --seed 42 --reg_lambda 0.001 --tag l2_1e-3

Mỗi seed tạo một thư mục log/<model>/<dataset>/<tag>/seed<k>/ chứa: config_resolved.json, environment.json,
run_manifest.json, training.log, validation_metrics.json, best_validation_model.pt, final_test_metrics.json,
final_test_groups.csv.

Khung chạy, bộ đọc cấu hình và vòng huấn luyện kế thừa từ ID-GRec (https://github.com/BlueGhostYi/ID-GRec).
"""

import hashlib
import importlib
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import time

import torch

import Parser
import utility.utility_data.data_loader as data_loader
import utility.utility_function.tools as tools
import utility.utility_train.batch_test as batch_test
from config_path.config_path import model_config_file, resolve_dataset, result_file, run_dir, verified_dataset_dir


def available_models():
    models_dir = Path(__file__).resolve().parent / "models"
    names = []
    for path in sorted(models_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        if (Path(__file__).resolve().parent / "configure" / (path.stem + ".txt")).is_file():
            names.append(path.stem)
    return names


def resolve_device(requested="auto", gpu_id=0):
    requested = str(requested).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("Yêu cầu CUDA nhưng máy không có GPU khả dụng. Dùng --device cpu.")
        return torch.device("cpu")
    return torch.device("cuda:{}".format(int(gpu_id)))


def runtime_environment(device):
    details = {"device": str(device), "python": platform.python_version(), "platform": platform.platform(),
               "torch": torch.__version__, "cuda": torch.version.cuda}
    if device.type == "cuda":
        details["gpu_name"] = torch.cuda.get_device_name(device)
    return details


def config_id(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "cfg-" + hashlib.sha256(canonical).hexdigest()[:10]


def configure_logging(log_path=None):
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return logging.getLogger("training")


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def run_one_seed(model_name, args, config, device, seed, environment):
    started = time()
    tag = args.tag or config_id(config)
    output_dir = run_dir(model_name, config["dataset"], tag, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "training.log")
    logger.info("Bắt đầu: model=%s dataset=%s seed=%d output=%s", model_name, config["dataset"], seed, output_dir)
    manifest = {"status": "running", "model": model_name, "dataset": config["dataset"], "seed": int(seed),
                "tag": tag, "started_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": dict(config),
                "environment": environment,
                "selection": "validation {}@{} [{}]".format(config.get("selection_metric", "ndcg"), config.get("selection_K", 10),
                                                            config.get("selection_group", "overall")),
                "test_policy": "mở test một lần sau khi nạp checkpoint tốt nhất trên validation"}
    try:
        tools.set_seed(seed)
        dataset = data_loader.Data(str(verified_dataset_dir(config["dataset"])), config, logger=logger)
        dataset.training_output_dir = output_dir
        write_json(output_dir / "config_resolved.json", config)
        write_json(output_dir / "environment.json", environment)
        write_json(result_file(output_dir, "manifest"), manifest)
        trainer_class = getattr(importlib.import_module("models." + model_name), "Trainer")
        recommender = trainer_class(args, config, dataset, device, logger)
        recommender.model.training_seed = int(seed)
        recommender.train()
        batch_test.final_test(dataset, recommender.model, device, config, logger)
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=round(time() - started, 3), completed_epochs=int(dataset.current_epoch),
                        train_seconds_per_epoch=round(sum(getattr(dataset, "epoch_seconds", [0.0]))
                                                      / max(len(getattr(dataset, "epoch_seconds", [])), 1), 4),
                        num_parameters=int(tools.count_parameters(recommender.model)))
        write_json(result_file(output_dir, "manifest"), manifest)
        logger.info("Hoàn tất seed %d (%.1f giây)", seed, time() - started)
        return 0
    except Exception:
        logger.exception("Lỗi khi chạy seed %d", seed)
        manifest["status"] = "failed"
        write_json(result_file(output_dir, "manifest"), manifest)
        return 1


def main():
    logger = configure_logging()
    preliminary = Parser.parse_model_args()
    models = available_models()
    if preliminary.model not in models:
        logger.error("Chọn --model trong: %s", ", ".join(models) or "(thư mục models/ chưa có mô hình nền)")
        return 1
    default_config = tools.read_configuration(str(model_config_file(preliminary.model)), preliminary.model)
    args = Parser.parse_args(default_config)
    config = Parser.apply_config_overrides(default_config, args)
    config["dataset"] = resolve_dataset(config["dataset"])   # ml-100k / ml-100k-c1 → dataset đang chọn (P3_DATASET)
    device = resolve_device(args.device, args.gpu_id)
    environment = runtime_environment(device)
    seeds = [args.seed] if args.seed is not None else list(dict.fromkeys(args.seeds))
    failed = [seed for seed in seeds if run_one_seed(preliminary.model, args, config, device, seed, environment) != 0]
    configure_logging()
    if failed:
        logging.getLogger("training").error("Các seed lỗi: %s", failed)
        return 1
    logging.getLogger("training").info("Hoàn tất các seed: %s", seeds)
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    sys.exit(main())
