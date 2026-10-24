"""Tiện ích chung: seed, đọc cấu hình, chia lô, ma trận thưa. Kế thừa từ ID-GRec."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def count_parameters(model, trainable_only=False):
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def should_run_validation(epoch, total_epochs, interval):
    """Kiểm tra validation ở mỗi `interval` epoch và luôn ở epoch cuối."""
    interval = int(interval)
    if interval <= 0:
        raise ValueError("interval must be a positive integer")
    completed = int(epoch) + 1
    return completed % interval == 0 or completed == int(total_epochs)


def read_configuration(filename, model):
    """Đọc tệp cấu hình dạng `khóa = giá trị` (bỏ qua dòng trống và dòng bắt đầu bằng #)."""
    if not os.path.exists(filename):
        raise IOError("Không thấy tệp cấu hình của {}: {}".format(model, filename))
    config = {}
    with open(filename, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "=" not in text:
                raise ValueError("Sai định dạng cấu hình {}:{}: {}".format(filename, line_number, text))
            name, value = text.split("=", 1)
            config[name.strip()] = value.split("#", 1)[0].strip()
    return config


def shuffle(*arrays):
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("Inputs to shuffle must have the same length.")
    indices = np.arange(len(arrays[0]))
    np.random.shuffle(indices)
    return arrays[0][indices] if len(arrays) == 1 else tuple(x[indices] for x in arrays)


def mini_batch(*tensors, batch_size=1024):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def convert_sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    index = torch.stack([torch.as_tensor(coo.row, dtype=torch.long), torch.as_tensor(coo.col, dtype=torch.long)])
    return torch.sparse_coo_tensor(index, torch.as_tensor(coo.data, dtype=torch.float32), torch.Size(coo.shape))
