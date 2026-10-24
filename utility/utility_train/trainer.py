"""Vòng huấn luyện chung cho mô hình có hàm mất mát BPR (kế thừa từ ID-GRec).

Theo giao thức G1: Adam, kiểm tra validation mỗi `interval` epoch, dừng sau `early_stopping` lần không cải thiện,
trần `training_epochs` epoch.
"""

from time import time

import torch

import utility.utility_function.tools as tools
import utility.utility_train.batch_test as batch_test

try:
    from tqdm import tqdm
except ImportError:  # tqdm không bắt buộc
    def tqdm(iterable, **_kwargs):
        return iterable


def universal_trainer(model, args, config, dataset, device, logger):
    total_epochs = int(config["training_epochs"])
    interval = int(config.get("interval", 5))
    batch_size = int(config["batch_size"])
    model.to(device)
    logger.info("Số tham số: %s", "{:,}".format(tools.count_parameters(model)))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(config["learn_rate"]))
    best = {"count": 0, "epoch": 0, "stop": False}
    started = time()
    dataset.epoch_seconds = []
    for epoch in range(total_epochs):
        dataset.current_epoch = epoch + 1
        epoch_started = time()
        model.train()
        samples = dataset.sample_data_to_train_all()
        users = torch.as_tensor(samples[:, 0], dtype=torch.long, device=device)
        positives = torch.as_tensor(samples[:, 1], dtype=torch.long, device=device)
        negatives = torch.as_tensor(samples[:, 2], dtype=torch.long, device=device)
        users, positives, negatives = tools.shuffle(users, positives, negatives)
        totals = None
        num_batches = 0
        batches = tools.mini_batch(users, positives, negatives, batch_size=batch_size)
        for batch_users, batch_pos, batch_neg in tqdm(batches, desc="epoch {}/{}".format(epoch + 1, total_epochs),
                                                      leave=False):
            losses = model(batch_users, batch_pos, batch_neg)
            loss = sum(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            values = [float(item.item()) for item in losses]
            totals = values if totals is None else [a + b for a, b in zip(totals, values)]
            num_batches += 1
        dataset.epoch_seconds.append(time() - epoch_started)
        logger.info("Epoch %d: loss=%s (%.2f giây)", epoch + 1,
                    " + ".join("{:.5f}".format(v / max(num_batches, 1)) for v in totals), dataset.epoch_seconds[-1])
        if tools.should_run_validation(epoch, total_epochs, interval):
            _, best = batch_test.general_test(dataset, model, device, config, epoch, best, optimizer, logger)
            if best.get("stop"):
                break
    logger.info("Kết thúc huấn luyện sau %s; epoch tốt nhất %d (%s=%.6f)", tools.format_duration(time() - started),
                best["epoch"], best.get("name"), best.get("value", float("nan")))
    return best
