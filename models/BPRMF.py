"""BPR-MF: phân rã ma trận học bằng hàm mất mát BPR (Rendle et al., UAI 2009).

Điểm s(u, i) = <e_u, e_i>. Đây là mô hình nền tham chiếu do giảng viên cấp (bộ đề: "checkpoint tham chiếu
do giảng viên cấp", phải ghi nguồn khi dùng).
"""

import torch
from torch import nn

import utility.utility_function.losses as losses
import utility.utility_train.trainer as trainer


class BPRMF(nn.Module):
    def __init__(self, config, dataset, device):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(config["reg_lambda"])
        size = int(config["embedding_size"])
        self.user_embedding = nn.Embedding(dataset.num_users, size)
        self.item_embedding = nn.Embedding(dataset.num_items, size)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

    def aggregate(self):
        """Vector dùng để chấm điểm (với MF chính là vector đầu vào)."""
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self, user, positive, negative):
        users, items = self.aggregate()
        u, p, n = users[user], items[positive], items[negative]
        bpr = losses.get_bpr_loss(u, p, n)
        reg = self.reg_lambda * losses.get_reg_loss(self.user_embedding(user), self.item_embedding(positive),
                                                    self.item_embedding(negative))
        return [bpr, reg]

    def get_rating_for_test(self, user):
        users, items = self.aggregate()
        return torch.matmul(users[user.long()], items.t())


class Trainer:
    def __init__(self, args, config, dataset, device, logger):
        self.model = BPRMF(config, dataset, device)
        self.args, self.config, self.dataset, self.device, self.logger = args, config, dataset, device, logger

    def train(self):
        return trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
