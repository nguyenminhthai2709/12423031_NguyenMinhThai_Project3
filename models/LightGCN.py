"""LightGCN (He et al., SIGIR 2020), số lớp lấy từ cấu hình `GCN_layer` (bộ đề dùng 1 lớp).

e^(l+1) = Â e^(l) với Â = D^-1/2 A D^-1/2 dựng chỉ từ train; vector cuối = trung bình các lớp 0..L.
Kế thừa từ ID-GRec; khác bản gốc: get_rating_for_test trả về điểm thô <e_u, e_i> (không qua sigmoid) để
tránh hòa điểm giả do sigmoid bão hòa ở float32. Thứ hạng không đổi.
"""

import torch
from torch import nn

import utility.utility_data.data_graph as data_graph
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer


class LightGCN(nn.Module):
    def __init__(self, config, dataset, device):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(config["reg_lambda"])
        self.num_layers = int(config.get("GCN_layer", 1))
        size = int(config["embedding_size"])
        self.user_embedding = nn.Embedding(dataset.num_users, size)
        self.item_embedding = nn.Embedding(dataset.num_items, size)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
        graph = tools.convert_sp_mat_to_sp_tensor(data_graph.normalized_adjacency(dataset))
        self.register_buffer("graph", graph.coalesce(), persistent=False)

    def aggregate(self):
        embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        layers = [embedding]
        for _ in range(self.num_layers):
            embedding = torch.sparse.mm(self.graph, embedding)
            layers.append(embedding)
        final = torch.stack(layers, dim=1).mean(dim=1)
        return torch.split(final, [self.dataset.num_users, self.dataset.num_items])

    def forward(self, user, positive, negative):
        users, items = self.aggregate()
        bpr = losses.get_bpr_loss(users[user], items[positive], items[negative])
        reg = self.reg_lambda * losses.get_reg_loss(self.user_embedding(user), self.item_embedding(positive),
                                                    self.item_embedding(negative))
        return [bpr, reg]

    def get_rating_for_test(self, user):
        users, items = self.aggregate()
        return torch.matmul(users[user.long()], items.t())


class Trainer:
    def __init__(self, args, config, dataset, device, logger):
        self.model = LightGCN(config, dataset, device)
        self.args, self.config, self.dataset, self.device, self.logger = args, config, dataset, device, logger

    def train(self):
        return trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
