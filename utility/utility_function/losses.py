"""Hàm mất mát dùng chung. Kế thừa từ ID-GRec."""

import torch


def get_bpr_loss(user_embedding, positive_embedding, negative_embedding):
    """BPR: -log sigmoid(s(u,i) - s(u,j)), lấy trung bình trên lô (Rendle et al., 2009)."""
    pos_score = torch.sum(user_embedding * positive_embedding, dim=1)
    neg_score = torch.sum(user_embedding * negative_embedding, dim=1)
    return torch.mean(-torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))


def get_reg_loss(*embeddings):
    """Phạt L2 trên vector đầu vào (ego) của lô, chia cho kích thước lô."""
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 0.5 * embedding.norm(2).pow(2) / float(embedding.shape[0])
    return reg_loss
