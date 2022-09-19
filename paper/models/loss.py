from torch import nn
import torch
import torch.nn.functional as F

from utils.helper import fold, right_shift


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label, weight):
        '''
            Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff * weight

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'batchmean':
            loss = loss.mean(0).sum()
        return loss


class BinaryLoss(nn.Module):
    def __init__(self, method, alpha=0.75, gamma=2, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.method = method

    def forward(self, logit, target, weight):
        if self.method == "BCE":
            # pos, neg = (target != 0).sum(), (target == 0).sum()
            # pos_weight = neg / pos
            return F.binary_cross_entropy_with_logits(logit, target, weight=weight)
        elif self.method == "Focal":
            return FocalLoss(self.alpha, self.gamma)(logit, target, weight=weight)


class MultiClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def forward(self, inputs, target, weight):
        all_loss = self.loss_fn(inputs.transpose(-2, -1), target)
        return (all_loss * weight.float().detach()).mean()


class ScoreLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin)

    def get_pairs(self, embedding, seg):
        seg_1, seg_2 = seg.unbind(-1)
        batch_idx = torch.arange(seg.size(0)).view(-1, 1)
        emb_1 = embedding[batch_idx, seg_1]
        emb_2 = embedding[batch_idx, seg_2]
        return emb_1, emb_2

    def forward(self, embedding, pos_seg, neg_seg):
        batch_size, max_len, dim = embedding.size()
        sample_num = pos_seg.size(1)
        pos_target = torch.ones(batch_size, sample_num).to(embedding.device)
        neg_target = torch.ones(batch_size, sample_num).neg().to(embedding.device)
        pos_input1, pos_input2 = self.get_pairs(embedding, pos_seg)
        neg_input1, neg_input2 = self.get_pairs(embedding, neg_seg)
        pos_loss = self.loss_fn(input1=pos_input1.view(-1, dim),
                                input2=pos_input2.view(-1, dim),
                                target=pos_target.view(-1))
        neg_loss = self.loss_fn(input1=neg_input1.view(-1, dim),
                                input2=neg_input2.view(-1, dim),
                                target=neg_target.view(-1))
        return neg_loss + pos_loss


class FinalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conf_loss = BinaryLoss(method="BCE")
        self.boundary_loss = BinaryLoss(method="BCE")
        self.end_loss = BinaryLoss(method="BCE")
        self.inside_loss = BinaryLoss(method="BCE")
        self.label_loss = MultiClassLoss()
        self.weight = config["weight"]
        self.chunk_num = config["chunk_num"]
        self.max_text_num = config["max_text_num"]
        self.chunk_size = self.max_text_num // self.chunk_num

    # def individual_forward(self, conf_logit, start_logit, end_logit, inside_logit, tags, mask, **kwargs):
    #     conf_loss = self.conf_loss(conf_logit.squeeze(-1), (tags != 0).float(), weight=mask.float().detach())
    #     start_loss = self.boundary_loss(start_logit.squeeze(-1), (tags == 1).float(), weight=mask.float().detach())
    #     end_loss = self.end_loss(end_logit.squeeze(-1), (tags == 3).float(), weight=mask.float().detach())
    #     # inside_loss = self.inside_loss(inside_logit.squeeze(-1), (tags == 2).float(), weight=mask.float().detach())
    #     total_loss = sum([
    #         self.weight["conf"] * conf_loss,
    #         self.weight["boundary"] * (start_loss + end_loss),
    #         # self.weight["inside"] * inside_loss
    #     ])
    #     return total_loss, {
    #         "total": total_loss.item(),
    #         "boundary": (start_loss + end_loss).item(),
    #         "conf": conf_loss.item(),
    #         # "inside": inside_loss.item()
    #     }

    def group_forward(self, out_logit, init_logit, final_logit, tags, mask, **kwargs):
        init_loss, final_loss = self.gate_loss(init_logit, final_logit, tags.float(), mask.float())
        cls_loss = self.label_loss(out_logit, tags, mask)
        total_loss = sum([
            self.weight["split"] * cls_loss,
            self.weight["inside"] * (init_loss + final_loss) / 2.0,
            # self.weight["inside"] * init_loss
        ])
        return total_loss, {
            "total": total_loss.item(),
            "boundary": cls_loss.item(),
            "init": init_loss.item(),
            "final": final_loss.item(),
        }

    def boundary_forward(self, conf_logit, split_logit, inside_logit, tags, mask, **kwargs):
        conf_loss = self.conf_loss(conf_logit.squeeze(-1), (tags != 0).float(), weight=mask.float().detach())
        split_loss = self.boundary_loss(split_logit.squeeze(-1), (torch.logical_or(tags == 1, tags == 3)).float(),
                                        weight=mask.float().detach())
        # inside_loss = self.inside_loss(inside_logit.squeeze(-1), (tags == 2).float(), weight=mask.float().detach())
        total_loss = sum([
            self.weight["conf"] * conf_loss,
            self.weight["boundary"] * split_loss,
            # self.weight["inside"] * inside_loss
        ])
        return total_loss, {
            "total": total_loss.item(),
            "boundary": split_loss.item(),
            "conf": conf_loss.item()
            # "inside": inside_loss.item()
        }

    # def single_gate_loss(self, pivot, block, tag, mask):
    #     blk_valid = (tag != 0) * mask
    #     pvt_valid = fold(blk_valid, -1, 1, size=self.chunk_size).mean(dim=-1)
    #     pvt_mask = fold(mask, -1, 1, size=self.chunk_num).sum(dim=-1) > 0
    #     pvt_loss = self.inside_loss(pivot, pvt_valid, weight=pvt_mask.float().detach())
    #     blk_loss = self.inside_loss(block, blk_valid, weight=mask.float().detach())
    #     return pvt_loss, blk_loss

    # def gate_loss(self, input_logit, block_logit, pivot_logit, tags, mask):
    #     pre_mask, post_mask = mask, right_shift(mask, 1, self.chunk_size // 2)  # (B, L)
    #     pre_tag, post_tag = tags, right_shift(tags, 1, self.chunk_size // 2)  # (B, L)
    #     pre_block, post_block = block_logit.unbind(1)  # (B, L)
    #     pre_pivot, post_pivot = pivot_logit.unbind(1)  # (B, N)
    #     pre_pvt_loss, pre_blk_loss = self.single_gate_loss(pre_pivot, pre_block, pre_tag, pre_mask)
    #     post_pvt_loss, post_blk_loss = self.single_gate_loss(post_pivot, post_block, post_tag, post_mask)
    #     input_loss = self.inside_loss(input_logit, (tags != 0) * mask, weight=mask.float().detach())
    #     return (pre_pvt_loss + post_pvt_loss) / 2.0,  (pre_blk_loss + post_blk_loss) / 2.0, input_loss

    def gate_loss(self, init_logit, final_logit, tags, mask):
        if init_logit is not None:
            init_loss = self.inside_loss(init_logit, (tags != 0) * mask, weight=mask.float().detach())
        else:
            init_loss = torch.zeros(1).to(mask.device)
        if final_logit is not None:
            final_loss = self.inside_loss(final_logit, (tags != 0) * mask, weight=mask.float().detach())
        else:
            final_loss = torch.zeros(1).to(mask.device)
        return init_loss, final_loss

    def gate_boundary_forward(self, split_logit, init_logit, final_logit, tags, mask, **kwargs):
        split_loss = self.boundary_loss(split_logit.squeeze(-1), ((tags == 1) + (tags == 3)).float(),
                                        weight=mask.float().detach())
        # start_loss = self.boundary_loss(start_logit.squeeze(-1), (tags == 1).float(), weight=mask.float().detach())
        # end_loss = self.boundary_loss(end_logit.squeeze(-1), (tags == 3).float(), weight=mask.float().detach())
        init_loss, final_loss = self.gate_loss(init_logit, final_logit, tags.float(), mask.float())
        # split_loss = (start_loss + end_loss) / 2.0
        total_loss = sum([
            self.weight["split"] * split_loss,
            self.weight["inside"] * (init_loss + final_loss) / 2.0,
            # self.weight["inside"] * init_loss
        ])
        return total_loss, {
            "total": total_loss.item(),
            "boundary": split_loss.item(),
            "init": init_loss.item(),
            "final": final_loss.item(),
        }

    def forward(self, *input, **kwargs):
        # return self.boundary_forward(*input, **kwargs)
        return self.gate_boundary_forward(*input, **kwargs)
        # return self.group_forward(*input, **kwargs)
        # return self.individual_forward(*input, **kwargs)
