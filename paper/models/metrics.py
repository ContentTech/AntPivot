import sys

import numpy as np
import torch
import torch.nn.functional as F

from models.new_metrics import calc_pr_dict, calc_item_pr, calculate_average_precision
from utils.helper import right_shift, left_shift, masked_operation

START = 1
END = 3

def max_min_norm(x: torch.Tensor, dim: int, upper_bound=None, lower_bound=None, bias=0):
    upper_bound = x.max(dim=dim, keepdim=True)[0] if upper_bound is None else torch.zeros_like(x) + upper_bound
    lower_bound = x.min(dim=dim, keepdim=True)[0] if lower_bound is None else torch.zeros_like(x) + lower_bound
    return bias + (1 - bias) * (x - lower_bound) / (upper_bound - lower_bound)


def calculate_background_score(prob, target, mask):
    # conf = logits.sigmoid()
    background_mask = (target == 0) * mask
    foreground_mask = (target != 0) * mask
    background = prob.masked_select(background_mask.bool())
    foreground = prob.masked_select(foreground_mask.bool())
    background_avg = 0 if background.numel() == 0 else background.mean().squeeze().item()
    foreground_avg = 0 if foreground.numel() == 0 else foreground.mean().squeeze().item()
    return background_avg, foreground_avg


def calculate_f1(correct_num, pred_num, gt_num):
    recall = correct_num / gt_num
    precision = correct_num / pred_num
    if recall + precision != 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0
    return f1, recall, precision

def calculate_accuracy(logits, target, mask):
    prediction = logits.max(-1)[1]   # (*, L)
    return ((prediction == target) * mask).sum() / mask.sum()


def calculate_iou1d(pred_first, pred_last, real_first, real_last):
    """
        calculate temporal intersection over union
    """
    return_type = type(pred_first)
    if type(pred_first) is torch.Tensor:
        pred_first, pred_last, real_first, real_last = map(lambda x: x.cpu().numpy(),
                                                           [pred_first, pred_last, real_first, real_last])
    elif type(pred_first) is list:
        pred_first, pred_last, real_first, real_last = map(lambda x: np.array(x),
                                                           [pred_first, pred_last, real_first, real_last])
    pred_first, pred_last = pred_first.astype(float), pred_last.astype(float)
    real_first, real_last = real_first.astype(float), real_last.astype(float)
    union = (np.min(np.stack([pred_first, real_first], 0), 0), np.max(np.stack([pred_last, real_last], 0), 0))
    inter = (np.max(np.stack([pred_first, real_first], 0), 0), np.min(np.stack([pred_last, real_last], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-9) / (union[1] - union[0] + 1e-9)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    union_len = np.clip(union[1] - union[0], 0, None)
    inter_len = np.clip(inter[1] - inter[0], 0, None)
    if return_type is torch.Tensor:
        return map(lambda x: torch.from_numpy(x), [iou, union_len, inter_len])
    elif return_type is list:
        return map(lambda x: list(x), [iou, union_len, inter_len])
    return [iou, union_len, inter_len]

def interval_matching(pred_intervals, gt_intervals):
    # pred in (L_p, 2), gt in (L_g, 2)
    pred_num, gt_num = pred_intervals.size(0), gt_intervals.size(0)
    ext_pred = pred_intervals.unsqueeze(0).repeat(gt_num, 1, 1)
    ext_gt = gt_intervals.unsqueeze(1).repeat(1, pred_num, 1)
    # both in (L_g, L_p, 2)
    all_iou, _, _ = calculate_iou1d(ext_pred[:, :, 0], ext_pred[:, :, 1], ext_gt[:, :, 0], ext_gt[:, :, 1])
    gt_idx = all_iou.max(dim=0)[1]  # (L_p)
    best_gt = gt_intervals[gt_idx]  # (L_p, 2)
    iou, _, inter_len = calculate_iou1d(pred_intervals[:, 0], pred_intervals[:, 1], best_gt[:, 0], best_gt[:, 1])
    # all in (L_g, L_p)
    pred_len = pred_intervals[:, 1] - pred_intervals[:, 0]
    gt_len = gt_intervals[:, 1] - gt_intervals[:, 0]
    all_union, all_inter = (pred_len.sum() + gt_len.sum()), inter_len.sum()
    start_shift = (best_gt[:, 0] - pred_intervals[:, 0]).abs()  # (L_p)
    end_shift = (best_gt[:, 1] - pred_intervals[:, 1]).abs()  # (L_p)
    start_correct = (start_shift <= 5).sum()
    end_correct = (end_shift <= 5).sum()
    overall_iou = all_inter / (all_union - all_inter)
    pred_len = pred_intervals[:, 1] - pred_intervals[:, 0]
    gt_len = gt_intervals[:, 1] - gt_intervals[:, 0]
    # FIXME: ? 交并比算的是所有inter比上所有union？ 是否考虑gt和pred的一对多关系？
    return {
        "start_correct": start_correct.item(),  # Int, the number of start lying in 5s intervals, MAX: L_p
        "start_shift": start_shift.cpu().numpy().tolist(),  # List(Float), the start shift list
        "end_correct": end_correct.item(),  # Same with start_correct
        "end_shift": end_shift.cpu().numpy().tolist(),  # Same with start_shift
        "overall_iou": overall_iou.item(),  # Float, the overall intersection over the overall union
        "pred_num": pred_num, # Int, the number of current prediction
        "gt_num": gt_num,  # Int, the number of current ground-truth
        "pred_len": pred_len.cpu().numpy().tolist(),  # List(Float), the list of length of predictions
        "gt_len": gt_len.cpu().numpy().tolist()  # List(Float), the list of length of gts
    }


def get_target_intervals(tag_seq, time, mask):
    pos_intervals, time_intervals, inside, start = [], [], False, 0
    max_length = mask.sum()
    for pos_idx in range(max_length):
        if tag_seq[pos_idx] == START:
            inside, start = True, pos_idx
        if tag_seq[pos_idx] == END and inside:
            inside, end = False, pos_idx
            pos_intervals.append([start, end])
            time_intervals.append([time[start][0], time[end][1]])
    return torch.tensor(time_intervals), torch.tensor(pos_intervals)


def simple_multi_greedy(logits, target, mask, time):
    """
    :param logits: Float Tensor, in (L, 4)
    :param target: Int Tensor, in (L)
    :param mask: Int Tensor, in (L)
    :param time: Int Tensor, in (L, 2)
    :return: all predicted intervals retrieved from logits and all ground-truth intervals
    """
    prediction = logits.max(-1)[1]
    pred_time, pred_pos = get_target_intervals(prediction, time, mask)
    gt_time, gt_pos = get_target_intervals(target, time, mask)
    return pred_time, gt_time, pred_pos, gt_pos

def filter(conf, thresh, mask):
    idx = torch.arange(conf.size(0))
    thresh_mask = conf >= thresh
    last_conf, cur_conf, next_conf = conf[:-2], conf[1:-1], conf[2:]
    peak_mask = F.pad((cur_conf >= last_conf) * (cur_conf >= next_conf), [1, 1], value=False)
    return idx.masked_select(torch.logical_and(
        torch.logical_and(thresh_mask, peak_mask),
        # torch.logical_or(thresh_mask, peak_mask),
        mask.bool()
    ))



# def simple_binary_greedy(split_logit, target, mask, time, thresh=0.3, smooth_radius=3):
#     """
#     :param split_logit: Float Tensor, in (L)
#     :param target: Int Tensor, in (L)
#     :param mask: Int Tensor, in (L)
#     :param time: Int Tensor, in (L, 2)
#     :param thresh: thresh for boundary confidence
#     :return: all predicted intervals retrieved from logits and all ground-truth intervals
#     """
#     split_pred = split_logit.sigmoid()
#     # start_pred = torch.avg_pool1d(start_pred.view(1, 1, -1), kernel_size=smooth_radius, padding=(smooth_radius - 1) // 2, stride=1).view(-1)
#     # end_pred = torch.avg_pool1d(end_pred.view(1, 1, -1), kernel_size=smooth_radius, padding=(smooth_radius - 1) // 2, stride=1).view(-1)
#     gt_time, gt_pos = get_target_intervals(target, time, mask)
#     pred_time, pred_pos = [], []
#     start_list = filter(start_pred, thresh, mask)
#     end_list = filter(end_pred, thresh, mask)
#     start_pointer, end_pointer = 0, 0
#     while start_pointer < len(start_list):
#         start_time = start_list[start_pointer]
#         while (end_pointer < len(end_list) and end_list[end_pointer] <= start_time):
#             end_pointer += 1
#         if end_pointer == len(end_list):
#             break
#         end_time = end_list[end_pointer]
#         pred_pos.append([start_time, end_time])
#         pred_time.append([time[start_time][0], time[end_time][1]])
#         while start_pointer < len(start_list) and start_list[start_pointer] <= end_time:
#             start_pointer += 1
#     return torch.tensor(pred_time), torch.tensor(gt_time), torch.tensor(pred_pos), torch.tensor(gt_pos)

def divide_conquer(split_score, split_idx, inside_conf, inside_thresh):
    # both in L, split: inside, conf: outside
    def recurrent(left, right):
        left_idx, right_idx = split_idx[left], split_idx[right]
        inside_cond = inside_conf[left_idx:right_idx + 1].mean() >= inside_thresh
        if left == right or (left == right - 1 and not inside_cond):
            return []
        if (left == right - 1 and inside_cond):
            return [[left_idx.item(), right_idx.item()]]
        best_idx = split_score[left + 1:right].max(-1)[1] + left + 1
        result = recurrent(left, best_idx) + recurrent(best_idx, right)
        if len(result) == 0:
            return [[left_idx.item(), right_idx.item()]] if inside_cond else []
        return result
    return recurrent(0, len(split_idx) - 1)


def pos2time(pos_pair, time):
    results = []
    for idx in range(len(pos_pair)):
        results.append([time[pos_pair[idx][0]][0], time[pos_pair[idx][1]][1]])
    return results

def check_score(embedding, mask, gt_start, gt_end):
    length = mask.sum()
    start_idx, end_idx = torch.meshgrid((torch.arange(length - 1), torch.arange(length - 1) + 1))
    rand_idx = np.random.choice((length - 1) * (length - 1), 200)
    rand_start = start_idx.reshape(-1)[rand_idx]
    rand_end = end_idx.reshape(-1)[rand_idx]
    rand_st_emb = embedding[rand_start]
    rand_ed_emb = embedding[rand_end]
    gt_st_emb = embedding[gt_start]
    gt_ed_emb = embedding[gt_end]
    gt_score = torch.cosine_similarity(gt_st_emb, gt_ed_emb, dim=-1).mean().item()
    rand_score = torch.cosine_similarity(rand_st_emb, rand_ed_emb, dim=-1).mean().item()
    return gt_score, rand_score

def generate_split_point(split_prob, tags, split_thresh):
    max_length = split_prob.size(0)
    idx = torch.arange(max_length)
    gt_split = idx[torch.logical_or(tags == 1, tags == 3)]
    pred_split = idx[split_prob >= split_thresh]
    pred_score = split_prob[pred_split]
    filter = torch.logical_or(pred_score >= left_shift(pred_score, dim=0, length=1),
                              pred_score >= right_shift(pred_score, dim=0, length=1))
    pred_split = pred_split[filter]
    pred_score = split_prob[pred_split]
    return pred_split, pred_score, gt_split

def dynamic_programming(split_prob, non_split_prob, inside_prob, outside_prob, mask):
    valid_len, max_len = mask.sum(-1), len(mask)
    last = valid_len - 1
    # no_inside_score = (1 - inside_score).masked_fill(1 - mask, -1e5)
    # inside_score = inside_score.masked_fill(1 - mask, -1e5)
    # split_score = split_score.masked_fill(1 - mask, -1e5)
    # sm_inside, sm_split = inside_score.log_softmax(-1), split_score.log_softmax(-1)
    # sm_no_inside = no_inside_score.log_softmax(-1)
    f = torch.zeros(max_len, 4)
    c = torch.zeros(max_len, 4).long()
    d = torch.ones(max_len).neg().long()
    f[0][0], f[0][3] = split_prob[0], outside_prob[0] * non_split_prob[0]
    # f[0][0], f[0][3] = split_prob[0] * inside_prob[0], outside_prob[0] * (1 - split_prob[0])
    # 0: start, 1: end, 2: inside, 3: outside
    p = {0: (1, 3), 1: (0, 2), 2: (0, 2), 3: (1, 3)}
    for i in range(1, valid_len):
        # print("#" * 5)
        # print(f"pos: {i}, start: {split_prob[i]:.2f}, end: {split_prob[i]:.2f}, "
        #       f"inside: {inside_prob[i] * non_split_prob[i]:.2f}, "
        #       f"outside: {outside_prob[i] * non_split_prob[i]:.2f}")
        v = {
            # 0: split_prob[i] * inside_prob[i],
            # 1: split_prob[i] * inside_prob[i],
            0: split_prob[i], 1: split_prob[i],
            2: inside_prob[i] * non_split_prob[i],
            3: outside_prob[i] * non_split_prob[i]
        }
        for t in range(4):
            score_1 = f[i-1][p[t][0]]
            score_2 = f[i-1][p[t][1]]
            f[i][t] = max(score_1, score_2) + v[t]
            c[i][t] = p[t][0] if score_1 >= score_2 else p[t][1]
    d[last] = 3 if f[last][3] >= f[last][1] else 1
    last = last - 1
    while last >= 0:
        d[last] = c[last + 1][d[last + 1]]
        last = last - 1
    idx = torch.arange(max_len)
    start = idx.masked_select(d == 0).cpu().numpy().tolist()
    end = idx.masked_select(d == 1).cpu().numpy().tolist()
    return list(zip(start, end))

def interval2pos(interval):
    final = []
    for item in interval:
        final.extend([item[0], item[1]])
    final.sort()
    return final

def normalize(prob, mask):
    length = mask.sum()
    valid_part = prob[:length]
    valid_part = max_min_norm(valid_part, -1)
    result = torch.zeros_like(prob)
    result[:length] = valid_part
    return result


def dp_split(split_prob, conf_prob, target, mask, time):
    non_split_prob = normalize(1 - split_prob, mask)
    outside_prob = normalize(1 - conf_prob, mask)
    split_prob, inside_prob = normalize(split_prob, mask), normalize(conf_prob, mask)
    # start_prob, end_prob = normalize(start_prob, mask), normalize(end_prob, mask)
    # non_split_prob = (1 - start_prob) * (1 - end_prob) # normalize((1 - start_prob) * (1 - end_prob), mask)
    # If no normalization, the length will be shorter
    # inside_prob = normalize(conf_prob, mask)
    gt_time, gt_pos = get_target_intervals(target, time, mask)
    pred_pos = dynamic_programming(split_prob, non_split_prob, inside_prob, outside_prob, mask)
    pred_time = pos2time(pred_pos, time)
    split_set = set(interval2pos(pred_pos))
    gt_split = interval2pos(gt_pos)
    # print("=" * 5)
    # print("PRED:", pred_pos)
    # print("GT:", gt_pos.cpu().numpy().tolist())
    # print("=" * 5)
    recall_items = [(item in split_set) for item in gt_split]
    recall = sum(recall_items) / len(gt_split)
    return torch.tensor(pred_time), torch.tensor(gt_time), torch.tensor(pred_pos), torch.tensor(gt_pos), recall

def simple_split_greedy(split_prob, conf_prob, target, mask, time, split_thresh, inside_thresh):
    inside_conf = conf_prob.masked_fill(mask == 0, 0)
    inside_conf = max_min_norm(inside_conf, dim=0)
    split_prob = split_prob.masked_fill(mask == 0, 0)
    split_prob = max_min_norm(split_prob, dim=0)
    split_idx, split_score, gt_split = generate_split_point(split_prob, target, split_thresh)
    if len(split_idx) == 0:
        return [None] * 5
    gt_time, gt_pos = get_target_intervals(target, time, mask)
    pred_pos = divide_conquer(split_score, split_idx, inside_conf, inside_thresh)
    pred_time = pos2time(pred_pos, time)
    split_set = set(split_idx.cpu().numpy().tolist())
    recall_items = [(item in split_set) for item in gt_split.cpu().numpy().tolist()]
    recall = sum(recall_items) / len(gt_split)
    # print("==" * 5)
    # print("PRED SPLIT:", split_idx.cpu().numpy().tolist())
    # print("GT SPLIT:", gt_split.cpu().numpy().tolist())
    # print("FILT SPLIT:", pred_pos)
    # print("GT SPLIT SCORE:", split_prob[gt_split].mean().item())
    # print("OVERALL SCORE:", split_prob.mean().item())
    # print("==" * 5)
    return torch.tensor(pred_time), torch.tensor(gt_time), torch.tensor(pred_pos), torch.tensor(gt_pos), recall


def calc_inside_prob(init_logit, final_logit):
    if init_logit is None:
        return final_logit.sigmoid()
    if final_logit is None:
        return init_logit.sigmoid()
    init_prob, final_prob = init_logit.sigmoid(), final_logit.sigmoid()
    return torch.sqrt(init_prob * final_prob)


class MultiClassMetric:
    def __init__(self, **kwargs):
        pass

    def calculate(self, out_logit, init_logit, final_logit, target, mask, time, **kwargs):
        # print("O:", out_logit.size())
        # print("I:", init_logit.size())
        # print("F:", final_logit.size())
        # print("T:", target.size())
        conf_prob = calc_inside_prob(init_logit, final_logit)
        bg_avg, fg_avg = calculate_background_score(conf_prob, target, mask)
        pred_time, gt_time, pred_pos, gt_pos = simple_multi_greedy(out_logit, target, mask, time)
        if pred_time is None or len(pred_time) == 0:
            return None
        metrics = interval_matching(pred_time, gt_time)
        prop_ap = calculate_average_precision(pred_time, gt_time)
        try:
            item_f1, item_precision, item_recall = calc_item_pr(pred_pos, gt_pos, mask)
        except Exception as e:
            print(pred_pos)
            print(gt_pos)
            raise e
        return {
            "metrics": {
                "bg_score": bg_avg,
                "fg_score": fg_avg,
                **metrics
            },
            "ap": prop_ap,
            "item": {
                "f1": item_f1, "precision": item_precision, "recall": item_recall
            }
        }

    def __call__(self, *input, **kwargs):
        new_input = [item[0] if item is not None and len(item) == 1 else item for item in input]
        new_kwargs = {key: (value[0] if value is not None and len(value) == 1 else value) for key, value in
                      kwargs.items()}
        return self.calculate(*new_input, **new_kwargs)


class DivideConquerMetric:
    def __init__(self, split_thresh, inside_thresh):
        self.split_thresh = split_thresh
        self.inside_thresh = inside_thresh

    def calculate(self, split_logit, init_logit, final_logit, target, mask, time, **kwargs):
        split_prob = split_logit.sigmoid()
        conf_prob = calc_inside_prob(init_logit, final_logit)
        bg_avg, fg_avg = calculate_background_score(conf_prob, target, mask)
        pred_time, gt_time, pred_pos, gt_pos, split_recall = simple_split_greedy(split_prob, conf_prob, target, mask, time,
                                                                                 self.split_thresh, self.inside_thresh)
        if pred_time is None or len(pred_time) == 0:
            return None
        metrics = interval_matching(pred_time, gt_time)
        # prop_f1, prop_precision, prop_recall = calc_pr_dict(pred_time, gt_time)
        prop_ap = calculate_average_precision(pred_time, gt_time)
        item_f1, item_precision, item_recall = calc_item_pr(pred_pos, gt_pos, mask)
        return {
            "metrics": {
                "bg_score": bg_avg,
                "fg_score": fg_avg,
                **metrics
            },
            "ap": prop_ap,
            "item": {
                "f1": item_f1, "precision": item_precision, "recall": item_recall
            }
        }

    def __call__(self, *input, **kwargs):
        new_input = [item[0] if item is not None and len(item) == 1 else item for item in input]
        new_kwargs = {key: (value[0] if value is not None and len(value) == 1 else value) for key, value in kwargs.items()}
        return self.calculate(*new_input, **new_kwargs)


class DPMetric:
    def __init__(self):
        super().__init__()


    def calculate(self, split_logit, init_logit, final_logit, target, mask, time, **kwargs):
        split_prob = split_logit.sigmoid()
        # start_prob, end_prob = start_logit.sigmoid(), end_logit.sigmoid()
        conf_prob = calc_inside_prob(init_logit, final_logit)
        bg_avg, fg_avg = calculate_background_score(conf_prob, target, mask)
        pred_time, gt_time, pred_pos, gt_pos, split_recall = dp_split(split_prob, conf_prob, target, mask, time)
        if pred_time is None or len(pred_time) == 0:
            return None
        metrics = interval_matching(pred_time, gt_time)
        # prop_f1, prop_precision, prop_recall = calc_pr_dict(pred_time, gt_time)
        prop_ap = calculate_average_precision(pred_time, gt_time)
        item_f1, item_precision, item_recall = calc_item_pr(pred_pos, gt_pos, mask)
        return {
            "metrics": {
                "bg_score": bg_avg,
                "fg_score": fg_avg,
                **metrics
            },
            "ap": prop_ap,
            "item": {
                "f1": item_f1, "precision": item_precision, "recall": item_recall
            }
        }

    def __call__(self, *input, **kwargs):
        new_input = [item[0] if item is not None and len(item) == 1 else item for item in input]
        new_kwargs = {key: (value[0] if value is not None and len(value) == 1 else value) for key, value in kwargs.items()}
        return self.calculate(*new_input, **new_kwargs)
