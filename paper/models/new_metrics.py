import numpy as np
import torch

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

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


def calc_pr_dict(predictions, targets):
    """
    :param predictions: FloatTensor, time, in (L_p, 2)
    :param targets: FloatTensor, time, in (L_t, 2)
    :return:
    """
    pred_num, gt_num = len(predictions), len(targets)
    iou_threshold = np.arange(0.7, 0.95, 0.05)
    predictions = predictions.view(-1, 1, 2).repeat(1, gt_num, 1)
    targets = targets.view(1, -1, 2).repeat(pred_num, 1, 1)
    iou, _, _ = calculate_iou1d(*predictions.unbind(-1), *targets.unbind(-1))  # [L_p, L_t]

    best_gt = iou.max(dim=0)[0]  # L_t
    best_pred = iou.max(dim=1)[0]  # L_p
    precision_dict, recall_dict, f1_dict = {}, {}, {}
    for thres in iou_threshold:
        pred_correct = best_pred >= thres
        gt_correct = best_gt >= thres
        precision = (pred_correct.sum() / pred_num).item()
        recall = (gt_correct.sum() / gt_num).item()
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall != 0) else 0
        precision_dict["{:.2f}".format(thres)] = precision
        recall_dict["{:.2f}".format(thres)] = recall
        f1_dict["{:.2f}".format(thres)] = f1
    return f1_dict, precision_dict, recall_dict


def calc_item_pr(predictions, targets, mask):
    max_len = mask.sum(-1)
    pred_seq, gt_seq = torch.zeros(max_len), torch.zeros(max_len)
    for pred_item in predictions:
        start, end = pred_item
        for frame_idx in range(start, end + 1):
            pred_seq[frame_idx] = 1
    for gt_item in targets:
        start, end = gt_item
        for frame_idx in range(start, end + 1):
            gt_seq[frame_idx] = 1
    false_pos = torch.logical_and(pred_seq == 1, gt_seq == 0).sum().item()
    true_pos = torch.logical_and(pred_seq == 1, gt_seq == 1).sum().item()
    false_neg = torch.logical_and(pred_seq == 0, gt_seq == 1).sum().item()
    # true_neg = torch.logical_and(pred_seq == 0, gt_seq == 0).sum().item()
    if (true_pos + false_pos == 0):
        precision, recall = 0, 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
    if true_pos != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return f1, precision, recall


def calculate_average_precision(predictions, targets):
    """
    :param predictions: FloatTensor, time, in (L_p, 2)
    :param targets: FloatTensor, time, in (L_t, 2)
    :return:
    """
    pred_num, gt_num = len(predictions), len(targets)
    iou_threshold = np.arange(0.5, 0.95, 0.1)
    ap = {}
    predictions = predictions.view(-1, 1, 2).repeat(1, gt_num, 1)
    targets = targets.view(1, -1, 2).repeat(pred_num, 1, 1)
    iou, _, _ = calculate_iou1d(*predictions.unbind(-1), *targets.unbind(-1))  # [L_p, L_t]
    tp = torch.zeros(len(iou_threshold), pred_num)
    fp = torch.zeros(len(iou_threshold), pred_num)
    pred_order = iou.max(-1)[0].argsort(-1, descending=True)  # [L_p], 按最大iou降序排序
    assignment = torch.empty(len(iou_threshold), gt_num).fill_(-1)
    for thres_i, thres in enumerate(iou_threshold):
        for pred_idx, cnt_pred in enumerate(pred_order):
            cnt_iou, cnt_gt_order = iou[cnt_pred].sort(-1, descending=True)  # [L_t]
            for gt_idx in range(gt_num):
                if cnt_iou[gt_idx] < thres:
                    fp[thres_i, pred_idx] = 1
                    break
                if assignment[thres_i, cnt_gt_order[gt_idx]] != -1:
                    continue
                tp[thres_i, pred_idx] = 1
                assignment[thres_i, cnt_gt_order[gt_idx]] = cnt_pred
                break
            if fp[thres_i, pred_idx] == 0 and tp[thres_i, pred_idx] == 0:
                fp[thres_i, pred_idx] = 1
    tp_cumsum = torch.cumsum(tp, dim=1).float()
    fp_cumsum = torch.cumsum(fp, dim=1).float()
    recall_cumsum = tp_cumsum / gt_num
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for thres_i, thres in enumerate(iou_threshold):
        ap["{:.1f}".format(thres)] = interpolated_prec_rec(precision_cumsum[thres_i, :], recall_cumsum[thres_i, :])
    return ap






