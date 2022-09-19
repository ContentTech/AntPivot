import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.new_dataset import BatchFortuneData, collate_data
from models.loss import FinalLoss
from models.metrics import calculate_f1, DPMetric, DivideConquerMetric, MultiClassMetric
from models.model import MainModel
# from models.past_model import MainModel
from runner.optimizer import GradualWarmupScheduler
from utils.container import metricsContainer
from utils.helper import move_to_cuda, move_to_cpu
from utils.timer import Timer


def norm_shift(num_list):
    def map_val(x, base):
        if x >= 0:
            return 2 * base ** (-x) / (1 + base ** (-x))
        else:
            return 2 * base ** x / (1 + base ** x)

    base = 1.1
    num_list = [map_val(x, base) for x in num_list]
    return num_list


class MainRunner:
    def __init__(self, config):
        print("Initialization Start.")
        self.config = config
        self._init_misc()
        self._init_dataset(config.dataset)
        self._init_model(config.model, config.loss, config.metric)
        self._init_optimizer(config.optimizer)
        print("Initialization End.")

    def _init_misc(self):
        seed = 8
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(self.config)
        self.model_saved_path = self.config["saved_path"]
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        print('GPU: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        self.initial_epoch = 0

    def _init_dataset(self, config):
        train_dataset = BatchFortuneData(config.data_dir, "train")
        eval_dataset = BatchFortuneData(config.data_dir, "eval")
        test_dataset = BatchFortuneData(config.data_dir, "test")
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8,
                                       collate_fn=collate_data(**config),
                                       pin_memory=False)
        self.eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=8,
                                      collate_fn=collate_data(**config),
                                      pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                      collate_fn=collate_data(**config),
                                      pin_memory=False)

    def _init_model(self, model_config, loss_config, metric_config):
        self.model = MainModel(**model_config).cuda()
        self.loss = FinalLoss(loss_config)
        self.metric = DPMetric()

    def _init_optimizer(self, config):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"],
                                           weight_decay=config["weight_decay"])

        self.sub_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["T_max"])
        self.main_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1,
                                                     total_epoch=config["warmup_epoch"],
                                                     after_scheduler=self.sub_scheduler)

    def _train_one_epoch(self, epoch, last_total_step):
        self.model.train()
        timer = Timer()
        batch_idx = 0
        total_time = []
        for batch_idx, data in enumerate(self.train_loader, 1):
            timer.reset()
            self.optimizer.zero_grad()
            batch_input = move_to_cuda(data["inputs"])
            target = move_to_cuda(data["target"])
            start_time = time.time()
            output = self.model(**batch_input)
            loss, loss_items = self.loss(**output, **target, mask=batch_input["mask"])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.models.parameters(), 10)

            # update
            self.optimizer.step()
            self.main_scheduler.step(epoch + batch_idx / len(self.train_loader))
            end_time = time.time()
            total_time.append(end_time - start_time)
            curr_lr = self.main_scheduler.get_last_lr()[0]
            time_interval = timer.elapsed_interval
            metricsContainer.update("loss", loss_items)
            metricsContainer.update("train_time", time_interval)

            if batch_idx % self.config.display_interval == 0:
                self._export_log(epoch, last_total_step + batch_idx, batch_idx, curr_lr,
                                 metricsContainer.calculate_average("loss"),
                                 metricsContainer.calculate_average("train_time"))

        if batch_idx % self.config.display_interval == 0:
            self._export_log(epoch, last_total_step + batch_idx, batch_idx,
                             self.main_scheduler.get_last_lr()[0],
                             metricsContainer.calculate_average("loss"),
                             metricsContainer.calculate_average("train_time"))
        print("Train Avg. Time: ", sum(total_time) / len(total_time))
        return batch_idx + last_total_step

    def _export_log(self, epoch, total_step, batch_idx, lr, loss_meter, time_meter):
        msg = 'Epoch {}, Batch ({} / {}), lr = {:.5f}, '.format(epoch, batch_idx,
                                                                len(self.train_loader), lr)
        for k, v in loss_meter.items():
            msg += '{} = {:.4f}, '.format(k, v)
        remaining = len(self.train_loader) - batch_idx
        msg += '{:.3f} s/batch ({}s left)'.format(time_meter, int(time_meter * remaining))
        print(msg + "\n")
        sys.stdout.flush()
        loss_meter.update({"epoch": epoch, "batch": total_step, "lr": lr})

    def save_model(self, path, epoch):
        state_dict = {
            'epoch': epoch,
            'config': self.config,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        print('save models to {}, epoch {}.'.format(path, epoch))

    def load_model(self, path):
        state_dict = torch.load(path)
        self.initial_epoch = state_dict['epoch']
        self.main_scheduler.step(self.initial_epoch)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        print('load models from {}, epoch {}.'.format(path, self.initial_epoch))

    def train(self, start_epoch=0):
        best_result, best_criterion = (), -float('inf')
        total_step = start_epoch * len(self.train_loader)
        for epoch in range(start_epoch, self.config["max_epoch"] + 1):
            saved_path = os.path.join(self.model_saved_path, 'models-{}.pt'.format(epoch))
            total_step = self._train_one_epoch(epoch, total_step)
            self.save_model(saved_path, epoch)
            print("Eval for Eval dataset.")
            eval_results = self.eval(epoch, "eval")
            print("Eval for Test dataset.")
            test_results = self.eval(epoch, "test")
            if eval_results is None or test_results is None:
                continue
            if eval_results[0] > best_criterion:
                best_result = test_results[1:]
                best_criterion = eval_results[0]
            print('=' * 60)
        print('-' * 120)
        print('Done.')
        print("Best Result:")
        output_metrics(*best_result)

    def _print_metrics(self, epoch, namespace, metrics, action):
        msg = "{} Epoch {}, {}: ".format(action, epoch, namespace)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                msg += '{} = {:.4f} | '.format(k, v)
        else:
            msg += str(metrics)
        print(msg)
        sys.stdout.flush()
        # metrics.update({"epoch": epoch})

    def eval(self, epoch, data):
        valid_mask = False
        total_time = []
        invalid_idx = []
        if data == "eval":
            data_loader = self.eval_loader
        elif data == "test":
            data_loader = self.test_loader
        else:
            raise NotImplementedError
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader, 1):
                net_input = move_to_cuda(batch['inputs'])
                start_time = time.time()
                output = self.model(**net_input)
                target = move_to_cuda(batch["target"])
                loss, _ = self.loss(**output, **target, mask=net_input["mask"])
                metrics = self.metric(**move_to_cpu(output),
                                      target=batch["target"]["tags"],
                                      mask=batch["inputs"]["mask"],
                                      time=batch["raw"]
                                      )
                end_time = time.time()
                total_time.append(end_time - start_time)
                if metrics is None:
                    invalid_idx.append(batch_idx)
                    continue
                metrics["metrics"]["loss"] = loss.item()
                # print(metrics)
                valid_mask = True
                for key, metric in metrics.items():
                    metricsContainer.update(key, metric)
        print("No valid results for Iteration: ", invalid_idx)
        if not valid_mask:
            return None
        accum = metricsContainer.get("metrics")
        avg = metricsContainer.calculate_average("metrics", reset=True)
        # precision = metricsContainer.calculate_average("precision")
        # recall = metricsContainer.calculate_average("recall")
        # f1 = metricsContainer.calculate_average("f1")
        ap = metricsContainer.calculate_average("ap")
        item = metricsContainer.calculate_average("item")
        output_metrics(epoch, avg, accum, ap, item)
        criterion = 0.5 * ap["0.5"] + 0.6 * ap["0.6"] + 0.7 * ap["0.7"] + 0.8 * ap["0.8"] + 0.9 * ap["0.9"]
        results = (criterion, epoch, avg, accum, ap, item)
        # results = (-avg["loss"], epoch, avg, accum, ap, item)
        print("Eval Avg. Time: ", sum(total_time) / len(total_time))
        return results


def table_output(name, metrics):
    msg = name + ": "
    for key, value in metrics.items():
        msg += "IoU@{} = {:.3f} |".format(key, value)
    print(msg + "\n")


def output_metrics(epoch, avg, accum, ap, item):
    print("Epoch {} - 验证结果".format(epoch))
    all_start_shift, all_end_shift = accum["start_shift"], accum["end_shift"]
    avg_start_shift, avg_end_shift = avg["start_shift"], avg["end_shift"]
    avg_iou, avg_pred_len, avg_gt_len = avg["overall_iou"], avg["pred_len"], avg["gt_len"]
    avg_pred_num, avg_gt_num = avg["pred_num"], avg["gt_num"]
    total_pred, total_gt = accum["pred_num"], accum["gt_num"]
    start_correct, end_correct = accum["start_correct"], accum["end_correct"]
    start_f1, start_recall, start_precision = calculate_f1(start_correct, total_pred, total_gt)
    end_f1, end_recall, end_precision = calculate_f1(end_correct, total_pred, total_gt)
    norm_start_median = norm_shift([np.median(all_start_shift)])[0]
    norm_end_median = norm_shift([np.median(all_end_shift)])[0]
    # table_output("Precision", mp)
    # table_output("Recall", mr)
    # table_output("F1", mf1)
    table_output("Average Precision", ap)
    print("损失函数值：{:.3f}".format(avg["loss"]))
    print("语句级别F1: {:.3f}，召回率：{:.3f}，精确度：{:.3f}".format(item["f1"], item["recall"], item["precision"]))
    # if "split_recall" in avg:
    #     print("分割点平均召回率: {:.3f}".format(avg["split_recall"]))
    if "pair_score" in avg:
        print("目标首尾语义相似度: {:.3f}".format(avg["pair_score"]))
        print("随机配对首尾相似度: {:.3f}".format(avg["avg_score"]))
    print("背景平均得分: {:.3f}".format(avg["bg_score"]))
    print("前景平均得分: {:.3f}".format(avg["fg_score"]))
    print("块平均数量: {:d} / {:d}".format(int(avg_pred_num), int(avg_gt_num)))
    print("块平均时长: {:.3f}s / {:.3f}s".format(avg_pred_len, avg_gt_len))
    print("预测块平均IOU: {:.3f}".format(avg_iou))
    # print("起始位置中位数偏移时长: {:.3f}s".format(np.median(all_start_shift)))
    # print("起始位置平均偏移时长: {:.3f}s".format(avg_start_shift))
    # print("结束位置中位数偏移时长: {:.3f}s".format(np.median(all_end_shift)))
    # print("结束位置平均偏移时长: {:.3f}s".format(avg_end_shift))
    # print("---------------------------------------------------------------------------------")
    print("A. 起始位置偏移五秒内得分: {:.3f} (Recall: {:.3f}  Precision: {:.3f})".format(start_f1, start_recall, start_precision))
    print("B. 结束位置偏移五秒内得分: {:.3f} (Recall: {:.3f}  Precision: {:.3f})".format(end_f1, end_recall, end_precision))
    # print("C. 起始位置偏移中位数得分: {:.3f}".format(norm_start_median))
    # print("D. 结束位置偏移中位数得分: {:.3f}".format(norm_end_median))
    score = 4 * start_f1 + 3 * end_f1 + 2 * norm_start_median + 1 * norm_end_median
    print("最终加权得分[4*A+3*B+2*C+1*D]: {:.3f}".format(score))
    print("*********************************************************************************")

def ctor(sizes):
    return {
        "sent_emb": torch.ones(*(sizes[0])).cuda(),
        "spk": torch.ones(*(sizes[1])).long().cuda(),
        "mask": torch.ones(*(sizes[2])).long().cuda(),
        "mel":torch.ones(*(sizes[3])).cuda()
    }