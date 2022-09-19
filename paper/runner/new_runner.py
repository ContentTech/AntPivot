import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import glob
import json
import argparse
from utils.logging import init_logger

import torch
from torch import optim
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from past.data_loader import FortuneData
from past.model import BiLstmCRF, BiLstmCRFV2
from past.utils_func import assert_rst_valid


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, help="mode", choices=['train', 'eval', 'eval_all'])
    parser.add_argument("-dataset", default='train', type=str, help="dataset name")
    parser.add_argument("-data_dir", default='data', type=str, help="dataset dir")
    parser.add_argument("-num_each_file", default=20, type=int, help="number examples in each data file")

    parser.add_argument("-model_name", default='bilstm', type=str, help="models name, [bilstm, bilstmv2]")
    parser.add_argument("-text_embd_dims", default=768, type=int, help="text embedding dims")
    parser.add_argument("-spk_num", default=3, type=int, help="spekear  dims")
    parser.add_argument("-input_dims", default=768, type=int, help="input dims")
    parser.add_argument("-hidden_size", default=768, type=int, help="hidden size")
    parser.add_argument("-num_layers", default=2, type=int, help="number layer ")
    parser.add_argument("-num_labels", default=3, type=int, help="num labels ")

    parser.add_argument("-train_from", default='', type=str, help="train from models")
    parser.add_argument("-test_from", default='', type=str, help="train from models")
    parser.add_argument("-test_num", default=20, type=int, help="test example numbers")
    parser.add_argument("-use_gpu", default=-1, type=int, help="train from models")

    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-lr_mode", default='multistep', type=str)
    parser.add_argument("-lr_step_size", default='26600,79800', type=str)
    parser.add_argument("-gamma", default=0.1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=5000, type=int)
    parser.add_argument("-t_mult", default=2, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-train_steps", default=400, type=int)
    parser.add_argument("-report_steps", default=5, type=int)
    parser.add_argument("-save_model_step", default=10, type=int)
    parser.add_argument("-save_dir", default='output', type=str)

    parser.add_argument("-loss_weight", default="0.05,1,1", type=str)
    parser.add_argument("-log_file", default="../logs/train.log", type=str)

    args = parser.parse_args()
    return args


def train(args):
    logger = init_logger(args.log_file, log_file_level=logging.DEBUG)
    logger.info(args)

    datasets = FortuneData(args.data_dir, args.dataset, args.num_each_file)
    logger.info('total {} examples'.format(datasets.get_num_examples()))

    dataloader = DataLoader(datasets, batch_size=1,
                            shuffle=False, num_workers=0, )
    device = "cpu" if args.use_gpu == -1 else "cuda"
    num_labels = args.num_labels
    if args.model_name == 'bilstm':
        model = BiLstmCRF(input_dims=args.input_dims,
                          hidden_size=args.hidden_size,
                          dropout=args.dropout,
                          num_labels=args.num_labels,
                          num_layers=args.num_layers,
                          device=device)
    elif args.model_name == 'bilstmv2':
        model = BiLstmCRFV2(input_dims=args.input_dims,
                            text_embd_dims=args.text_embd_dims,
                            hidden_size=args.hidden_size,
                            dropout=args.dropout,
                            num_labels=args.num_labels,
                            num_layers=args.num_layers,
                            device=device)
    else:
        raise ValueError('models name is not supported!')
    logger.info(model)
    # build optim
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(args.beta1, args.beta2),
                               )
    else:
        raise ValueError("unspport optim method!")
    if args.lr_mode == 'multistep':
        args.lr_step_size = [int(i) for i in args.lr_step_size.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step_size)
    elif args.lr_mode == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warmup_steps,
                                                                            T_mult=args.t_mult)
    else:
        raise ValueError("unspport lr models  method!")
    # loss_weight = [float(i) for i in args.loss_weight.split(',')]
    # loss_weight = torch.Tensor(loss_weight).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='none', weight=loss_weight)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    record_writer = SummaryWriter(args.save_dir + datetime.now().strftime("/events-%b-%d_%H-%M-%S"))
    cnt = 0
    while cnt < args.train_steps:
        for idx, batch in enumerate(dataloader):
            cnt += 1
            optimizer.zero_grad()
            text_embedding, spk_info, labels = batch
            text_embedding = text_embedding.to(device=device)
            spk_info = spk_info.to(device=device)
            labels = labels.to(device=device)
            loss = model(text_embedding, spk_info, labels.squeeze(0))
            loss = loss.sum()
            (loss / loss.numel()).backward()
            optimizer.step()
            lr_scheduler.step()
            if cnt % args.report_steps == 0:
                logger.info('step: {}, loss: {}'.format(cnt, float(loss._data.cpu().numpy())))
                logger.info('step: {}, lr: {}'.format(cnt, lr_scheduler.get_last_lr()[0]))
                record_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], cnt)
                record_writer.add_scalar('loss', float(loss._data.cpu().numpy()), cnt)

            if cnt % args.save_model_step == 0:
                save_model_name = os.path.join(args.save_dir, 'step_{}.pt'.format(cnt))
                torch.save(model.state_dict(), save_model_name)

            if cnt >= args.train_steps:
                break
    save_model_name = os.path.join(args.save_dir, 'step_{}.pt'.format(cnt))
    torch.save(model.state_dict(), save_model_name)


def eval_all(args):
    logger = init_logger(args.log_file, log_file_level=logging.INFO)
    logger.info(args)

    print(args)
    device = "cpu" if args.use_gpu == -1 else "gpu"
    num_labels = args.num_labels
    if args.model_name == 'bilstm':
        model = BiLstmCRF(input_dims=args.input_dims,
                          hidden_size=args.hidden_size,
                          dropout=args.dropout,
                          num_labels=args.num_labels,
                          num_layers=args.num_layers,
                          device=device)
    elif args.model_name == 'bilstmv2':
        model = BiLstmCRFV2(input_dims=args.input_dims,
                            text_embd_dims=args.text_embd_dims,
                            hidden_size=args.hidden_size,
                            dropout=args.dropout,
                            num_labels=args.num_labels,
                            num_layers=args.num_layers,
                            device=device)
    else:
        raise ValueError('models name is not supported!')
    logger.info(model.transitions)
    logger.info(model)
    model_name_list = glob.glob(os.path.join(args.test_from, 'step_*.pt'))
    logger.info('number of models {}'.format(len(model_name_list)))
    best_acc = 0
    best_model_name = ""
    for name in model_name_list:
        logger.info('models name: {}'.format(name))

        model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage),
                              strict=True)
        model.eval()

        datasets = FortuneData(args.data_dir, args.dataset, args.num_each_file, is_test=True)
        dataloader = DataLoader(datasets, batch_size=1,
                                shuffle=False, num_workers=0, )
        cnt_total = 0
        cnt_acc = 0
        loss_total = 0
        num_examples = 0
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                text_embedding, spk_info, labels = batch
                score, predict_label = model.predict(text_embedding, spk_info)
                # predict_label = post_rst_min_interval(predict_label, score)
                if assert_rst_valid(predict_label, end_id=args.num_labels - 1):
                    logger.info('start and end is equal')
                else:
                    logger.info('**** start and end is not equal *****')
                    logger.info('pred label')
                    logger.info(predict_label)
                    logger.info('gt label')
                    logger.info(labels)

                num_examples += 1
                predict_label = torch.Tensor(predict_label).long().view(1, -1)
                cur_cnt_acc = (labels == predict_label).sum()
                cur_cnt_total = predict_label.numel()
                logger.debug('example {}: acc: {}( {}/{} )'.format(idx, cur_cnt_acc * 1.0 / cur_cnt_total, cur_cnt_acc,
                                                                   cur_cnt_total))
                cnt_acc += cur_cnt_acc
                cnt_total += cur_cnt_total
                # abcd acc
                if idx % args.report_steps == 0:
                    logger.debug('pred labels')
                    logger.debug(predict_label)
                    logger.debug('gt label')
                    logger.debug(labels)
                if num_examples >= args.test_num:
                    break
            total_acc = cnt_acc * 1.0 / cnt_total
            logger.info('acc:{}, ({}/{})'.format(cnt_acc * 1.0 / cnt_total, cnt_acc, cnt_total))
            if total_acc > best_acc:
                best_acc = total_acc
                best_model_name = name
    logging.info('best models name: {}, acc:{}'.format(best_model_name, best_acc))
    args.test_from = best_model_name

    eval(args)


def eval(args):
    logger = init_logger(args.log_file, log_file_level=logging.INFO)
    logger.info(args)

    # dataset_path = os.path.join(args.data_dir, args.dataset + '.0.pt')
    datasets = FortuneData(args.data_dir, args.dataset, args.num_each_file, is_test=True)
    dataloader = DataLoader(datasets, batch_size=1,
                            shuffle=False, num_workers=0, )
    device = "cpu" if args.use_gpu == -1 else "gpu"
    num_labels = args.num_labels
    if args.model_name == 'bilstm':
        model = BiLstmCRF(input_dims=args.input_dims,
                          hidden_size=args.hidden_size,
                          dropout=args.dropout,
                          num_labels=args.num_labels,
                          num_layers=args.num_layers,
                          device=device)
    elif args.model_name == 'bilstmv2':
        model = BiLstmCRFV2(input_dims=args.input_dims,
                            text_embd_dims=args.text_embd_dims,
                            hidden_size=args.hidden_size,
                            dropout=args.dropout,
                            num_labels=args.num_labels,
                            num_layers=args.num_layers,
                            device=device)
    else:
        raise ValueError('models name is not supported!')
    logger.info(model.transitions)
    logger.info(model)
    # load models
    model.load_state_dict(torch.load(args.test_from, map_location=lambda storage, loc: storage),
                          strict=True)
    model.eval()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cnt_total = 0
    cnt_acc = 0
    loss_total = 0
    num_examples = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            text_embedding, spk_info, labels = batch
            score, predict_label = model.predict(text_embedding, spk_info)
            # predict_label = post_rst_min_interval(predict_label, score)
            if assert_rst_valid(predict_label, end_id=args.num_labels - 1):
                logger.debug('start and end is equal')
            else:
                logger.info('**** start and end is not equal *****')
                logger.info('pred label')
                logger.info(predict_label)
                logger.info('gt label')
                logger.info(labels)
            # save label
            save_name = os.path.join(args.save_dir, str(num_examples) + '.json')
            json.dump(predict_label, open(save_name, 'w'))

            num_examples += 1
            predict_label = torch.Tensor(predict_label).long().view(1, -1)
            cur_cnt_acc = (labels == predict_label).sum()
            cur_cnt_total = predict_label.numel()
            logger.info(
                'example acc: {}( {}/{} )'.format(cur_cnt_acc * 1.0 / cur_cnt_total, cur_cnt_acc, cur_cnt_total))
            cnt_acc += cur_cnt_acc
            cnt_total += cur_cnt_total
            # abcd acc
            if idx % args.report_steps == 0:
                logger.info('pred labels')
                logger.info(predict_label)
                logger.info('gt label')
                logger.info(labels)
            if num_examples >= args.test_num:
                break
    logger.info('acc:{}, ({}/{})'.format(cnt_acc * 1.0 / cnt_total, cnt_acc, cnt_total))


def main():
    args = arg_parse()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'eval_all':
        eval_all(args)
    else:
        raise ValueError('unknown mode {}'.format(args.mode))


if __name__ == '__main__':
    main()


