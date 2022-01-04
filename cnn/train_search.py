import os
import sys
import time
import glob
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

import utils
from architect import Architect
from model_search import Network

CIFAR_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    args = parser.parse_args()

    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    return args


def main(args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_loader, val_loader, model, architect, criterion, optimizer, lr, **vars(args))
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(val_loader, model, criterion, **vars(args))
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_loader, val_loader, model, architect, criterion, optimizer, lr, unrolled, grad_clip, report_freq, **kwargs):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with logging_redirect_tqdm():
        with tqdm(train_loader) as tbar:
            for step, (inputs, targets) in enumerate(tbar):
                inputs, targets = inputs.cuda(), targets.cuda()
                model.train()
                n = inputs.size(0)

                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(val_loader))
                input_search, target_search = input_search.cuda(), target_search.cuda()

                architect.step(inputs, targets, input_search, target_search, lr, optimizer, unrolled=unrolled)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                tbar.set_postfix({'loss': loss.item(), 'top1': prec1.item(), 'top5': prec5.item()})
                if step % report_freq == 0:
                    logging.info('train %03d loss_avg=%e top1_avg=%f top5_avg=%f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(val_loader, model, criterion, report_freq, **kwargs):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with logging_redirect_tqdm():
        with tqdm(val_loader) as tbar:
            for step, (inputs, targets) in enumerate(tbar):
                inputs, targets = inputs.cuda(), targets.cuda()

                logits = model(inputs)
                loss = criterion(logits, targets)

                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                tbar.set_postfix({'loss': loss.item(), 'top1': prec1.item(), 'top5': prec5.item()})
                if step % report_freq == 0:
                    logging.info('valid %03d loss_avg=%e top1_avg=%f top5_avg=%f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    args_ = parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args_.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    main(args_)
