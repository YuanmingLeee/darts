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

import genotypes
import utils
from model_search import Network

from model import NetworkCIFAR as Network


def parse_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    args = parser.parse_args()

    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    return args

CIFAR_CLASSES = 10


def main(args):

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = getattr(genotypes, args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_loader, model, criterion, optimizer, epoch=epoch, **vars(args))
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(val_loader, model, criterion, epoch=epoch, **vars(args))
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_loader, model, criterion, optimizer, auxiliary, grad_clip, report_freq, epoch, **kwargs):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    with logging_redirect_tqdm():
        with tqdm(train_loader, desc=f'train {epoch}') as tbar:
            for step, (inputs, targets) in enumerate(tbar):
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                logits, logits_aux = model(inputs)
                loss = criterion(logits, targets)
                if auxiliary:
                    loss_aux = criterion(logits_aux, targets)
                    loss += kwargs['auxiliary_weight'] * loss_aux
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), grad_clip)
                optimizer.step()

                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                tbar.set_postfix({'loss': loss.item(), 'top1': prec1.item(), 'top5': prec5.item()})
                if step % report_freq == 0:
                    logging.info('train %03d loss_avg=%e top1_avg=%f top5_avg=%f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(val_loader, model, criterion, report_freq, epoch, **kwargs):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with logging_redirect_tqdm():
        with tqdm(val_loader, desc=f'Val  {epoch}') as tbar:
            for step, (inputs, targets) in enumerate(tbar):
                inputs, targets = inputs.cuda(), targets.cuda()

                logits, _ = model(inputs)
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
