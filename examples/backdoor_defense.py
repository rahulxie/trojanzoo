#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --color --verbose 1 --attack badnet --defense neural_cleanse --validate_interval 1 --epochs 50 --lr 1e-2
import time

import numpy as np
import torch
from torch import nn
from torch.quantization import quantize_fx
import os, sys

os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.insert(0, "../")
import trojanvision
import argparse
from trojanzoo.utils.logger import AverageMeter





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    trojanvision.defenses.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
    defense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, **kwargs)
    if kwargs.get('est'):
        defense.est = True
    if kwargs['ptsq']:
        defense.ptsq = True
        if kwargs.get('q'):
            defense.q = kwargs['q']
        backend = "fbgemm"
        # backend = "qnnpack"
        defense.attack.model._model.eval()
        # prepare
        defense.attack.model._model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(defense.attack.model._model, inplace=True)
        # 加一个Calibrate
        defense.attack.model._validate(loader=defense.attack.dataset.loader['train'], **kwargs)
        # defense.attack.model._validate(loader=defense.attack.dataset.loader['valid'], **kwargs)
        torch.quantization.convert(defense.attack.model._model, inplace=True)
    elif kwargs.get('qat'):
        defense.qat = True
        if kwargs.get('q'):
            defense.q = kwargs['q']
        def accuracy(output, target, topk=(1,)):
            """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
                batch_size = target.size(0)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
                return res

        def evaluate(model, criterion, data_loader):
            model.eval()
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            cnt = 0
            with torch.no_grad():
                for image, target in data_loader:
                    output = model(image)
                    loss = criterion(output, target)
                    cnt += 1
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    print('.', end='')
                    top1.update(acc1[0], image.size(0))
                    top5.update(acc5[0], image.size(0))

            return top1, top5

        def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
            model.train()
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            avgloss = AverageMeter('Loss', '1.5f')

            cnt = 0
            for image, target in data_loader:
                start_time = time.time()
                print('.', end='')
                cnt += 1
                image, target = image.to(device), target.to(device)
                output = model(image)
                loss = criterion(output, target)
                loss.requires_grad_(True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))
                avgloss.update(loss, image.size(0))
                # if cnt >= ntrain_batches:
                #     print('Loss', avgloss.avg)
                #
                #     print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                #           .format(top1=top1, top5=top5))
                #     return

            print('Full train set:  * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

        optimizer = torch.optim.SGD(defense.attack.model._model.parameters(), lr = 0.0001)
        defense.attack.model._model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        defense.attack.model._model.train()
        torch.quantization.prepare_qat(defense.attack.model._model, inplace=True)

        num_train_batches = 50
        criterion = nn.CrossEntropyLoss()
        # QAT takes time and one needs to train over a few epochs.
        # Train and check accuracy after each epoch
        for nepoch in range(8):
            train_one_epoch(defense.attack.model._model, criterion, optimizer, defense.attack.dataset.loader['train'], torch.device('cpu'), num_train_batches)
            # if nepoch > 3:
            #     # Freeze quantizer parameters
            #     defense.attack.model._model.apply(torch.quantization.disable_observer)
            # if nepoch > 2:
            #     # Freeze batch norm mean and variance estimates
            #     defense.attack.model._model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            # Check the accuracy after each epoch
            quantized_model = torch.quantization.convert(defense.attack.model._model.eval(), inplace=False)
            quantized_model.eval()
            top1, top5 = evaluate(quantized_model, criterion, defense.attack.dataset.loader['valid'])
            print('Epoch %d :Evaluation accuracy: %f' % (
            nepoch, top1.avg))
        defense.attack.model._model = quantized_model

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)
    defense.detect(**trainer)