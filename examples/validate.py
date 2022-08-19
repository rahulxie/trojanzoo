#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/validate.py --color --verbose 1 --dataset cifar10 --pretrained --model resnet18_comp
"""  # noqa: E501

import os, sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.insert(0, "../")
import trojanvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model)
    acc, loss = model._validate()
