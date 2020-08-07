from collections import deque
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from .custom_transforms import *
from optims import *
from PIL import Image
import cv2
import numpy as np
from custom_models import *
import os
import glob
from config import *
import datetime
import re
import math
import GPUtil


__all__ = ['AverageMeter', 'Timer',
           'get_optim', 'get_model', 'get_transforms',
           'get_scripts', 'get_loss_weights',
           'calc_accuracy', 'print_precision', 'get_combined_npy',
           'save_ckpt', 'load_ckpt', 'get_available_gpu_ids',
           'print_network_info']


class AverageMeter:

    def __init__(self, length=None):
        self.length = length
        if self.length is not None:
            self.list = deque()
        else:
            self.sum = 0
            self.count = 0
        self.current = 0

    def update(self, value):
        if self.length:
            if len(self.list) >= self.length:
                self.list.popleft()
            self.list.append(value)
            self.current = sum(self.list) / len(self.list)
        else:
            self.sum += value
            self.count += 1
            self.current = self.sum / self.count
        return self.current

    def __str__(self):
        return str(self.current)


class Timer:

    def __init__(self, print_info=None):
        self.current = None
        self.print_info = print_info

    def tic(self):
        self.current = time.time()

    def pause(self):
        self.elapsed = time.time() - self.current
        self.current = None

    def resume(self):
        self.current = time.time() - self.elapsed

    def toc(self):
        assert self.current is not None
        delta = time.time() - self.current

        if self.print_info is not None:
            print(self.print_info.format(delta))

        return delta


def get_optim(opt, model, lr=0.01):
    if opt == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    # e.g. lookahead+adam
    elif opt.startswith('lookahead'):
        optimizer = get_optim(opt.split('+')[-1], model, lr)
        return Lookahead(optimizer)


def get_model(model, num_classes, pretrained=True):
    model = eval(f'{model}({num_classes}, {pretrained})')
    return model


def get_transforms(type):
    """
    transforms for the dataset
        square Q
        plain P
        content_plain CP
        plain+ P+
    """
    if type in ['square', 'Q']:
        return transforms.Compose([
            transforms.Resize((config['input_size'], config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(**norms['common'])
        ])
    elif type in ['plain', 'P']:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize(config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(1, 1, 1))
        ])
    elif type in ['content_plain', 'CP']:
        return transforms.Compose([
            ContentSquarePad(),
            transforms.Resize(config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(1, 1, 1))
        ])
    elif type in ['plain+', 'P+']:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize(config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(**norms['imagenet'])
        ])


def get_scripts(index=None):
    """
    get the scripts
    """
    if not index:
        return script_list[config['scripts']]
    else:
        return script_list[config['scripts']][index]


def get_loss_weights(dataset):
    return script_weights[dataset]


def calc_accuracy(result_map):
    overall = np.sum(result_map[i, i] for i in range(
        len(result_map))) / np.sum(result_map)
    mean = np.mean([result_map[i, i] / np.sum(result_map[i])
                    for i in range(len(result_map))])

    return overall, mean


def print_precision(result_map, scripts):
    """
    print the confusion matrix and the accuracy
    """

    def print_line(iterable, index, width=6, last_width=8, newline=False):
        for item in iterable:
            print(f'{item:>6}', end='')
        if newline:
            print()

    print(' ' * 3, end='')
    print_line([title[:3] for title in scripts], -1)
    print('   Accu.')

    for index, line in enumerate(result_map):
        print(f'{scripts[index][:3]}', end='')
        print_line(line, index)
        accu = line[index] / np.sum(line)
        print(f'{accu:9.5f}')


def get_combined_npy(npy_list, verbose=True):
    assert npy_list  # npy list 不为空
    if len(npy_list) == 1:
        file = os.path.join(config['data_dir'], npy_list[0] + '.npy')
        if verbose:
            print(f'loading {file}')
        return np.load(file)
    else:
        label_lists = []
        for npy in npy_list:
            file = os.path.join(config['data_dir'], npy + '.npy')
            if verbose:
                print(f'loading {file}')
            labels = np.load(file)
            label_lists.append(labels)
        return np.concatenate(label_lists, axis=0)


def save_ckpt(folder, model, args, info, file=None):
    ckpt = {
        'model': model.module.state_dict(),
        'info': info,
        'config': config,
        'params': {
            'model': args.model
        }
    }
    if not file:
        file = datetime.datetime.today().strftime('%b-%d-%I_%M%p')
    filename = os.path.join(folder, file + '.pt')
    print(f'model saved as {filename}')
    torch.save(ckpt, filename)


def load_ckpt(file):
    print(f'loading model from {file}')
    ckpt = torch.load(file, map_location=torch.device('cpu'))
    config.update(ckpt['config'])
    return ckpt


def get_available_gpu_ids(count):
    return GPUtil.getAvailable(limit=count, maxLoad=1.0, maxMemory=.01)


def print_network_info(args):
    print('========================================')
    print('Basic Information:')
    print('------------------------------')
    print(f'gpu ids: {args.gpu_ids}')
    print(f'model: {args.model}, pretrained: {not args.scratch}')
    print('scripts: ' + ', '.join([t[:3] for t in get_scripts()]))
    print(f'checkpoint path: {args.ckpt_file}')
    print(
        f'input size: {config["input_size"]}, transform: {", ".join(args.transforms)}')
    print(f'steps: {" -> ".join(map(str, args.steps))}')
    print(f'optimizer: {args.optim}')
    if args.loss != 'ce':
        print(f'loss function: {args.loss}')
    print('========================================')
