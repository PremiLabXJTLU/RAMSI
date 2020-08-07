import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
from utils import *
from torch.optim import lr_scheduler
from arguments import *
import numpy as np
import os
import warnings
from config import *
from custom_models import focal_loss
import math
import time
import datetime

# train with specific GPU ids
if args.gpu_ids is None:
    args.gpu_ids = get_available_gpu_ids(args.gpu_count)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
# hide the warnings
warnings.simplefilter('ignore')

# the precise date when the program starts. looks like Nov-1-2_43_56PM
start_date = datetime.datetime.today().strftime('%b-%d-%I_%M_%S%p')


def train_model(model, dataloaders):
    if args.weighted is not None:
        weight = torch.tensor(get_loss_weights(
            args.weighted), dtype=torch.float).cuda()
    else:
        weight = None
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif args.loss == 'focal':
        criterion = focal_loss.FocalLoss(weight=weight)
    running = AverageMeter(config['frequency']['averagemeter'])
    optimizer = get_optim(args.optim, model, args.lr)
    start_iter = 0

    if len(args.steps) == 1:
        scheduler = None
    else:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=args.steps[:-1], last_epoch=start_iter-1)

    total_iter = 0
    epoch = 0

    best_accu = 0.0
    best_result_map = None

    timer = Timer()
    timer.tic()

    # start training
    while total_iter < args.steps[-1]:
        epoch += 1
        print(f'>>>>>>>>>>>>>> Epoch {epoch:} <<<<<<<<<<<<<<')

        model.train()
        iters = math.ceil(
            len(dataloaders['train'].dataset) / args.batch_size)

        prefetcher = DataPrefetcher(dataloaders['train'])

        inputs, labels = prefetcher.next()

        while inputs is not None:

            total_iter += 1

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running.update(loss.item())

            if total_iter % config['frequency']['printloss'] == 0:
                # get the current learning rate from the optimizer
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'iter: {total_iter}', end=', ')
                print(f'loss: {running.current:.5f}, lr: {lr:.2e}', end=', ')
                print(f'time: {timer.toc():.2f}s')
                timer.tic()

            if total_iter % config['frequency']['validate'] == 0:

                timer.pause()

                current_accu, result_map = validate_and_save_best(
                    model, dataloaders, best_accu, args.ckpt_file)
                if current_accu > best_accu:
                    best_accu = current_accu
                    best_result_map = result_map

                model.train()

                timer.resume()

            inputs, labels = prefetcher.next()

    # last evaluation when the training finishes
    validate_and_save_best(model, dataloaders, best_accu, args.ckpt_file)

    print(f'-' * 30)
    print(f'overall best result:')
    print(f'mean average precision: {best_accu:.6f}')
    print_precision(best_result_map, get_scripts())
    print(f'model saved as {args.ckpt_file}')


def validate_and_save_best(model, dataloaders, best_accu, ckpt_file):
    """
    validate, and if the performance is the current best result, save the model parameters
    """
    current_accu, result_map = validate(model, dataloaders)
    if current_accu > best_accu or not args.best_only:
        save_ckpt(config['ckpt_dir'], model, args, {
                  'accu': current_accu, 'map': result_map}, file=ckpt_file)
        print_precision(result_map, get_scripts())
    else:
        print(
            f'current accuracy is lower than the best ({best_accu:.6f}). ckpt discarded.')
    return current_accu, result_map


def validate(model, dataloaders, *, log_false=False):
    """
    validate using the current model parameters
    """
    model.eval()
    with torch.no_grad():
        if log_false:
            logger = open('test_log.txt', 'w', encoding='utf-8')

        correct = 0
        result_mat = np.zeros(
            (len(get_scripts()), len(get_scripts())), dtype=np.int32)
        index = 0
        for ims, labels in tqdm(dataloaders['val'], ascii=True):
            ims = ims.cuda()
            outputs = model(ims)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu()
            correct += torch.sum(preds == labels.data)
            for i, (l, p) in enumerate(zip(labels.numpy(), preds.numpy())):
                result_mat[l, p] += 1
                if log_false and l != p:
                    gt = dataloaders["val"].dataset.labels[index+i]
                    logger.write(f'{gt[1]},{gt[2]},{get_scripts(p)}\n')
            index += len(labels)

        if log_false:
            logger.close()

        overall, mean = calc_accuracy(result_mat)
        print(f'overall: {overall:.6f}')
        print(f'mean   : {mean:.6f}')

        return overall, result_mat


def main():
    config['scripts'] = args.scripts
    config['input_size'] = args.input_size

    if args.load_ckpt is None:
        model = get_model(args.model, len(get_scripts()), not args.scratch)
    else:
        ckpt = load_ckpt(os.path.join(config['ckpt_dir'], args.load_ckpt))
        config['scripts'] = args.scripts
        config['input_size'] = args.input_size
        model = get_model(ckpt['params']['model'], len(get_scripts()), False)

    if args.steps is not None and args.steps[0] != 'auto':
        args.steps = [round(int(float(s)) / args.batch_size)
                      for s in args.steps]

    if args.ckpt_file is None:
        # looks like VGG16_224_plain_ic17_Nov-1-2_43_56PM
        args.ckpt_file = f'{args.model}_{args.input_size}_{args.transforms[0]}_{args.scripts}_{start_date}'

    if len(args.transforms) == 1:
        train_transform = val_transform = args.transforms[0]
    elif len(args.transforms) == 2:
        train_transform, val_transform = args.transforms

    config['transform']['train'] = train_transform
    config['transform']['val'] = val_transform

    if args.load_ckpt:
        model.load_state_dict(ckpt['model'], strict=False)

        model.cls[-3] = nn.Conv2d(256, len(get_scripts()), 1)

    model = model.cuda()
    model = nn.DataParallel(
        model, device_ids=[i - min(args.gpu_ids) for i in args.gpu_ids])

    save_ckpt(config['ckpt_dir'], model, args, {
        'accu': 0, 'map': []}, file='test_ckpt.pt')

    print_network_info(args)

    dataloaders = {}
    print('------------------------------')
    trainset = TrainvalSet(get_combined_npy(args.gt_files), config['data_dir'], get_scripts(),
                           get_transforms(train_transform))
    dataloaders['train'] = DataLoader(trainset, args.batch_size, shuffle=True,
                                          num_workers=config['num_workers'])

    valset = TrainvalSet(get_combined_npy(args.val_files), config['data_dir'], get_scripts(),
                         get_transforms(val_transform))
    dataloaders['val'] = DataLoader(
        valset, args.batch_size, False, num_workers=config['num_workers'])

    print(f'trainset size: {len(trainset)}')
    print(f'valset size  : {len(valset)}')
    print(f'total size   : {len(trainset) + len(valset)}')
    print(
        f'batch size: {args.batch_size}, 1 epoch = {math.ceil(len(trainset)/args.batch_size)} iters')
    if args.weighted:
        print(f'weights: {get_loss_weights(args.weighted)}')

    print('========================================')

    train_model(model, dataloaders)


if __name__ == "__main__":
    main()
