import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from config import config
from utils.utils import load_ckpt, get_scripts, get_model, get_transforms
from dataset import TestSet
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import json
import warnings


parser = argparse.ArgumentParser()
parser.add_argument('--test-path', help='test set path')
parser.add_argument('--ckpt', help='load checkpoint')
parser.add_argument('--output', help='output txt file')
parser.add_argument('--gpu-ids', type=int, default=[0], nargs='+')
parser.add_argument('--batch-size', type=int, default=24)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
warnings.simplefilter('ignore')


def predict(model, dataloader):
    model.eval()

    print('predicting')

    scripts = get_scripts()

    results = []
    details = []

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            results.extend(preds.cpu().numpy().tolist())
            details.extend(outputs.cpu().detach().numpy().tolist())

    print('writing outputs')
    output_file = Path(args.output)
    f = output_file.open('w', encoding='utf-8')
    for filename, result, detail in zip(testset.im_list, results, details):
        if not isinstance(filename, Path):
            filename = Path(filename)
        f.write(f'{filename.name},{scripts[result]}\n')
    f.close()
    print(f'finish. writen as {args.output}')


if __name__ == "__main__":
    ckpt = load_ckpt(os.path.join(config['ckpt_dir'], args.ckpt))

    if not args.output:
        args.output = f"{ckpt['params']['model']}_{ckpt['config']['transform']['val']}_{ckpt['config']['input_size']}.txt"

    transform = get_transforms(ckpt['config']['transform']['val'])
    print(f'transform: {ckpt["config"]["transform"]["val"]}')
    print(f'scripts: {get_scripts()}')

    testset = TestSet(args.test_path, transform)
    print(f'testset: {args.test_path}')

    dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=config['num_workers'])

    num_classes = len(get_scripts())
    model = get_model(ckpt['params']['model'], num_classes, False)

    model = model.cuda()
    model.load_state_dict(ckpt['model'])
    model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_ids))))

    predict(model, dataloader)
