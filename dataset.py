import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import os


class TrainvalSet(Dataset):
    """
    load the .npy label of the dataset

        array looks like
                      [[filename1, label1],
                       [filename2, label2],
                                    ...
                       [filenamex, labelx]]
    """

    def __init__(self, gt, image_root, scripts, transform):
        if isinstance(gt, str):
            self.labels = np.load(gt)
        elif isinstance(gt, np.ndarray):
            self.labels = gt
        self.image_root = Path(image_root)
        self.transform = transform
        self.scripts = np.array(scripts)

    def __len__(self):
        return len(self.labels)

    def getlabel(self, index):
        _, label = self.labels[index]
        return np.where(self.scripts == label)[0].item()

    def __getitem__(self, index):
        filename, _ = self.labels[index]
        im = Image.open(self.image_root / filename)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im = self.transform(im)
        gt = torch.tensor(self.getlabel(index)).long()
        return im, gt


class TestSet(Dataset):

    def __init__(self, path, transform, ext='*.png', rglob=False):
        path = Path(path)
        if not rglob:
            self.im_list = list(path.glob(ext))
        else:
            self.im_list = list(path.rglob(ext))
        self.im_list = sorted(self.im_list, key=lambda f: int(f.stem[5:]))
        self.transform = transform

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im = Image.open(self.im_list[index])
        if im.mode != 'RGB':
            im = im.convert('RGB')
        return self.transform(im)

class DataPrefetcher():
    """
    https://gist.github.com/xhchrn/45585e33c4f1f18864309221eda2f046
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data_1, self.next_data_2 = next(self.loader)
        except StopIteration:
            self.next_data_1 = None
            self.next_data_2 = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data_1 = self.next_data_1.cuda(non_blocking=True)
            self.next_data_2 = self.next_data_2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_1, data_2 = self.next_data_1, self.next_data_2
        self.preload()
        return data_1, data_2
