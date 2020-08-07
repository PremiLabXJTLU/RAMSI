import argparse

parser = argparse.ArgumentParser()
# =========================
# * regular
# =========================
group0 = parser.add_argument_group(title='Relugar parameters')
group0.add_argument('-G', '--gpu-ids', type=int, nargs='*',
                    help='indicate GPU ids to be used')
group0.add_argument('-g', '--gpu-count', type=int,
                    default=1, help='how many GPU to use')
group0.add_argument('--ckpt-file', help='save checkpoint as')
group0.add_argument('-bo', '--best-only', default=True, help='only save the ckpt with the best accu')
# =========================
# * hyper-parameters
# =========================
group1 = parser.add_argument_group(title='Hyper-parameters')
group1.add_argument('-B', '--batch-size', type=int, default=24)
group1.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate')
group1.add_argument('--steps', nargs='*',
                    default=['4e5'], help='learning rate decay steps')
group1.add_argument('--weighted', help='weights of cross entropy loss')
# =========================
# * data
# =========================
group2 = parser.add_argument_group(title='Data related')
group2.add_argument('-I', '--input-size', type=int, default=224,
                    help='Input image height')
group2.add_argument('--scripts', default='ic17', help='script class')
group2.add_argument('--transforms', nargs='*',
                    default=['P+'], help='transforms of train and val dataset')
group2.add_argument('--gt-files', nargs='+',
                    default=['combine'], help='training set')
group2.add_argument('--val-files', nargs='*',
                    default=['ic17val'], help='validation set')
# =========================
# * model
# =========================
group3 = parser.add_argument_group(title='Network related')
group3.add_argument('--model', help='network structure')
group3.add_argument('--scratch', action='store_true',
                    help='train from scratch; pretrained on imagenet by default')
group3.add_argument('-l', '--load-ckpt', help='load checkpoint from')
group3.add_argument('--loss', default='ce', help='loss function')
group3.add_argument('--optim', default='sgd', help='optimizer')

args = parser.parse_args()
