import platform
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='caltech', type=str, choices=['cityperson', 'caltech'],
                    help='dataset name')
parser.add_argument('--gpu', default=[2], type=int, nargs='+', help='GPU id to use.')
parser.add_argument('--run_number', default='1', type=str, help='run number')
parser.add_argument('--NNL', action='store_true', default=True, help='NNL')
parser.add_argument('--neck', default='mulbn', type=str, choices=['no', 'res', 'mulbn'], help='加多bn模块或者残差块')
parser.add_argument('--neck_depth', default=4, type=int, help='加多bn模块或者残差块的数量')
parser.add_argument('--loss_weight', default=[1, 1, 1], type=float, nargs='+', help='lamda_center, lamda_size, lamda_offset')
parser.add_argument('--mixed_precision_training', action='store_true', default=False)
parser.add_argument('--tensorrt', action='store_true', default=False, help='tensorrt加速')
parser.add_argument('--filt_weight', action='store_true', default=False, help='filt weight')
args = parser.parse_args()

filt_weight_alpha = (0.001,)
filt_weight_beta = (1, -0.999)
filt_weight_fs = 1

model_name = 'CSP'
num_workers = 8
pin_memory = True
cudnn_benchmark = False
cudnn_deterministic = True
down_factor = 8
resume_epochs = True

train_print_freq = 5
optm = 'adam'
init_lr = 1  # 0.04 for voc-5
weight_decay = 0.000
mul_size_train = None  # (288, 352)
FLOPs = True
tensorboard = False
draw_graph = False
seed = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)[1:-1]



if args.dataset == 'caltech':
    from config_caltech import *
elif args.dataset == 'cityperson':
    from config_cityperson import *
