import sys
import argparse
import gc
import logging
import os

# print("Current Working Directory " , os.getcwd())
# sys.path.append(os.getcwd())

# sys.path.append("/scratch/sz2257/sgan")
sys.path.append("../")
import time
import json
# import yaml

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryDiscriminator
from cnn.cnn_models import CNNTrajectoryGenerator

from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path


torch.backends.cudnn.benchmark = True
writer = SummaryWriter()

time_str="_".join(writer.get_logdir().split("/")[1].split("_")[:2])
# output_dir="/media/felicia/Data/sgan_results/{}".format(time_str)

output_dir="/scratch/sz2257/sgan/sgan_results/{}".format(time_str)

# data_dir='/media/felicia/Data/basketball-partial'
data_dir='/scratch/sz2257/sgan/basketball-partial'

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='01.02.2016.PHX.at.SAC.new', type=str) #default:zara1
parser.add_argument('--dataset_dir', default=data_dir, type=str)
parser.add_argument('--image_path', default="./images/cross.05.pt", type=str)
parser.add_argument('--delim', default=',') #default: ' '
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--metric', default="meter", type=str)
parser.add_argument("--model", default="baseline", type=str)
# Optimization
parser.add_argument('--batch_size', default=128, type=int) #32
parser.add_argument('--num_iterations', default=20000, type=int) #default:10000
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int) #64
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag) #default:0-bool_flag
parser.add_argument('--mlp_dim', default=64, type=int) #default: 1024
parser.add_argument('--interaction_activation', default="none", type=str)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int) #default:64
parser.add_argument('--decoder_h_dim_g', default=32, type=int) #default:128
parser.add_argument('--noise_dim', default=(8,), type=int_tuple) # default: None-int_tuple
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global') #default:pred
parser.add_argument('--clipping_threshold_g', default=1.5, type=float) #default:0
parser.add_argument('--g_learning_rate', default=1e-3, type=float) #default:5e-4,0.001
parser.add_argument('--g_steps', default=1, type=int)
parser.add_argument('--g_gamma', default=0.8, type=float) #default:5e-4, 0.001
# Discriminator Options
parser.add_argument('--d_type', default='local', type=str) #default:'local'
parser.add_argument('--encoder_h_dim_d', default=64, type=int) #default:64
parser.add_argument('--d_learning_rate', default=1e-3, type=float) #default:5e-4, 0.001
parser.add_argument('--d_steps', default=2, type=int) #default:2
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--d_activation', default='relu', type=str) # 'relu'
parser.add_argument('--d_gamma', default=0.8, type=float) #default:5e-4, 0.001

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net') #default:'pool_net'
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag) #default:1-bool_flag

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int) # 1024

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float) #default:0->1
parser.add_argument('--best_k', default=10, type=int) #default:1
parser.add_argument('--l2_loss_mode', default="raw", type=str) #default:"raw"


# Output
parser.add_argument('--output_dir', default=output_dir) # os.getcwd()
parser.add_argument('--print_every', default=10, type=int) #default:5
parser.add_argument('--checkpoint_every', default=50, type=int) #default:100
parser.add_argument('--checkpoint_name', default='basketball_phx_sac')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int) #default:1
parser.add_argument('--num_samples_check', default=5000, type=int)
parser.add_argument("--tb_path", default=writer.get_logdir(), type=str)

# Misc
parser.add_argument('--use_gpu', default=1, type=int) # 1: use_gpu
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    # train_path = get_dset_path(args.dataset_name, 'train')
    # val_path = get_dset_path(args.dataset_name, 'val')
    train_path= os.path.join(args.dataset_dir,args.dataset_name,'train_sample') # 10 files:0-9
    val_path= os.path.join(args.dataset_dir,args.dataset_name,'val_sample') # 5 files: 10-14

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)
    torch.save(
        {
            "train_image_channels": train_loader.image_channels,
            "valid_image_channels": val_loader.image_channels,
        },
        args.image_path
    )



if __name__ == '__main__':
    args = parser.parse_args()
    log_path="{}/config.txt".format(writer.get_logdir())
    with open(log_path,"a") as f:
        json.dump(args.__dict__,f,indent=2)

    main(args)


