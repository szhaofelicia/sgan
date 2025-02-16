import sys
import argparse
import gc
import logging
import os


sys.path.append("/scratch/sz2257/sgan")
# sys.path.append("/home/felicia/research/sgan")

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

from sgan.models import TrajectoryGenerator as GeneratorBaseline, TrajectoryDiscriminator as DiscriminatorBaseline
# from sgan.models_old import TrajectoryGenerator,  TrajectoryDiscriminator
from sgan.models_teampos import TrajectoryGenerator as TeamPosGenerator, TrajectoryDiscriminator as TeamPosDiscriminator

from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path


torch.backends.cudnn.benchmark = True
writer = SummaryWriter()

time_str="_".join(writer.get_logdir().split("/")[1].split("_")[:2])

# output_dir="/media/felicia/Data/sgan_results/{}".format(time_str)
output_dir="/scratch/sz2257/sgan_results/{}".format(time_str)

# data_dir='/media/felicia/Data/basketball-partial'
data_dir='/scratch/sz2257/basketball-partial'

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='01.02.2016.PHX.at.SAC.new', type=str) #default:zara1
parser.add_argument('--dataset_dir', default=data_dir, type=str)
parser.add_argument('--delim', default=',') #default: ' '
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--metric', default="foot", type=str) # Denote the original metric, dataset would convert it to meter unless --metric is original
parser.add_argument("--model", default="team_pos", type=str) # "baseline" or "team_pos"
parser.add_argument("--dset", default="dota", type=str) # "basketball","csgo","dota","nfl"
parser.add_argument("--trajD", default=2, type=int) # 2 or 3


# Optimization
parser.add_argument('--batch_size', default=128, type=int) #32
parser.add_argument('--num_iterations', default=20000, type=int) #default:10000
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int) #64
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--tp_dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag) #default:0-bool_flag
parser.add_argument('--mlp_dim', default=64, type=int) #default: 1024
parser.add_argument('--team_embedding_dim', default=16, type=int) #default: 1024
parser.add_argument('--pos_embedding_dim', default=32, type=int) #default: 1024
parser.add_argument('--interaction_activation', default="none", type=str) # none, attention,attentiontp

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
parser.add_argument('--d_gamma', default=0.8, type=float) #default:5e-4, 0.001
parser.add_argument('--d_steps', default=2, type=int) #default:2
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--d_activation', default='relu', type=str) # 'relu'


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


# MODELS = {
#     "baseline": (GeneratorBaseline, DiscriminatorBaseline),
#     "team_pos": (TeamPosGenerator, TeamPosDiscriminator)
# }


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
    train_path= os.path.join(args.dataset_dir,args.dataset_name,'train') # train or train_sample
    val_path= os.path.join(args.dataset_dir,args.dataset_name,'valid') # valid or val_sample

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )


    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        tp_dropout=args.tp_dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        team_embedding_dim=args.team_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        interaction_activation=args.interaction_activation,
        trajD=args.trajD
    )

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    generator = generator.cuda()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tp_dropout=args.tp_dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type,
        activation=args.d_activation, # default: relu,
        pos_embedding_dim=args.pos_embedding_dim,
        team_embedding_dim=args.team_embedding_dim,
        interaction_activation=args.interaction_activation,
        trajD=args.trajD
    )

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    discriminator = discriminator.cuda()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )
    scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[10, 50], gamma=args.g_gamma)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[10, 50], gamma=args.d_gamma)
    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        scheduler_g.step()
        scheduler_d.step()

        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    # logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    # logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

                ## log scalars
                for k, v in sorted(losses_d.items()):
                    writer.add_scalar("loss/{}".format(k), v, t)
                for k, v in sorted(losses_g.items()):
                    writer.add_scalar("loss/{}".format(k), v, t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    # logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    # logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                ## log scalars
                for k, v in sorted(metrics_val.items()):
                    writer.add_scalar("val/{}".format(k), v, t)
                for k, v in sorted(metrics_train.items()):
                    writer.add_scalar("train/{}".format(k), v, t)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                # checkpoint_path = os.path.join(
                #     args.output_dir, '{}_with_model_{:06d}.pt'.format(args.checkpoint_name,t)
                # )
                checkpoint_path = os.path.join(args.output_dir, '{}_with_model.pt'.format(args.checkpoint_name))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items

                # checkpoint_path = os.path.join(
                #     args.output_dir, '{}_no_model_{:06d}.pt' .format(args.checkpoint_name,t))

                checkpoint_path = os.path.join(args.output_dir, '{}_no_model.pt' .format(args.checkpoint_name))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


        ## log scalars
        # for k, v in sorted(losses_d.items()):
        #     writer.add_scalar("train/{}".format(k),v,epoch)
        # for k, v in sorted(losses_g.items()):
        #     writer.add_scalar("train/{}".format(k),v,epoch)

def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
     non_linear_ped, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    all_team_vec = torch.cat([obs_team_vec, pred_team_vec], dim=0)
    all_pos_vec = torch.cat([obs_pos_vec, pred_pos_vec], dim=0)
    scores_fake = discriminator(traj_fake, traj_fake_rel, all_team_vec, all_pos_vec, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, all_team_vec, all_pos_vec, seq_start_end)


    # Compute loss with optional gradient penalty
    # data_loss = d_loss_fn(scores_real, scores_fake)
    # losses['D_data_loss'] = data_loss.item()
    # loss += data_loss

    d_loss_real, d_loss_fak=d_loss_fn(scores_real, scores_fake)
    losses['D_real_loss'] = d_loss_real.item()
    losses['D_fake_loss'] = d_loss_fak.item()
    loss += d_loss_real+d_loss_fak

    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
     non_linear_ped, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec)


        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode=args.l2_loss_mode # default:"raw"
            ))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    all_team_vec = torch.cat([obs_team_vec, pred_team_vec], dim=0)
    all_pos_vec = torch.cat([obs_pos_vec, pred_pos_vec], dim=0)
    scores_fake = discriminator(traj_fake, traj_fake_rel, all_team_vec, all_pos_vec,seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            all_team_vec = torch.cat([obs_team_vec, pred_team_vec], dim=0)
            all_pos_vec = torch.cat([obs_pos_vec, pred_pos_vec], dim=0)
            scores_fake = discriminator(traj_fake, traj_fake_rel, all_team_vec, all_pos_vec,seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, all_team_vec, all_pos_vec, seq_start_end)

            # d_loss = d_loss_fn(scores_real, scores_fake)
            # d_losses.append(d_loss.item())

            d_loss_real, d_loss_fak = d_loss_fn(scores_real, scores_fake)
            d_loss=d_loss_real+d_loss_fak
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()

    MODELS = {
        "baseline": (GeneratorBaseline, DiscriminatorBaseline),
        "team_pos": (TeamPosGenerator, TeamPosDiscriminator)
    }

    TrajectoryGenerator, TrajectoryDiscriminator = MODELS[args.model]
    log_path="{}/config.txt".format(writer.get_logdir())
    with open(log_path,"a") as f:
        json.dump(args.__dict__,f,indent=2)

    # log_path="{}/config.yaml".format(writer.get_logdir())
    # with open(log_path,'w') as file:
    #     args_file=yaml.dump(args,file)
    # print(args_file)
    writer = SummaryWriter(args.tb_path)
    main(args)
    writer.flush()


