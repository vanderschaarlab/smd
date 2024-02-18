import os
import argparse
import json
import codecs

import torch
from utils import Trainer
from nn import UNet
from models import GaussianDiffusion


parser = argparse.ArgumentParser()

parser.add_argument('--in_memory', action='store_true', default=False)
parser.add_argument('--latent_denoising', action='store_true', default=False)
parser.add_argument('--fid_test', action='store_true', default=False)
parser.add_argument('--learnable_embeddings', action='store_true', default=False)

parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--vae_dir', type=str, default=None)
parser.add_argument('--data_type', type=str, required=True, choices=['vec', 'image', 'latent'])
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--channel_dim', type=int, default=3)
parser.add_argument('--scaling_factor', type=float, default=0.18215)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--training_steps', type=int, default=100000)
parser.add_argument('--save_every', type=int, default=10000)
parser.add_argument('--test_bsz', type=int, default=10000)
parser.add_argument('--demo_samples', type=int, default=10000)
parser.add_argument('--test_samples', type=int, default=5000)
parser.add_argument('--worker_procs', type=int, default=10)

parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--denoising_iters', type=int, default=1000)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--dim_mults', type=int, default=1)
parser.add_argument('--latent_mults', type=str, default=1)
parser.add_argument('--multi_nums', type=str, default=None)
parser.add_argument('--resnet_groups', type=int, default=8)
parser.add_argument('--time_dim', type=int, default=32)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--head_dim', type=int, default=64)
parser.add_argument('--exp_scale', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--hypernet_layers', type=int, default=2)

args = parser.parse_args()
print(json.dumps(args.__dict__, indent=True, ensure_ascii=False), end='\n\n')

dim_multipliers = tuple([2**i for i in range(0, args.dim_mults)]) \
    if (args.multi_nums is None) else \
    [int(i) for i in args.multi_nums.split(',')]
model = UNet(
    dim=args.hidden_dim, 
    channels=args.channel_dim, 
    flash_attn=False,
    dim_mults=dim_multipliers, 
    full_attn=(False,) * len(dim_multipliers), 
    latent_dim=(args.latent_dim if args.latent_denoising else 0),
    hypernet_layers=(args.hypernet_layers if args.latent_denoising else None),
    resnet_block_groups=args.resnet_groups,
    learned_sinusoidal_dim=args.time_dim, 
    learned_sinusoidal_cond=args.learnable_embeddings,
    attn_dim_head=args.head_dim, 
    attn_heads=args.attn_heads
)
red_scale = 2**args.exp_scale
identifier = None if not args.latent_denoising else UNet(
    dim=args.hidden_dim // red_scale, 
    channels=args.channel_dim * 2, 
    flash_attn=False,
    dim_mults=[int(i) for i in args.latent_mults.split(',')], 
    full_attn=(False,) * len(args.latent_mults.split(',')),
    out_dim=args.latent_dim, 
    resnet_block_groups=args.resnet_groups // red_scale, 
    learned_sinusoidal_dim=args.time_dim // red_scale, 
    learned_sinusoidal_cond=args.learnable_embeddings,
    attn_dim_head=args.head_dim // red_scale, 
    attn_heads=args.attn_heads // red_scale
)
diffusion = GaussianDiffusion(
    model=model, 
    identifier=identifier, 
    image_size=(1 if args.data_type == 'vec' else args.img_size), 
    timesteps=args.denoising_iters, 
    auto_normalize=(True if args.data_type == 'image' else False)
)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
config_path = os.path.join(args.save_dir, 'config.json')
with codecs.open(config_path, 'w', 'utf-8') as fw:
    json.dump(args.__dict__, fw, indent=True, ensure_ascii=False)

trainer = Trainer(
    diffusion_model=diffusion, 
    folder=args.data_dir, 
    results_folder=args.save_dir,
    train_batch_size=args.batch_size, 
    test_batch_size=args.test_bsz, 
    train_lr=args.learning_rate,
    train_num_steps=args.training_steps, 
    save_and_sample_every=args.save_every, 
    num_samples=args.demo_samples, 
    calculate_fid=(True if args.data_type == 'image' else False) & args.fid_test,
    num_fid_samples=args.test_samples, 
    data_type=args.data_type, 
    io_workers=args.worker_procs,
    in_memory=args.in_memory,
    pretrained_vae_dir=args.vae_dir, 
    scaling_factor=args.scaling_factor,
)
trainer.train()
