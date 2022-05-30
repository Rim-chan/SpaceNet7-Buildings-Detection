import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg ("--seed", type=int, default=26012022, help="Random Seed")
    arg("--generator", default=torch.Generator().manual_seed(26012022), help='Train Validate Predict Seed')
    arg("--base_dir", type=str, default="LiveEO_ML_intern_challenge", help="Data Directory")
    arg ("--num_workers", type=int, default=2, help="DataLoader num_workers")
    arg ("--in_channels", type=int, default=3, help="#Input Channels")
    arg ("--out_channels", type=int, default=1, help="#Output Channels")
    arg ("--learning_rate", type=float, default=1e-4, help="Learning Rate")
    arg ("--weigh_decay", type=float, default=1e-5, help="Weigh Decay")
    arg ("--kernels",  default=[[3, 3]] * 5, help="Convolution Kernels")
    arg ("--strides",  default=[[1, 1]] +  [[2, 2]] * 4, help="Convolution Strides")

    arg("--num_epochs", type=int, default=20, help="Number of Epochs")
    arg("--exec_mode", type=str, default='train', help='Execution Mode')
    arg("--ckpt_path", type=str, default=None, help='Checkpoint Path')
    arg("--save_path", type=str, default='./', help='Saves Path')

    arg("--samples_per_epoch", type=int, default=1000, help="Random Samples Per Epoch")
    arg("--crop_size", type=int, default=480, help="centered crop size")
    arg("--batch_size", type=int, default=12, help="batch size")
    return parser.parse_args(args=[])




