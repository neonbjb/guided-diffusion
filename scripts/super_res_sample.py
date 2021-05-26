"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import random

import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import _list_image_files_recursively, ImageDataset
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #dist_util.setup_dist()
    torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12345', world_size=1, rank=0)
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.data_dir, args.batch_size, args.small_size)

    logger.log("creating samples...")
    all_images = []
    i = 1
    while len(all_images) * args.batch_size < args.num_samples:
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs={'low_res': next(data)[0].to('cuda')}
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_samples = th.cat(gathered_samples, dim=0)
        torchvision.utils.save_image((gathered_samples+1)/2, os.path.join(logger.get_dir(), f'{i}.png'))
        i += 1
        logger.log(f"created {len(gathered_samples) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(data_dir, batch_size, small_image_size):
    all_files = _list_image_files_recursively(data_dir)
    dataset = ImageDataset(resolution=small_image_size, image_paths=all_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return iter(dataloader)



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
