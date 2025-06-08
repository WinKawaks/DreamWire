import copy
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import shutil
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (224, 224), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            224, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(224))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask

# 3 x 224 x 224
def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 90)
    im_min = np.percentile(grayscale_im, 10)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def format_np_output(np_arr):
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def prim(graph):
    n = len(graph)
    visited = [False] * n
    parent = [-1] * n
    key = [float('inf')] * n
    key[0] = 0

    for _ in range(n):
        u = min_key(key, visited)
        visited[u] = True
        for v in range(n):
            if graph[u][v] and not visited[v] and graph[u][v] < key[v]:
                parent[v] = u
                key[v] = graph[u][v]

    return parent

def min_key(key, visited):
    min_val = float('inf')
    min_idx = -1
    for i in range(len(key)):
        if key[i] < min_val and not visited[i]:
            min_val = key[i]
            min_idx = i
    return min_idx

def total_length(points, args):
    num = points.shape[0]
    distances = torch.zeros(num, num, 4).to(args.device)
    for i in range(num):
        for j in range(num):
            for k in range(2):
                for l in range(2):
                    diff = points[i, k] - points[j, l]
                    dist = torch.norm(diff, p=2)
                    # dist = torch.sum(diff * diff)
                    if dist == 0 and i != j:
                        dist = 0.01
                    distances[i, j, 2 * k + l] = dist
    min_dis = torch.min(distances, dim=2).values
    tree = prim(min_dis)
    loss = 0
    for i, node in enumerate(tree):
        if node != -1:
            loss += min_dis[i, node]
    return loss

def main(args):
    loss_func = Loss(args)
    loader = None
    renderer = load_renderer(args)

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}

    renderer.set_random_noise(0)
    original_shape = None
    if args.points_init != 'none':
        original_shape = torch.load(args.points_init)
    with torch.cuda.amp.autocast(enabled=False):
        img_start = renderer.init_image(original_shape=original_shape, args=args, camera=None)
    optimizer.init_optimizers()

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # not using tdqm for jupyter demo
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        # epoch_range = range(args.num_iter)
        epoch_range = tqdm(range(args.num_iter))
    length_loss = None
    final_length = 0

    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        # renderer.set_random_noise(0)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        grad_clip = 2 + 6 * min(1, counter / args.num_iter)
        # grad_clip = 0.002

        start = time.time()
        optimizer.zero_grad_()

        camera = None
        img = None
        if loader is not None:
            camera, img = next(iter(loader))
        with torch.cuda.amp.autocast(enabled=False):
            sketches = renderer.init_image(original_shape=renderer._3D_points, args=args, camera=camera)

        sketches = torch.cat(sketches, dim=0)

        if args.train_with_diffusion:
            with torch.cuda.amp.autocast(enabled=args.fp16):
                losses_dict = loss_func(sketches, img, renderer.get_color_parameters(), renderer, counter, optimizer, grad_clip=grad_clip)
                sds_loss = sum(list(losses_dict.values()))

            scaler.scale(sds_loss).backward()
            # assert(torch.isfinite(sss.grad).all())
            optimizer.step_(scaler)
            # optimizer.step_()
            scaler.update()

            with torch.cuda.amp.autocast(enabled=args.fp16):
                budget = total_length(renderer.get_start_end_points(), args)
                length_loss = budget * args.length_weight
            if args.length_weight == 0:
                length_loss = length_loss.detach()
            else:
                optimizer.tree_zero_grad_()
                scaler.scale(length_loss).backward()
                optimizer.tree_step_(scaler)
                scaler.update()
            temp_points = copy.deepcopy(renderer.get_points_parans())
        else:
            losses_dict = loss_func(sketches, img.detach(), renderer.get_color_parameters(), renderer, counter, optimizer, grad_clip=grad_clip)
            loss = sum(list(losses_dict.values()))
            loss.backward()
            optimizer.step_()
            temp_points = copy.deepcopy(renderer.get_points_parans())

        if length_loss is not None:
            final_length = budget.item()
            if args.use_wandb:
                wandb.log({'sds_loss': sds_loss.item(), 'length_loss': budget.item()}, step=counter)

        if epoch % args.save_interval == 0:
            utils.plot_batch(img, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            saved_points = []
            for point in temp_points:
                saved_points.append(point.data)
            torch.save(saved_points, f"{args.output_dir}/points_{epoch}.pt")

        if args.train_with_diffusion and epoch % args.update_interval == 0 and epoch != 0 and args.init_point == 'none':
            renderer.points_restrict(args)
            optimizer.init_optimizers()
            renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}_reset")

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)

        del sketches
        torch.cuda.empty_cache()

        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")
    saved_points = []
    for point in renderer.get_points_parans():
        saved_points.append(point.data)
    torch.save(saved_points, f"{args.output_dir}/points_final.pt")

    return configs_to_save, final_length

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save, final_length = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    np.save(f"{args.output_dir}/length.npy", final_length)
    if args.use_wandb:
        wandb.finish()
