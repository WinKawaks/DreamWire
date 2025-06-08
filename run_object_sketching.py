import sys
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import multiprocessing as mp
import os
import subprocess as sp
from shutil import copyfile

import numpy as np
import torch
from IPython.display import Image as Image_colab
from IPython.display import display, SVG, clear_output
from ipywidgets import IntSlider, Output, IntProgress, Button
import time

parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--output_name", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int, default=16,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--num_iter", type=int, default=2000,
                    help="number of iterations")
parser.add_argument("--num_segments", type=int, default=5)
parser.add_argument("--fix_scale", type=int, default=0,
                    help="if the target image is not squared, it is recommended to fix the scale")
parser.add_argument("--mask_object", type=int, default=0,
                    help="if the target image contains background, it's better to mask it out")
parser.add_argument("--num_sketches", type=int, default=3,
                    help="it is recommended to draw 3 sketches and automatically chose the best one")
parser.add_argument("--multiprocess", type=int, default=0,
                    help="recommended to use multiprocess if your computer has enough memory")

parser.add_argument('--train_with_diffusion', action='store_true')
parser.add_argument('--control', action='store_true')
parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
parser.add_argument('--fp16', action='store_true', help="use float16 for training")
parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
parser.add_argument('--text_prompt', type=str, default=None)
parser.add_argument('--text_negative_prompt', type=str, default='')
parser.add_argument("--attention_init", type=int, default=0,
                        help="if True, use the attention heads of Dino model to set the location of the initial strokes")
parser.add_argument("--sketches_edit", type=str, default='none')
parser.add_argument('--bbox', type=str, default='none')
parser.add_argument('--init_point', type=str, default='none')

parser.add_argument('-colab', action='store_true')
parser.add_argument('-cpu', action='store_true')
parser.add_argument('-display', action='store_true')
parser.add_argument("--points_init", type=str, default='none')
parser.add_argument("--image_scale", type=int, default=512)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--length_weight", type=float, default=0)
parser.add_argument("--num_aug_clip", type=int, default=3)

parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--use_wandb", type=int, default=1)
parser.add_argument("--z_inverse", type=int, default=0)


args = parser.parse_args()
# print(args)

multiprocess = not args.colab and args.num_sketches > 1 and args.multiprocess

abs_path = '/home/DreamWire'

target = f"{args.target_file}"

test_name = f'{args.output_name}_{args.length_weight}'
output_dir = f"{abs_path}/output/{test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_iter = args.num_iter
save_interval = 50
use_gpu = not args.cpu

if not torch.cuda.is_available():
    use_gpu = False
    print("CUDA is not configured with GPU, running with CPU instead.")
    print("Note that this will be very slow, it is recommended to use colab.")

if args.colab:
    print("=" * 50)
    print(f"Processing [{args.target_file}] ...")
    if args.colab or args.display:
        img_ = Image_colab(target)
        display(img_)
        print(f"GPU: {use_gpu}, {torch.cuda.current_device()}")
    print(f"Results will be saved to \n[{output_dir}] ...")
    print("=" * 50)

start = args.seed_start
seeds = list(range(start, start + args.num_sketches * 1, 1))

exit_codes = []
manager = mp.Manager()
losses_all = manager.dict()


def run(seed, wandb_name):
    print(wandb_name)
    exit_code = sp.run(["python", f"{abs_path}/painterly_rendering.py",
                            "--target", target,
                            "--num_paths", str(args.num_strokes),
                            "--output_dir", output_dir,
                            "--output_name", args.output_name,
                            "--wandb_name", wandb_name,
                            "--num_iter", str(num_iter),
                            "--num_segments", str(args.num_segments),
                            "--save_interval", str(save_interval),
                            "--seed", str(seed),
                            "--use_gpu", str(int(use_gpu)),
                            "--fix_scale", str(args.fix_scale),
                            "--mask_object", str(args.mask_object),
                            "--mask_object_attention", str(args.mask_object),
                            "--display_logs", str(int(args.colab)),
                            "--display", str(int(args.display)),
                            "--points_init", str(args.points_init),
                            "--sketches_edit", str(args.sketches_edit),
                            "--attention_init", str(args.attention_init),
                            "--train_with_diffusion", str(int(args.train_with_diffusion)),
                            "--control", str(int(args.control)),
                            "--sd_version", str(args.sd_version),
                            "--hf_key", str(args.hf_key),
                            "--fp16", str(args.fp16),
                            "--vram_O", str(args.vram_O),
                            "--text_prompt", str(args.text_prompt),
                            "--text_negative_prompt", str(args.text_negative_prompt),
                            '--bbox', str(args.bbox),
                            '--init_point', str(args.init_point),
                            '--image_scale', str(int(args.image_scale)),
                            '--lr', str(args.lr),
                            '--length_weight', str(args.length_weight),
                            "--image_scale", str(args.image_scale),
                            "--num_aug_clip", str(args.num_aug_clip),
                            "--use_wandb", str(args.use_wandb),
                            "--z_inverse", str(args.z_inverse),
                        ])
    if exit_code.returncode:
        sys.exit(1)

    # config = np.load(f"{output_dir}/{wandb_name}/config.npy",
    #                  allow_pickle=True)[()]
    # loss_eval = np.array(config['loss_eval'])
    # inds = np.argsort(loss_eval)
    # losses_all[wandb_name] = loss_eval[inds][0]


def display_(seed, wandb_name):
    path_to_svg = f"{output_dir}/{wandb_name}/svg_logs/"
    intervals_ = list(range(0, num_iter, save_interval))
    filename = f"svg_iter0.svg"
    display(IntSlider())
    out = Output()
    display(out)
    for i in intervals_:
        filename = f"svg_iter{i}.svg"
        not_exist = True
        while not_exist:
            not_exist = not os.path.isfile(f"{path_to_svg}/{filename}")
            continue
        with out:
            clear_output()
            print("")
            display(IntProgress(
                        value=i,
                        min=0,
                        max=num_iter,
                        description='Processing:',
                        bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                        style={'bar_color': 'maroon'},
                        orientation='horizontal'
                    ))
            display(SVG(f"{path_to_svg}/svg_iter{i}.svg"))


if multiprocess:
    ncpus = 10
    P = mp.Pool(ncpus)  # Generate pool of workers

for seed in seeds:
    print(seed)
    wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
    if multiprocess:
        P.apply_async(run, (seed, wandb_name))
    else:
        run(seed, wandb_name)

if args.display and multiprocess:
    time.sleep(10)
    P.apply_async(display_, (0, f"{test_name}_{args.num_strokes}strokes_seed0"))

if multiprocess:
    P.close()
    P.join()  # start processes
    sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
    copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.svg", f"{output_dir}/{list(sorted_final.keys())[0]}_best.svg")
