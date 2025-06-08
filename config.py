import argparse
import os
import random

import numpy as np
import pydiffvg
import torch
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_arguments():
    parser = argparse.ArgumentParser()
    # =================================
    # ============ general ============
    # =================================
    parser.add_argument("--target", type=str, help="target image path")
    parser.add_argument("--output_name", type=str,
                        help="target image file, located in <target_images>")
    parser.add_argument("--output_dir", type=str,
                        help="directory to save the output images and loss")
    parser.add_argument("--path_svg", type=str, default="none",
                        help="if you want to load an svg file and train from it")
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_object", type=int, default=0)
    parser.add_argument("--fix_scale", type=int, default=0)
    parser.add_argument("--display_logs", type=int, default=0)
    parser.add_argument("--display", type=int, default=0)

    # =================================
    # ============ wandb ============
    # =================================
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="winkawaks")
    parser.add_argument("--wandb_name", type=str, default="test")
    parser.add_argument("--wandb_project_name", type=str, default="DreamWire")

    # =================================
    # =========== training ============
    # =================================
    parser.add_argument("--num_iter", type=int, default=2000,
                        help="number of optimization iterations")
    parser.add_argument("--num_stages", type=int, default=0,
                        help="training stages, you can train x strokes, then freeze them and train another x strokes etc.")
    parser.add_argument("--lr_scheduler", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--color_lr", type=float, default=0.01)
    parser.add_argument("--color_vars_threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="for optimization it's only one image")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--update_interval", type=int, default=50)
    parser.add_argument("--image_scale", type=int, default=512)

    # =================================
    # ======== strokes params =========
    # =================================
    parser.add_argument("--num_paths", type=int,
                        default=16, help="number of strokes")
    parser.add_argument("--width", type=float,
                        default=3, help="stroke width")
    parser.add_argument("--control_points_per_seg", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=5,
                        help="number of segments for each stroke, each stroke is a bezier curve with 4 control points")

    parser.add_argument("--xdog_intersec", type=int, default=1)
    parser.add_argument("--mask_object_attention", type=int, default=0)
    parser.add_argument("--softmax_temp", type=float, default=0.3)

    # =================================
    # ============= loss ==============
    # =================================
    parser.add_argument("--percep_loss", type=str, default="none",
                        help="the type of perceptual loss to be used (L2/LPIPS/none)")
    parser.add_argument("--perceptual_weight", type=float, default=0,
                        help="weight the perceptual loss")
    parser.add_argument("--num_aug_clip", type=int, default=3)

    parser.add_argument("--include_target_in_aug", type=int, default=0)
    parser.add_argument("--augment_both", type=int, default=1,
                        help="if you want to apply the affine augmentation to both the sketch and image")
    parser.add_argument("--augemntations", type=str, default="affine",
                        help="can be any combination of: 'affine_noise_eraserchunks_eraser_press'")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--aug_scale_min", type=float, default=0.7)
    parser.add_argument("--force_sparse", type=float, default=0,
                        help="if True, use L1 regularization on stroke's opacity to encourage small number of strokes")

    parser.add_argument("--text_target", type=str, default="none")
    parser.add_argument("--points_init", type=str, default="none")
    parser.add_argument("--sketches_edit", type=str, default='none')

    parser.add_argument('--train_with_diffusion', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--control', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default='None', help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--vram_O', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--text_prompt', type=str, default=None)
    parser.add_argument('--text_negative_prompt', type=str, default='')
    parser.add_argument('--bbox', type=str, default='none')
    parser.add_argument('--init_point', type=str, default='none')
    parser.add_argument("--length_weight", type=float, default=0)
    parser.add_argument("--z_inverse", type=float, default=0)

    args = parser.parse_args()
    set_seed(args.seed)


    abs_path = '/home/DreamWire'

    import subprocess as sp
    if not os.path.isfile(f"{abs_path}/U2Net_/saved_models/u2net.pth"):
        sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
               "-O", "U2Net_/saved_models/"])

    test_name = args.output_name
    args.wandb_name = f"{args.output_name}_{args.num_paths}strokes_seed{args.seed}"

    args.output_dir = os.path.join(args.output_dir, args.wandb_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    jpg_logs_dir = f"{args.output_dir}/jpg_logs"
    svg_logs_dir = f"{args.output_dir}/svg_logs"
    if not os.path.exists(jpg_logs_dir):
        os.mkdir(jpg_logs_dir)
    if not os.path.exists(svg_logs_dir):
        os.mkdir(svg_logs_dir)

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                   config=args, name=f'{args.wandb_name}_seg{args.num_segments}_{args.length_weight}', id=wandb.util.generate_id())

    if args.use_gpu:
        args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    else:
        args.device = torch.device("cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available() and args.use_gpu)
    pydiffvg.set_device(args.device)
    return args


if __name__ == "__main__":
    # for cog predict
    args = parse_arguments()
    final_config = vars(args)
    np.save(f"{args.output_dir}/config_init.npy", final_config)
