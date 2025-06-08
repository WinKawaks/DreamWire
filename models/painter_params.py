import random
import os
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from svgpathtools import svg2paths, Path, wsvg


class Painter(torch.nn.Module):
    def __init__(self, args,
                num_strokes=4,
                num_segments=4,
                imsize=512,
                device=None,
                target_im=None,
                mask=None,):
        super(Painter, self).__init__()

        self.args = args
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        self._3D_points = []
        self.shapes_all = []
        self.camera_in_matrix = torch.tensor([[35, 0, 0, 0], [0, 35, 0, 0], [0, 0, 1, 0]]).float()

        self.shape_groups = []
        self.device = device
        self.cube_size = imsize
        self.points_vars = []
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention

        self.text_target = args.text_target # for clip gradients
        if target_im is not None:
            self.define_attention_input(target_im)
        self.mask = mask
        background_path = None if args.sketches_edit == "none" else args.sketches_edit
        self.background = []
        if background_path is not None:
            path_split = background_path.split('|')
            for path in path_split:
                background = Image.open(path).convert("RGB").resize((self.cube_size, self.cube_size))
                background = transforms.ToTensor()(background)
                background = background.permute(1, 2, 0)
                self.background.append(background)

        self.strokes_counter = 0 # counts the number of calls to "get_path"
        self.epoch = 0
        self.final_epoch = args.num_iter - 1


    def init_image(self, original_shape=None, args=None, camera=None):
        if len(self.shapes_all) == 0:
            default_views = 3
            if args.text_prompt is not None:
                text_split = args.text_prompt.split("|")
                default_views = len(text_split)

            views = default_views if camera is None else camera.shape[0]
            for i in range(views):
                self.shapes_all.append([])

        for i in range(self.num_paths):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            if original_shape is not None:
                paths, points = self.get_path(original_path=original_shape[i].to(args.device), camera=camera)
            else:
                paths, points = self.get_path(args=args, index=i, camera=camera)
            if len(self._3D_points) < self.num_paths:
                self._3D_points.append(points)
            if len(self.shapes_all[0]) == self.num_paths:
                for shape in self.shapes_all:
                    shape.clear()
            for shape, path in zip(self.shapes_all, paths):
                shape.append(path)

            if len(self.shape_groups) < self.num_paths:
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes_all[0]) - 1]),
                                                    fill_color=None,
                                                    stroke_color=stroke_color)
                self.shape_groups.append(path_group)

        self.optimize_flag = [True for i in range(len(self.shapes_all[0]))]

        return self.get_image()
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_image(self):
        imgs = self.render_warp()
        rasters = []
        for img in imgs:
            opacity = img[:, :, 3:4]
            img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
            rasters.append(img)
        return rasters

    def get_path(self, original_path=None, args=None, index=None, camera=None):
        if original_path is None:
            points = []
            self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
            if args.bbox != 'none' and args.bbox is not None:  # bbox is in format [x1, y1, z1, x2, y2, z2]
                bbox = args.bbox.split(',')
                p0 = (float(bbox[0]) + (float(bbox[3]) - float(bbox[0])) * random.random(), float(bbox[1]) + (float(bbox[4]) - float(bbox[1])) * random.random(), float(bbox[2]) + (float(bbox[5]) - float(bbox[2])) * random.random())
            elif args.init_point != 'none' and args.init_point is not None:
                init_point = args.init_point.split(',')
                try:
                    p0 = (float(init_point[index * 3]), float(init_point[index * 3 + 1]), float(init_point[index * 3 + 2]))
                except IndexError:
                    p0 = (random.random(), random.random(), random.random())
            else:
                p0 = (random.random() * 0.6 + 0.2, random.random() * 0.6 + 0.2, random.random() * 0.6 + 0.2)
            points.append(p0)

            for j in range(self.num_segments):
                radius = 0.05
                for k in range(self.control_points_per_seg - 1):
                    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5), p0[2] + radius * (random.random() - 0.5))
                    points.append(p1)
                    p0 = p1
            points = torch.tensor(points).to(self.device)  # [0, 1]
            points = points * 2 - 1  # [-1, 1]
            if camera is None:
                points[:, 0] *= (self.cube_size / 2)  # [-256, 256]
                points[:, 1] *= (self.cube_size / 2)
                points[:, 2] *= (self.cube_size / 2)
        else:
            self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
            points = original_path

        points.requires_grad = True
        if camera is None:
            points_offset = points + (self.cube_size / 2)
            results = []
            path_x = pydiffvg.Path(num_control_points=self.num_control_points,
                                 points=points_offset[:, 1:].contiguous(),
                                 stroke_width=torch.tensor(self.width),
                                 is_closed=False)
            path_y = pydiffvg.Path(num_control_points=self.num_control_points,
                                 points=points_offset[:, [0, 2]].contiguous(),
                                 stroke_width=torch.tensor(self.width),
                                 is_closed=False)
            if self.args.z_inverse:
                w = torch.tensor([1, -1]).to(self.device)
                b = torch.tensor([0, self.cube_size]).to(self.device)
                z_inverse = w * points_offset[:, :2].contiguous() + b
                path_z = pydiffvg.Path(num_control_points=self.num_control_points,
                                        # points=points_offset[:, :2].contiguous(),
                                        points=z_inverse,
                                        stroke_width=torch.tensor(self.width),
                                        is_closed=False)
            else:
                path_z = pydiffvg.Path(num_control_points=self.num_control_points,
                                        points=points_offset[:, :2].contiguous(),
                                        # points=z_inverse,
                                        stroke_width=torch.tensor(self.width),
                                        is_closed=False)
            if len(self.shapes_all) == 1:
                results.append(path_x)
            elif len(self.shapes_all) == 2:
                results.append(path_x)
                results.append(path_y)
            elif len(self.shapes_all) == 3:
                results.append(path_x)
                results.append(path_y)
                results.append(path_z)
            self.strokes_counter += 1
            return results, points
        else:
            camera = camera.to(self.device)  # (Num_Views, 4, 4)
            results = []
            world_points = torch.cat((points, torch.ones(points.shape[0], 1, device=self.device)), dim=1)  # (Num_Points, 3) -> (Num_Points, 4)
            for i in range(camera.shape[0]):
                camera_points = torch.mm(camera[i], world_points.t())  # (4, 4) x (4, Num_Points) -> (4, Num_Points)
                Zc = camera_points[2, :].contiguous()  # (Num_Points)
                _2D_points = torch.mm(self.camera_in_matrix.to(self.device), camera_points) / Zc  # (3, 4) x (4, Num_Points) -> (3, Num_Points)
                _2D_points_offset = _2D_points + (self.cube_size / 2)
                path = pydiffvg.Path(num_control_points=self.num_control_points,
                                     points=_2D_points_offset[:2, :].t().contiguous(),
                                     stroke_width=torch.tensor(self.width),
                                     is_closed=False)
                results.append(path)
            self.strokes_counter += 1
            return results, points

    @torch.no_grad()
    def points_restrict(self, args):
        self.save_svg(f'{args.output_dir}', 'temp')
        paths, attributes, svg_attributes = svg2paths(f'{args.output_dir}/temp_0.svg', return_svg_attributes=True)

        new_paths = []
        for i, (path, attribute) in enumerate(zip(paths, attributes)):
            # print("Path {}: {}".format(i, path.length()))
            if path.length() < 120:
                new_paths.append(i)

        for i in new_paths:
            self.update_path(i, args)

        os.remove(f'{args.output_dir}/temp_0.svg')

    # update self.shape
    @torch.no_grad()
    def update_path(self, index, args):
        paths, points = self.get_path(args=args, index=index)
        for shapes, path in zip(self.shapes_all, paths):
            shapes[index] = path
        self._3D_points[index] = points

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.cube_size, self.cube_size)
                for shapes in self.shapes_all:
                    for path in shapes:
                        path.points.data.add_(eps * torch.randn_like(path.points))

        results = []
        for shapes in self.shapes_all:
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.cube_size, self.cube_size, shapes, self.shape_groups)

            img = _render(self.cube_size,  # width
                          self.cube_size,  # height
                          2,  # num_samples_x
                          2,  # num_samples_y
                          0,  # seed
                          None,  # background
                          *scene_args)
            results.append(img)
        return results

    def parameters(self):
        self.points_vars = []
        # storkes' location optimization
        for i, points in enumerate(self._3D_points):
            if self.optimize_flag[i]:
                points.requires_grad = True
                self.points_vars.append(points)
        return self.points_vars

    def get_start_end_points(self):
        points_vars = []
        for i, points in enumerate(self._3D_points):
            if self.optimize_flag[i]:
                points_vars.append(points[[0, -1]].contiguous())
        points_vars = torch.stack(points_vars)
        points_vars /= (self.cube_size / 2)  # [-1, 1]
        return points_vars

    def get_points_parans(self):
        return self.points_vars

    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def get_color_parameters(self):
        return self.color_vars

    def save_svg(self, output_dir, name):
        for i, shapes in enumerate(self.shapes_all):
            pydiffvg.save_svg('{}/{}_{}.svg'.format(output_dir, name, i), self.cube_size, self.cube_size, shapes, self.shape_groups)

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations

class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.args = args
        self.optim_color = args.force_sparse

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr, eps=1e-5)
        if self.args.train_with_diffusion:
            self.tree_points_optim = torch.optim.SGD(self.renderer.parameters(), lr=self.points_lr, momentum=0)
        if self.optim_color:
            self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr, eps=1e-5)

    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr

    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()

    def tree_zero_grad_(self):
        if self.args.train_with_diffusion:
            self.tree_points_optim.zero_grad()

    def step_(self, scaler=None):
        if scaler is not None:
            scaler.step(self.points_optim)
        else:
            self.points_optim.step()
        if self.optim_color:
            self.color_optim.step()

    def tree_step_(self, scaler=None):
        if self.args.train_with_diffusion:
            if scaler is not None:
                scaler.step(self.tree_points_optim)
            else:
                self.tree_points_optim.step()

    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

def interpret(args, image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=args.image_scale, mode='bicubic')
    image_relevance = image_relevance.reshape(args.image_scale, args.image_scale).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
