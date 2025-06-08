import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

from sd import StableDiffusion


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.percep_loss = args.percep_loss
        self.losses_to_apply = self.get_losses_to_apply()
        self.loss_mapper = {"diffusion": DiffusionLoss(args)}

    def get_losses_to_apply(self):
        losses_to_apply = []
        losses_to_apply.append("diffusion")
        return losses_to_apply

    def forward(self, sketches, targets, color_parameters, renderer, epoch, points_optim=None, grad_clip=None, mode="train"):
        loss = 0
        losses_dict = dict.fromkeys(
            self.losses_to_apply, torch.tensor([0.0]).to(self.args.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)

        for loss_name in self.losses_to_apply:
            if loss_name == "diffusion":
                losses_dict[loss_name] = self.loss_mapper[loss_name](sketches, as_latent=False, grad_clip=grad_clip)
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
            # loss = loss + self.loss_mapper[loss_name](sketches, targets).mean() * loss_coeffs[loss_name]

        for key in self.losses_to_apply:
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        return losses_dict

class DiffusionLoss(nn.Module):
    def __init__(self, args):
        super(DiffusionLoss, self).__init__()
        self.args = args
        if args.hf_key.lower() == 'none':
            args.hf_key = None
        self.model = StableDiffusion(args, args.fp16, args.vram_O, args.sd_version, args.hf_key)
        self.pad_prompt = ' on a white background'
        # self.pad_prompt = ''
        for p in self.model.parameters():
            p.requires_grad = False

        self.prepare_text_embeddings()
        self.prepare_image_control()
        self.NUM_AUGS = args.num_aug_clip
        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                512, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
            augemntations.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.augment_trans = transforms.Compose(augemntations)

    # calculate the text embs.
    # text xyz * N
    def prepare_text_embeddings(self):
        if self.args.text_prompt is None:
            self.text_z = None
            return
        self.args.text_prompt = self.args.text_prompt.replace("_", " ")
        text_split = self.args.text_prompt.split("|")
        for i, t in enumerate(text_split):
            text_split[i] = t + self.pad_prompt
        text_list = text_split * self.args.num_aug_clip
        self.text_z = self.model.get_text_embeds(text_list, [self.args.text_negative_prompt] * self.args.num_aug_clip * len(text_split))

    # condition xyz * N
    def prepare_image_control(self):
        background_path = None if self.args.sketches_edit == "none" else self.args.sketches_edit
        self.background = []
        if background_path is not None:
            paths = background_path.split("|")
            for path in paths:
                background = Image.open(path).convert("RGB").resize((512, 512))
                background = ImageOps.invert(background)
                self.background.append(background)

    # latent xyz * N
    def forward(self, sketches, as_latent=False, grad_clip=None, mode="train"):
        if mode == "eval":
            # for regular clip distance, no augmentations
            with torch.no_grad():
                loss = self.model.train_step(self.text_z, sketches * 2 - 1, as_latent=as_latent, grad_clip=grad_clip, image=self.background)
                return loss

        if self.NUM_AUGS == 1:
            sketch_batch = sketches * 2 - 1
        else:
            sketch_augs = []
            for n in range(self.NUM_AUGS):
                augmented_pair = self.augment_trans(sketches)
                sketch_augs.append(augmented_pair)

            sketch_batch = torch.cat(sketch_augs)

        loss = self.model.train_step(self.text_z, sketch_batch, as_latent=as_latent, grad_clip=grad_clip, image=self.background)
        return loss
